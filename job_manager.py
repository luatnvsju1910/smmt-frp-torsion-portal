"""
Background Job Manager for FRP RC Beam SMMT ML Dashboard.
Handles asynchronous process execution, PID management, live log streaming,
and persistent job status across Streamlit sessions.
"""

import os
import sys
import json
import time
import psutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any

# Root workspace directory
WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
LOGS_DIR = os.path.join(WORKSPACE_DIR, "logs")
REGISTRY_PATH = os.path.join(LOGS_DIR, "job_registry.json")
PYTHON_EXE = sys.executable

# Ensure logs directory exists
os.makedirs(LOGS_DIR, exist_ok=True)

# In-memory dictionary of active Popen objects (for current session)
_ACTIVE_POPN: Dict[str, subprocess.Popen] = {}


def _load_registry() -> Dict[str, Dict[str, Any]]:
    """Loads job registry from disk."""
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_registry(registry: Dict[str, Dict[str, Any]]) -> None:
    """Saves job registry to disk atomically."""
    temp_path = REGISTRY_PATH + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, REGISTRY_PATH)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def update_job_status(job: Dict[str, Any]) -> Dict[str, Any]:
    """Updates status of a single job using psutil and log inspection."""
    status = job.get("status", "UNKNOWN")
    pid = job.get("pid")
    job_id = job.get("job_id")

    if status == "RUNNING" and pid:
        # Check process existence
        try:
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    # Still alive
                    return job
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # If we reached here, process is no longer active
        exit_code = None
        if job_id in _ACTIVE_POPN:
            exit_code = _ACTIVE_POPN[job_id].poll()

        if exit_code is None:
            # Check log file for completion indicators
            log_path = job.get("log_path", "")
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    if "Traceback (most recent call last)" in content or "Error:" in content:
                        exit_code = 1
                    elif "All figures" in content or "STAGE 12" in content or "Saved flowcharts" in content or "done" in content:
                        exit_code = 0
                except Exception:
                    pass

        if exit_code == 0:
            job["status"] = "COMPLETED"
            job["exit_code"] = 0
        elif exit_code is not None:
            job["status"] = "FAILED"
            job["exit_code"] = exit_code
        else:
            job["status"] = "FINISHED"
            job["exit_code"] = 0

        job["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return job


def get_all_jobs() -> List[Dict[str, Any]]:
    """Returns all registered jobs with updated statuses, newest first."""
    registry = _load_registry()
    updated = False
    jobs_list = []

    for job_id, job in registry.items():
        prev_status = job.get("status")
        updated_job = update_job_status(job)
        if updated_job.get("status") != prev_status:
            registry[job_id] = updated_job
            updated = True
        jobs_list.append(updated_job)

    if updated:
        _save_registry(registry)

    # Sort descending by start_time
    jobs_list.sort(key=lambda j: j.get("start_time", ""), reverse=True)
    return jobs_list


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Returns a specific job dict with refreshed status."""
    registry = _load_registry()
    if job_id not in registry:
        return None
    job = registry[job_id]
    prev_status = job.get("status")
    updated_job = update_job_status(job)
    if updated_job.get("status") != prev_status:
        registry[job_id] = updated_job
        _save_registry(registry)
    return updated_job


def launch_job(name: str, cmd_args: List[str], description: str = "") -> Dict[str, Any]:
    """
    Launches a command in the background with unbuffered output directed to a log file.
    cmd_args: List of command arguments, e.g. [PYTHON_EXE, "-u", "smmt_figures.py", "fig_montecarlo"]
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    job_id = f"{timestamp}_{clean_name}"
    log_filename = f"{job_id}.log"
    log_path = os.path.join(LOGS_DIR, log_filename)

    # Open log file for writing output directly
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    # On Windows, create a new process group to manage lifecycle cleanly
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        proc = subprocess.Popen(
            cmd_args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=WORKSPACE_DIR,
            creationflags=creation_flags,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
    except Exception as e:
        log_file.write(f"Failed to launch process: {str(e)}\n")
        log_file.close()
        raise e

    _ACTIVE_POPN[job_id] = proc

    job_record = {
        "job_id": job_id,
        "name": name,
        "description": description,
        "command": " ".join(cmd_args),
        "pid": proc.pid,
        "status": "RUNNING",
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None,
        "exit_code": None,
        "log_path": log_path,
        "log_filename": log_filename
    }

    registry = _load_registry()
    registry[job_id] = job_record
    _save_registry(registry)

    return job_record


def kill_job(job_id: str) -> bool:
    """Terminates a running job process and its child processes recursively."""
    registry = _load_registry()
    if job_id not in registry:
        return False

    job = registry[job_id]
    pid = job.get("pid")
    terminated = False

    if pid:
        try:
            if psutil.pid_exists(pid):
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                parent.kill()
                terminated = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            terminated = True

    # Also terminate Popen handle if present
    if job_id in _ACTIVE_POPN:
        try:
            _ACTIVE_POPN[job_id].kill()
        except Exception:
            pass

    job["status"] = "KILLED"
    job["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job["exit_code"] = -9
    registry[job_id] = job
    _save_registry(registry)

    # Append notification to log file
    log_path = job.get("log_path", "")
    if os.path.exists(log_path):
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[DASHBOARD] Process {pid} was manually stopped by user at {job['end_time']}.\n")
        except Exception:
            pass

    return terminated


def read_log_tail(log_path: str, max_lines: int = 400) -> str:
    """Reads the last N lines of a log file safely."""
    if not os.path.exists(log_path):
        return "Log file not created yet."

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) <= max_lines:
            return "".join(lines)
        return "".join(lines[-max_lines:])
    except Exception as e:
        return f"Error reading log: {str(e)}"


def clear_completed_jobs() -> int:
    """Cleans up completed/failed jobs from registry (preserves logs)."""
    registry = _load_registry()
    to_remove = []
    for jid, job in registry.items():
        if job.get("status") in ["COMPLETED", "FAILED", "KILLED", "FINISHED"]:
            to_remove.append(jid)
    for jid in to_remove:
        del registry[jid]
    _save_registry(registry)
    return len(to_remove)
