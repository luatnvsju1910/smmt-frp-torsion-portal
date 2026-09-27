"""
=============================================================================
 SMMT-FRP RC Beam Torsion: Online Engineering Design & Surrogate Tool
=============================================================================
 Academic Publication-Grade Web Interface (Clean Engineering Light Theme)
 Features:
   1. Interactive Beam Cross-Section & Reinforcement Designer
   2. Instant ML Surrogate Inference (< 10 ms) with Conformal Uncertainty Bounds
   3. Capacity-to-Demand Verification (Design Safety Factor & FRP Gain)
   4. Real-Time Explainable AI (Local SHAP Waterfall & What-If Sensitivity)
   5. Batch CSV Ingestion & Vectorized Prediction Engine
   6. Publication Visualizer: Strictly Sequenced 26 Manuscript Figures (No Duplicates)
   7. Model Zoo & Validation Telemetry
=============================================================================
"""

import os
import sys
import glob
import json
import time
import io
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import psutil
import joblib
import platform

# Optional ML libraries
import xgboost as xgb
from catboost import CatBoostRegressor
import shap

# Import background job manager
import job_manager

# Workspace paths
WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
FIGURES_DIR = os.path.join(WORKSPACE_DIR, "figures")
ML_OUT_DIR = os.path.join(WORKSPACE_DIR, "ml_out")
MANUSCRIPT_DIR = os.path.join(WORKSPACE_DIR, "manuscript_figs")
LOGS_DIR = os.path.join(WORKSPACE_DIR, "logs")
META_PATH = os.path.join(ML_OUT_DIR, "inference_meta.json")
PYTHON_EXE = sys.executable

# Page configuration
st.set_page_config(
    page_title="SMMT-FRP Beam Torsion Design Portal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Clean Academic Engineering Light Theme (Replaces all dark/black elements)
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Base */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1e293b;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Academic Header Banner */
    .academic-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #1e3a8a;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .academic-header h1 {
        color: #0f2942;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.01em;
    }
    
    .academic-header p {
        color: #475569;
        font-size: 0.95rem;
        margin: 6px 0 0 0;
    }
    
    /* Clean Academic Card Surfaces */
    .paper-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    
    /* Primary Result Highlight Card */
    .result-hero-card {
        background: #f0f7ff;
        border: 1.5px solid #2563eb;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(37,99,235,0.08);
    }
    
    /* Status Badges */
    .status-pill-pass {
        display: inline-block;
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        border-radius: 9999px;
        padding: 5px 14px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    
    .status-pill-fail {
        display: inline-block;
        background-color: #fff1f2;
        color: #9f1239;
        border: 1px solid #fecdd3;
        border-radius: 9999px;
        padding: 5px 14px;
        font-size: 0.88rem;
        font-weight: 600;
    }
    
    .status-pill-info {
        display: inline-block;
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    
    .status-pill-warn {
        display: inline-block;
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    
    /* Light Engineering Console */
    .light-console {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #0f172a;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.03);
    }
    
    /* Figure Visualizer Card */
    .fig-item-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: border-color 0.15s ease;
    }
    
    .fig-item-card:hover {
        border-color: #2563eb;
    }
    
    .fig-index-tag {
        display: inline-block;
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 7px;
        margin-bottom: 6px;
    }
    
    /* Streamlit UI Tweaks */
    button[kind="primary"] {
        background-color: #1e3a8a !important;
        border-color: #1e3a8a !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1.5px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: #475569;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        color: #1e3a8a !important;
        border-bottom-color: #1e3a8a !important;
        font-weight: 600;
    }
    
    /* Academic Metric Instrument Display */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 8px 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.80rem !important;
        font-weight: 500 !important;
        color: #475569 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: #0f2942 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Model & Metadata Cache Loader
# =============================================================================

@st.cache_resource
def load_inference_artifacts():
    meta = None
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            meta = json.load(f)
            
    models = {}
    
    # Stacking
    stack_path = os.path.join(ML_OUT_DIR, "stack_Tu.joblib")
    if os.path.exists(stack_path):
        try:
            models["Stacking Ensemble"] = joblib.load(stack_path)
        except Exception: pass
        
    # CatBoost
    cb_path = os.path.join(ML_OUT_DIR, "catboost_Tu.cbm")
    if os.path.exists(cb_path):
        try:
            cb = CatBoostRegressor()
            cb.load_model(cb_path)
            models["CatBoost"] = cb
        except Exception: pass
        
    # XGBoost
    xb_path = os.path.join(ML_OUT_DIR, "xgb_Tu.json")
    if os.path.exists(xb_path):
        try:
            xb = xgb.XGBRegressor()
            xb.load_model(xb_path)
            models["XGBoost"] = xb
        except Exception: pass
        
    # LightGBM
    lgb_path = os.path.join(ML_OUT_DIR, "lgb_Tu.joblib")
    if os.path.exists(lgb_path):
        try:
            models["LightGBM"] = joblib.load(lgb_path)
        except Exception: pass

    return models, meta


MODELS, META = load_inference_artifacts()
FEATS = META["features"] if META else ['fc', 'b', 'h', 'fyl', 'fyt', 'rho_l', 'rho_t', 'rho_f', 'Efeq', 'wf_sf']


# =============================================================================
# Canonical Manuscript Figures Sequence (Strict Article Order, No Duplicates)
# =============================================================================

CANONICAL_MANUSCRIPT_FIGURES = [
    {
        "id": 1,
        "section": "1. Database Statistical Distributions & Correlation",
        "title": "Fig. 1: Parametric Database Feature & Target Distributions",
        "subtitle": "LHS distributions for geometry (b, h), concrete (fc), reinforcement (ρl, ρt, ρf), and torque (Tu)",
        "rel_path": "figures/fig01_distributions.png"
    },
    {
        "id": 2,
        "section": "1. Database Statistical Distributions & Correlation",
        "title": "Fig. 2: Feature Correlation Matrix",
        "subtitle": "Heatmap of pairwise Pearson linear and Spearman rank correlation coefficients",
        "rel_path": "figures/fig02_correlation.png"
    },
    {
        "id": 3,
        "section": "2. Candidate Model Performance & Residual Diagnostics",
        "title": "Fig. 3: Multi-Metric Benchmark Across 10 ML Models",
        "subtitle": "Comparison of R², RMSE, MAE, MAPE, and a20-index on holdout testing samples (N = 1,000)",
        "rel_path": "figures/fig03_model_comparison.png"
    },
    {
        "id": 4,
        "section": "2. Candidate Model Performance & Residual Diagnostics",
        "title": "Fig. 4: Parity Plots Across All 10 Candidate Models",
        "subtitle": "Predicted ultimate torque vs MSMMT ground truth on holdout test set with ±10% and ±20% bounds",
        "rel_path": "figures/fig04_01_parity_all_models.png"
    },
    {
        "id": 5,
        "section": "2. Candidate Model Performance & Residual Diagnostics",
        "title": "Fig. 5: Residual Error Distributions Across All Models",
        "subtitle": "Residual distribution histograms (y_true - y_pred) showing zero-mean centering and dispersion",
        "rel_path": "figures/fig04_02_residual_all_models.png"
    },
    {
        "id": 6,
        "section": "2. Candidate Model Performance & Residual Diagnostics",
        "title": "Fig. 6: Quantile-Quantile (Q-Q) Normality Plots",
        "subtitle": "Theoretical Gaussian quantiles vs sample standardized residuals for residual normality testing",
        "rel_path": "figures/fig04_03_qq_all_models.png"
    },
    {
        "id": 7,
        "section": "3. Explainable AI: Global & Local SHAP Interpretability",
        "title": "Fig. 7: SHAP Global Feature Importance Ranking",
        "subtitle": "Mean absolute SHAP value (|SHAP|) indicating primary drivers of torsional capacity",
        "rel_path": "figures/fig05_02_shap_mean_importance.png"
    },
    {
        "id": 8,
        "section": "3. Explainable AI: Global & Local SHAP Interpretability",
        "title": "Fig. 8: SHAP Summary Beeswarm Distribution",
        "subtitle": "Impact of low-to-high feature values on model output with dense sample-level point clouds",
        "rel_path": "figures/fig05_01_shap_beeswarm.png"
    },
    {
        "id": 9,
        "section": "3. Explainable AI: Global & Local SHAP Interpretability",
        "title": "Fig. 9: SHAP Feature Interaction Matrix Heatmap",
        "subtitle": "Second-order Shapley interaction values demonstrating coupled concrete-FRP confinement",
        "rel_path": "figures/fig05_03_shap_heatmap.png"
    },
    {
        "id": 10,
        "section": "3. Explainable AI: Global & Local SHAP Interpretability",
        "title": "Fig. 10: Local SHAP Waterfall Attribution (Specimen B-C1B1-100)",
        "subtitle": "Step-by-step contribution of each design parameter from population mean base to final predicted Tu",
        "rel_path": "figures/fig05_04_shap_waterfall_BC1_100.png"
    },
    {
        "id": 11,
        "section": "3. Explainable AI: Global & Local SHAP Interpretability",
        "title": "Fig. 11: SHAP Partial Dependence Curves Grid",
        "subtitle": "Non-linear individual feature responses colored by primary interacting variables",
        "rel_path": "figures/fig05_05_shap_dependence_grid.png"
    },
    {
        "id": 12,
        "section": "4. Probabilistic Reliability & Monte Carlo Uncertainty",
        "title": "Fig. 12: Monte Carlo PDF & Empirical Capacity Histogram",
        "subtitle": "Statistical distribution of ultimate torque with fitted Lognormal & Weibull PDF (N = 100,000 runs)",
        "rel_path": "figures/fig07_montecarlo_individual/fig07_01_mc_pdf_histogram.png"
    },
    {
        "id": 13,
        "section": "4. Probabilistic Reliability & Monte Carlo Uncertainty",
        "title": "Fig. 13: Monte Carlo CDF & Reliability Exceedance Curve",
        "subtitle": "Cumulative Distribution Function and characteristic fractiles (P05 = 39.46 kN·m, P95 = 54.34 kN·m)",
        "rel_path": "figures/fig07_montecarlo_individual/fig07_02_mc_cdf_reliability.png"
    },
    {
        "id": 14,
        "section": "4. Probabilistic Reliability & Monte Carlo Uncertainty",
        "title": "Fig. 14: Monte Carlo Statistical Convergence Trace",
        "subtitle": "Stabilization of mean and standard deviation of Tu across 10^1 to 10^5 simulation iterations",
        "rel_path": "figures/fig07_montecarlo_individual/fig07_03_mc_convergence.png"
    },
    {
        "id": 15,
        "section": "4. Probabilistic Reliability & Monte Carlo Uncertainty",
        "title": "Fig. 15: Tornado Sensitivity Diagram (OAT Contributions)",
        "subtitle": "Variance sensitivity contributions under material and geometric manufacturing uncertainties",
        "rel_path": "figures/fig07_montecarlo_individual/fig07_04_mc_tornado_sensitivity.png"
    },
    {
        "id": 16,
        "section": "4. Probabilistic Reliability & Monte Carlo Uncertainty",
        "title": "Fig. 16: Probabilistic Input Random Variable Distributions",
        "subtitle": "Normal, Lognormal, and Weibull distributions assigned to fc, fyl, fyt, b, h, and FRP thickness",
        "rel_path": "figures/fig07_montecarlo_individual/fig07_05_mc_input_distributions.png"
    },
    {
        "id": 17,
        "section": "5. Extrapolation, Cross-Validation & HPO Convergence",
        "title": "Fig. 17: Extrapolation Robustness & Domain of Validity",
        "subtitle": "Error penalty comparison between interior domain interpolation vs exterior extrapolation",
        "rel_path": "figures/fig08_extrapolation.png"
    },
    {
        "id": 18,
        "section": "5. Extrapolation, Cross-Validation & HPO Convergence",
        "title": "Fig. 18: Multidimensional Performance Radar Benchmark",
        "subtitle": "Radar chart mapping normalized accuracy, residual compactness, and generalizability",
        "rel_path": "figures/fig09_radar.png"
    },
    {
        "id": 19,
        "section": "5. Extrapolation, Cross-Validation & HPO Convergence",
        "title": "Fig. 19: 10-Fold CV Error Distributions (RMSE, MAE, MAPE)",
        "subtitle": "Fold-wise boxplots showing consistent cross-validation error stability without fold bias",
        "rel_path": "figures/fig10_01_cv_error_metrics.png"
    },
    {
        "id": 20,
        "section": "5. Extrapolation, Cross-Validation & HPO Convergence",
        "title": "Fig. 20: 10-Fold CV Accuracy Distributions (R² & a20-index)",
        "subtitle": "Fold-wise distribution of coefficient of determination and a20-index (99.7% compliance)",
        "rel_path": "figures/fig10_02_cv_accuracy_metrics.png"
    },
    {
        "id": 21,
        "section": "5. Extrapolation, Cross-Validation & HPO Convergence",
        "title": "Fig. 21: Residual Error Distribution & Kernel Density Estimation",
        "subtitle": "Error distribution with fitted Gaussian kernel density showing unbiased estimation",
        "rel_path": "figures/fig11_error_distribution.png"
    },
    {
        "id": 22,
        "section": "5. Extrapolation, Cross-Validation & HPO Convergence",
        "title": "Fig. 22: Individual Conditional Expectation (ICE) 6-Panel Grid",
        "subtitle": "ICE instance curves and Partial Dependence Plots across primary structural features",
        "rel_path": "figures/fig12_ice_pdp_6panel.png"
    },
    {
        "id": 23,
        "section": "5. Extrapolation, Cross-Validation & HPO Convergence",
        "title": "Fig. 23: Optuna Hyperparameter Optimization Convergence",
        "subtitle": "Objective validation loss vs trial index showing rapid Bayesian optimization convergence",
        "rel_path": "figures/fig13b_hpo_convergence_comparison.png"
    },
    {
        "id": 24,
        "section": "5. Extrapolation, Cross-Validation & HPO Convergence",
        "title": "Fig. 24: Iterative Boosting & Epoch Training History",
        "subtitle": "Train vs validation loss progression across boosting iterations for CatBoost, XGBoost, LightGBM, and MLP",
        "rel_path": "figures/fig13c_iterative_training_history.png"
    }
]


# =============================================================================
# Helper Utilities
# =============================================================================

def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def get_image_info(path: str) -> dict:
    info = {"size_str": "0 B", "dims": "N/A", "mtime": "N/A"}
    try:
        stat = os.stat(path)
        info["size_str"] = format_file_size(stat.st_size)
        info["mtime"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        with Image.open(path) as img:
            info["dims"] = f"{img.width} × {img.height} px"
    except Exception:
        pass
    return info

def predict_beam_capacity(feature_vector: dict, model_name: str = "Stacking Ensemble", conf_level: str = "95"):
    model = MODELS.get(model_name) or MODELS.get("Stacking Ensemble") or MODELS.get("CatBoost") or MODELS.get("XGBoost")
    if model is None:
        raise ValueError("Model artifact not found.")

    x_in = np.array([[feature_vector[f] for f in FEATS]])
    pred_log = float(model.predict(x_in)[0])
    tu_pred = float(np.exp(pred_log))
    
    # Conformal Prediction Interval
    q_val = 0.06
    if META and "conformal_q" in META:
        mod_key = "Stacking" if "Stacking" in model_name else ("CatBoost" if "CatBoost" in model_name else "CatBoost")
        model_q = META["conformal_q"].get(mod_key, {})
        q_val = model_q.get(conf_level, 0.06)
        
    tu_min = float(np.exp(pred_log - q_val))
    tu_max = float(np.exp(pred_log + q_val))
    
    # Unstrengthened RC Baseline (rho_f = 0, wf_sf = 0)
    x_unstr = x_in.copy()
    x_unstr[0, FEATS.index("rho_f")] = 0.0
    x_unstr[0, FEATS.index("wf_sf")] = 0.0
    tu_unstr = float(np.exp(model.predict(x_unstr)[0]))
    
    frp_gain = max(0.0, tu_pred - tu_unstr)
    frp_gain_pct = (frp_gain / tu_unstr * 100.0) if tu_unstr > 0 else 0.0
    
    # Extrapolation Check
    out_of_bounds = []
    if META and "bounds" in META:
        for f in FEATS:
            b_info = META["bounds"].get(f)
            if b_info:
                val = feature_vector[f]
                if val < b_info["min"] or val > b_info["max"]:
                    out_of_bounds.append((f, val, b_info["min"], b_info["max"]))
                    
    return {
        "Tu": tu_pred,
        "Tu_min": tu_min,
        "Tu_max": tu_max,
        "Tu_unstr": tu_unstr,
        "frp_gain": frp_gain,
        "frp_gain_pct": frp_gain_pct,
        "conf_level": conf_level,
        "out_of_bounds": out_of_bounds,
        "is_in_domain": len(out_of_bounds) == 0,
        "x_vector": x_in
    }


# =============================================================================
# Clean Academic Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### 🏛️ SMMT-FRP Portal")
    st.caption("Structural Engineering AI Surrogate")
    
    st.markdown('<div class="paper-card" style="padding:12px; margin-bottom:12px;">', unsafe_allow_html=True)
    st.markdown("**Core Models Status:**")
    if MODELS:
        st.markdown(f"<span class='status-pill-pass'>Ready: {len(MODELS)} Models</span>", unsafe_allow_html=True)
        st.caption(f"Active: {', '.join(MODELS.keys())}")
    else:
        st.markdown("<span class='status-pill-warn'>No Models Loaded</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # System Info
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        cpu_str = f"{cpu_percent}%"
        ram_str = f"{ram.percent}%"
    except Exception:
        cpu_str, ram_str = "N/A", "N/A"

    st.markdown('<div class="paper-card" style="padding:12px; margin-bottom:12px;">', unsafe_allow_html=True)
    st.caption(f"**Host Environment:** {platform.system()} {platform.machine()}")
    st.caption(f"**Compute Load:** CPU {cpu_str} | RAM {ram_str}")
    st.caption(f"**Runtime:** Python {sys.version.split()[0]}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Background Pipelines
    jobs = job_manager.get_all_jobs()
    running_jobs = [j for j in jobs if j.get("status") == "RUNNING"]
    if running_jobs:
        st.markdown('<div class="paper-card" style="padding:12px; margin-bottom:12px; border-left: 3px solid #2563eb;">', unsafe_allow_html=True)
        st.markdown(f"**Background Execution ({len(running_jobs)}):**")
        for rj in running_jobs:
            st.caption(f"• {rj['name']} (PID {rj['pid']})")
        if st.button("🛑 Terminate Background Job", use_container_width=True):
            for rj in running_jobs:
                job_manager.kill_job(rj["job_id"])
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.divider()
    if st.button("🔄 Refresh Data & View", use_container_width=True):
        st.rerun()


# =============================================================================
# Academic Header Banner
# =============================================================================

st.markdown("""
<div class="academic-header">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div>
            <h1>Torsional Capacity Design Tool for FRP-Strengthened RC Beams</h1>
            <p>
                Physics-Informed Machine Learning Surrogate (MSMMT-FRP) • Split-Conformal Uncertainty Quantification • Validated against 5,000 SMMT Solutions
            </p>
        </div>
        <div style="text-align:right; display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end;">
            <span class="status-pill-pass">Test R² = 0.9965</span>
            <span class="status-pill-info">RMSE = 2.16 kN·m</span>
            <span class="status-pill-info">Split-Conformal 95%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# Navigation Tabs
# =============================================================================

tab_design, tab_batch, tab_plots, tab_models, tab_console = st.tabs([
    "Beam Designer & Analysis",
    "Batch Evaluation Engine",
    "Manuscript Figure Gallery",
    "Model Zoo & Validation",
    "Execution Console & Diagnostics"
])


# =============================================================================
# TAB 1: INTERACTIVE BEAM DESIGNER & PREDICTION WORKSPACE
# =============================================================================

with tab_design:
    # Preset Selector Row
    col_p_title, col_p_drop = st.columns([1.2, 3])
    with col_p_title:
        st.markdown("**Load Reference Benchmark Specimen:**")
    with col_p_drop:
        preset_choice = st.selectbox(
            "Select Reference Specimen:",
            [
                "Custom User Design",
                "Benchmark 1: Standard Medium Beam + Continuous CFRP Jacket (b=250, h=400 mm, f'c=35 MPa)",
                "Benchmark 2: High-Strength Deep Beam + 2-Ply Discrete CFRP Strips (b=300, h=550 mm, f'c=50 MPa)",
                "Benchmark 3: Slender Beam + High-Modulus CFRP (b=200, h=450 mm, f'c=40 MPa, Ef=390 GPa)",
                "Benchmark 4: Experimental Specimen B-C1B1-100 (b=250, h=300 mm, f'c=32.34 MPa, Exp. Tu=53.7 kN·m)"
            ],
            index=1,
            label_visibility="collapsed"
        )
        
    if "Benchmark 1" in preset_choice:
        def_b, def_h, def_fc = 250, 400, 35.0
        def_n, def_Dl, def_fyl = 4, 16, 400.0
        def_Dt, def_s, def_c, def_fyt = 8, 150, 25, 400.0
        def_wrap, def_nf, def_tf, def_Ef, def_wf, def_sf = "Continuous Wrap", 1, 0.167, 230.0, 100.0, 100.0
        def_Td = 40.0
    elif "Benchmark 2" in preset_choice:
        def_b, def_h, def_fc = 300, 550, 50.0
        def_n, def_Dl, def_fyl = 6, 20, 400.0
        def_Dt, def_s, def_c, def_fyt = 10, 100, 25, 400.0
        def_wrap, def_nf, def_tf, def_Ef, def_wf, def_sf = "Discrete Strips", 2, 0.167, 250.0, 100.0, 150.0
        def_Td = 85.0
    elif "Benchmark 3" in preset_choice:
        def_b, def_h, def_fc = 200, 450, 40.0
        def_n, def_Dl, def_fyl = 4, 14, 400.0
        def_Dt, def_s, def_c, def_fyt = 6, 120, 25, 400.0
        def_wrap, def_nf, def_tf, def_Ef, def_wf, def_sf = "Continuous Wrap", 1, 0.167, 390.0, 100.0, 100.0
        def_Td = 35.0
    elif "Benchmark 4" in preset_choice:
        def_b, def_h, def_fc = 250, 300, 32.34
        def_n, def_Dl, def_fyl = 4, 18, 364.3
        def_Dt, def_s, def_c, def_fyt = 6, 150, 20, 378.6
        def_wrap, def_nf, def_tf, def_Ef, def_wf, def_sf = "Discrete Strips", 1, 0.238, 285.0, 100.0, 200.0
        def_Td = 45.0
    else:
        def_b, def_h, def_fc = 250, 400, 35.0
        def_n, def_Dl, def_fyl = 4, 16, 400.0
        def_Dt, def_s, def_c, def_fyt = 8, 150, 25, 400.0
        def_wrap, def_nf, def_tf, def_Ef, def_wf, def_sf = "Continuous Wrap", 1, 0.167, 230.0, 100.0, 100.0
        def_Td = 40.0

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Two-Column Layout (Inputs vs Results)
    col_input, col_result = st.columns([1.1, 1.3], gap="large")

    with col_input:
        st.markdown("#### 1. Structural & Material Input Console")
        st.caption("Standard physical units: Dimension (mm), Concrete/Steel strength (MPa), FRP Modulus (GPa).")
        
        # Section A: Geometry & Concrete
        with st.expander("A. Section Geometry & Concrete Compressive Strength", expanded=True):
            col_b, col_h = st.columns(2)
            with col_b:
                inp_b = st.slider("Section Width b (mm):", min_value=150, max_value=350, value=int(def_b), step=5)
            with col_h:
                inp_h = st.slider("Section Height h (mm):", min_value=150, max_value=700, value=int(def_h), step=10)
            inp_fc = st.slider("Concrete Cylinder Strength f'c (MPa):", min_value=20.0, max_value=60.0, value=float(def_fc), step=0.5)

        # Section B: Steel Reinforcement
        with st.expander("B. Internal Steel Reinforcement System", expanded=True):
            rebar_mode = st.radio("Reinforcement Detailing Mode:", ["Physical Rebar Specs (n, Dl, s, Dt)", "Direct Ratios (ρl, ρt)"], horizontal=True)
            
            if rebar_mode == "Physical Rebar Specs (n, Dl, s, Dt)":
                st.markdown("**Longitudinal Bars:**")
                col_n, col_dl, col_fyl = st.columns(3)
                with col_n:
                    inp_n = st.selectbox("Bar Count n:", [4, 6, 8, 10, 12], index=[4, 6, 8, 10, 12].index(def_n) if def_n in [4, 6, 8, 10, 12] else 0)
                with col_dl:
                    inp_Dl = st.selectbox("Bar Dia. Dl (mm):", [10, 12, 14, 16, 18, 20, 22, 25], index=[10, 12, 14, 16, 18, 20, 22, 25].index(int(def_Dl)) if int(def_Dl) in [10, 12, 14, 16, 18, 20, 22, 25] else 3)
                with col_fyl:
                    inp_fyl = st.number_input("Yield fyl (MPa):", min_value=300.0, max_value=500.0, value=float(def_fyl), step=10.0)
                    
                Ac = inp_b * inp_h
                Asl = inp_n * np.pi * (inp_Dl**2) / 4.0
                calc_rho_l = Asl / Ac
                st.caption(f"↳ Asl = `{Asl:.1f} mm²` | **ρl = {calc_rho_l*100:.2f}%**")
                
                st.markdown("**Transverse Stirrups:**")
                col_dt, col_s, col_fyt = st.columns(3)
                with col_dt:
                    inp_Dt = st.selectbox("Stirrup Dt (mm):", [6, 8, 10, 12], index=[6, 8, 10, 12].index(int(def_Dt)) if int(def_Dt) in [6, 8, 10, 12] else 1)
                with col_s:
                    inp_s = st.slider("Spacing s (mm):", min_value=50, max_value=250, value=int(def_s), step=10)
                with col_fyt:
                    inp_fyt = st.number_input("Yield fyt (MPa):", min_value=300.0, max_value=500.0, value=float(def_fyt), step=10.0)
                    
                inp_c = st.number_input("Clear Cover c (mm):", min_value=15, max_value=40, value=int(def_c), step=5)
                Ast = np.pi * (inp_Dt**2) / 4.0
                pst = 2.0 * (inp_b + inp_h) - 8.0 * (inp_c - inp_Dt / 2.0)
                calc_rho_t = (Ast * pst) / (Ac * inp_s)
                st.caption(f"↳ Stirrup Perimeter pst = `{pst:.1f} mm` | **ρt = {calc_rho_t*100:.3f}%**")
            else:
                col_rl, col_rt = st.columns(2)
                with col_rl:
                    calc_rho_l = st.slider("Longitudinal Ratio ρl (%):", min_value=0.5, max_value=3.5, value=1.5, step=0.1) / 100.0
                    inp_fyl = st.number_input("Yield fyl (MPa):", min_value=300.0, max_value=500.0, value=400.0, step=10.0)
                with col_rt:
                    calc_rho_t = st.slider("Transverse Stirrup Ratio ρt (%):", min_value=0.1, max_value=0.8, value=0.4, step=0.05) / 100.0
                    inp_fyt = st.number_input("Yield fyt (MPa):", min_value=300.0, max_value=500.0, value=400.0, step=10.0)
                inp_c = 25

        # Section C: FRP Strengthening
        with st.expander("C. External FRP Composite Strengthening", expanded=True):
            inp_wrap = st.radio("FRP Configuration:", ["Continuous Wrap (wf = sf)", "Discrete Strips (wf, sf)"], index=0 if def_wrap == "Continuous Wrap" else 1, horizontal=True)
            col_nf, col_tf, col_ef = st.columns(3)
            with col_nf:
                inp_nf = st.selectbox("Plies nf:", [1, 2, 3, 4, 5], index=min(4, int(def_nf)-1))
            with col_tf:
                inp_tf = st.number_input("Thickness tf (mm):", min_value=0.05, max_value=0.50, value=float(def_tf), step=0.02)
            with col_ef:
                inp_Ef = st.number_input("Modulus Ef (GPa):", min_value=120.0, max_value=420.0, value=float(def_Ef), step=10.0)
                
            if inp_wrap == "Continuous Wrap (wf = sf)":
                calc_wf_sf = 1.0
                calc_wf = 100.0
                calc_sf = 100.0
                calc_rho_f = 2.0 * inp_nf * inp_tf * 1.0 * (inp_b + inp_h) / (inp_b * inp_h)
            else:
                col_wf, col_sf = st.columns(2)
                with col_wf:
                    calc_wf = st.slider("Strip Width wf (mm):", min_value=30.0, max_value=200.0, value=float(def_wf), step=10.0)
                with col_sf:
                    calc_sf = st.slider("Strip Spacing sf (mm):", min_value=50.0, max_value=300.0, value=float(def_sf), step=10.0)
                calc_wf_sf = min(1.0, calc_wf / calc_sf)
                calc_rho_f = (2.0 * inp_nf * inp_tf * calc_wf * (inp_b + inp_h)) / (inp_b * inp_h * calc_sf)
                
            calc_Efeq = inp_Ef
            st.caption(f"↳ **ρf = {calc_rho_f*100:.3f}%** | Efeq = `{calc_Efeq:.1f} GPa` | wf/sf = `{calc_wf_sf:.2f}`")

        # Section D: Factored Demand & Model Controls
        with st.expander("D. Applied Torsional Demand & Surrogate Controls", expanded=True):
            col_td, col_model, col_uq = st.columns(3)
            with col_td:
                inp_Td = st.number_input("Design Torque Td (kN·m):", min_value=1.0, max_value=300.0, value=float(def_Td), step=5.0)
            with col_model:
                avail_models = list(MODELS.keys()) if MODELS else ["Stacking Ensemble"]
                chosen_model = st.selectbox("Surrogate Model:", avail_models, index=0)
            with col_uq:
                chosen_uq = st.selectbox("Confidence Level:", ["95%", "90%"], index=0)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: PREDICTION & EVALUATION RESULTS
    # -------------------------------------------------------------------------
    with col_result:
        st.markdown("#### 2. Mechanical Evaluation & Verification")
        
        feat_vector = {
            "fc": float(inp_fc), "b": float(inp_b), "h": float(inp_h),
            "fyl": float(inp_fyl), "fyt": float(inp_fyt),
            "rho_l": float(calc_rho_l), "rho_t": float(calc_rho_t), "rho_f": float(calc_rho_f),
            "Efeq": float(calc_Efeq), "wf_sf": float(calc_wf_sf)
        }
        
        aspect_ratio = inp_h / inp_b
        b_prime = inp_b - 2.0 * inp_c
        h_prime = inp_h - 2.0 * inp_c
        A0 = 0.85 * b_prime * h_prime
        ph = 2.0 * (b_prime + h_prime)
        sqrt_fc = np.sqrt(inp_fc)
        frp_stiffness = calc_Efeq * calc_rho_f
        
        # Physics-Derived Derived Features Card
        st.markdown('<div class="paper-card" style="padding:14px 18px;">', unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#475569; letter-spacing:0.04em; text-transform:uppercase; margin-bottom:8px;'>Physics-Derived Geometric & Mechanical Parameters</div>", unsafe_allow_html=True)
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("Aspect Ratio h/b", f"{aspect_ratio:.2f}")
            st.metric("Long. Steel ρl", f"{calc_rho_l*100:.2f}%")
        with col_d2:
            st.metric("Core Area A0", f"{A0:,.0f} mm²")
            st.metric("Stirrup Ratio ρt", f"{calc_rho_t*100:.3f}%")
        with col_d3:
            st.metric("Core Perimeter ph", f"{ph:,.0f} mm")
            st.metric("FRP Ratio ρf", f"{calc_rho_f*100:.3f}%")
        with col_d4:
            st.metric("Concrete Metric √f'c", f"{sqrt_fc:.2f} MPa")
            st.metric("FRP Rigidity Ef·ρf", f"{frp_stiffness:.3f} GPa")
        st.markdown('</div>', unsafe_allow_html=True)

        try:
            pred_res = predict_beam_capacity(feat_vector, model_name=chosen_model, conf_level=chosen_uq.replace("%", ""))
            
            # Domain of validity pill
            if pred_res["is_in_domain"]:
                st.markdown("<span class='status-pill-pass'>Applicability Domain: Within 5,000 SMMT Training Bounds</span>", unsafe_allow_html=True)
            else:
                warn_msg = ", ".join([f"{item[0]}={item[1]:.2f} (domain [{item[2]:.1f}, {item[3]:.1f}])" for item in pred_res["out_of_bounds"]])
                st.markdown(f"<span class='status-pill-warn'>Extrapolation Notice: {warn_msg}</span>", unsafe_allow_html=True)

            Tu_pred = pred_res["Tu"]
            Tu_min = pred_res["Tu_min"]
            Tu_max = pred_res["Tu_max"]
            Tu_unstr = pred_res["Tu_unstr"]
            frp_gain = pred_res["frp_gain"]
            frp_gain_pct = pred_res["frp_gain_pct"]
            
            # Primary Highlight Result Card
            st.markdown(f"""
            <div class="result-hero-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                    <div>
                        <div style="color:#1e3a8a; font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">
                            Predicted Ultimate Torsional Resistance (Tu)
                        </div>
                        <div style="color:#0f2942; font-size:2.8rem; font-weight:800; line-height:1.1; margin-top:2px;">
                            {Tu_pred:.2f} <span style="font-size:1.3rem; font-weight:500; color:#475569;">kN·m</span>
                        </div>
                        <div style="color:#059669; font-size:0.92rem; font-weight:600; margin-top:6px;">
                            Split-Conformal {chosen_uq} Bound: [{Tu_min:.2f}, {Tu_max:.2f}] kN·m
                            <span style="color:#64748b; font-weight:400; margin-left:4px;">(Margin ±{(Tu_max - Tu_min)/2:.2f} kN·m)</span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="color:#475569; font-size:0.82rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">FRP Strengthening Gain</div>
                        <div style="color:#059669; font-size:1.8rem; font-weight:700; margin-top:2px;">
                            +{frp_gain:.2f} kN·m
                        </div>
                        <div style="color:#2563eb; font-size:0.92rem; font-weight:600;">
                            (+{frp_gain_pct:.1f}% over unstrengthened RC core)
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Capacity-to-Demand Check
            sf = Tu_pred / inp_Td if inp_Td > 0 else 1.0
            margin = Tu_pred - inp_Td
            
            col_v1, col_v2, col_v3 = st.columns([1.5, 1, 1])
            with col_v1:
                if sf >= 1.0:
                    st.markdown("<span class='status-pill-pass'>PASS: Adequate Torsional Capacity (Tu ≥ Td)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='status-pill-fail'>FAIL: Insufficient Torsional Capacity (Tu &lt; Td)</span>", unsafe_allow_html=True)
            with col_v2:
                st.metric("Ratio Tu / Td", f"{sf:.2f}", delta=f"{sf - 1.0:+.2f}")
            with col_v3:
                st.metric("Reserve Margin Tu - Td", f"{margin:+.2f} kN·m")

            # Clean Plotly Visualizations (Academic White Cards)
            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                st.markdown('<div class="paper-card" style="padding:12px 14px 4px 14px; margin-bottom:12px;">', unsafe_allow_html=True)
                max_gauge = max(Tu_pred * 1.35, inp_Td * 1.3, 100.0)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=Tu_pred,
                    delta={
                        'reference': inp_Td,
                        'increasing': {'color': '#059669'},
                        'decreasing': {'color': '#dc2626'},
                        'prefix': "Δ vs Demand: ",
                        'font': {'size': 12, 'family': 'Inter, sans-serif'}
                    },
                    domain={'x': [0.05, 0.95], 'y': [0.08, 0.88]},
                    title={
                        'text': "<b>Torsional Capacity Assessment</b>",
                        'font': {'size': 13.5, 'color': '#0f2942', 'family': 'Inter, sans-serif'}
                    },
                    number={
                        'font': {'color': '#1e3a8a', 'size': 24, 'family': 'Inter, sans-serif'},
                        'suffix': " kN·m",
                        'valueformat': '.1f'
                    },
                    gauge={
                        'axis': {'range': [0, max_gauge], 'tickcolor': '#64748b', 'tickwidth': 1, 'tickfont': {'size': 10}},
                        'bar': {'color': '#2563eb', 'thickness': 0.28},
                        'bgcolor': "#f8fafc",
                        'borderwidth': 1,
                        'bordercolor': "#cbd5e1",
                        'steps': [
                            {'range': [0, inp_Td], 'color': '#fee2e2'},
                            {'range': [inp_Td, max_gauge], 'color': '#ecfdf5'}
                        ],
                        'threshold': {
                            'line': {'color': "#dc2626", 'width': 3},
                            'thickness': 0.8,
                            'value': inp_Td
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=310,
                    margin=dict(l=20, r=20, t=45, b=20),
                    paper_bgcolor="#ffffff",
                    font_color="#0f2942"
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with plot_col2:
                st.markdown('<div class="paper-card" style="padding:12px 14px 4px 14px; margin-bottom:12px;">', unsafe_allow_html=True)
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=['Design Capacity'],
                    y=[Tu_unstr],
                    name='RC Concrete + Steel Core',
                    marker_color='#94a3b8',
                    text=[f"{Tu_unstr:.1f} kN·m"] if Tu_unstr > 12 else [""],
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='#0f172a', size=11, family='Inter, sans-serif')
                ))
                fig_bar.add_trace(go.Bar(
                    x=['Design Capacity'],
                    y=[frp_gain],
                    name='FRP Strengthening Gain',
                    marker_color='#059669',
                    text=[f"+{frp_gain:.1f} kN·m"] if frp_gain > 8 else [""],
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='#ffffff', size=11, family='Inter, sans-serif')
                ))
                # Factored demand horizontal dashed reference line
                fig_bar.add_hline(
                    y=inp_Td,
                    line_dash="dash",
                    line_color="#dc2626",
                    line_width=2,
                    annotation_text=f"Demand Td = {inp_Td:.1f} kN·m",
                    annotation_position="top left",
                    annotation_font=dict(size=11, color="#dc2626", family='Inter, sans-serif')
                )
                # Total capacity annotation at the top of the stacked bar
                fig_bar.add_annotation(
                    x=0,
                    y=Tu_pred,
                    text=f"<b>Tu = {Tu_pred:.1f} kN·m</b>",
                    showarrow=False,
                    yshift=14,
                    font=dict(size=12, color='#1e3a8a', family='Inter, sans-serif')
                )
                max_bar_y = max(Tu_pred, inp_Td) * 1.3
                fig_bar.update_layout(
                    barmode='stack',
                    title=dict(
                        text="<b>Capacity Breakdown vs Factored Demand</b>",
                        font=dict(size=13.5, color='#0f2942', family='Inter, sans-serif'),
                        x=0.02,
                        y=0.96
                    ),
                    height=310,
                    margin=dict(l=45, r=20, t=45, b=65),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font_color="#0f2942",
                    yaxis=dict(title="Torsional Moment (kN·m)", gridcolor="#f1f5f9", range=[0, max_bar_y]),
                    xaxis=dict(tickfont=dict(size=11, color='#0f2942')),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.22,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10.5, color='#334155')
                    )
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Local Feature Attribution & Parametric Sensitivity Tabs
            shap_t, sweep_t = st.tabs(["Local Feature Attribution (SHAP)", "Parametric Sensitivity Analysis"])
            
            with shap_t:
                if "XGBoost" in MODELS:
                    try:
                        explainer = shap.TreeExplainer(MODELS["XGBoost"])
                        shap_vals = explainer.shap_values(pred_res["x_vector"])[0]
                        
                        feat_academic_labels = {
                            "fc": "f'c (Concrete Strength)",
                            "b": "b (Section Width)",
                            "h": "h (Section Depth)",
                            "fyl": "fyl (Long. Steel Yield)",
                            "fyt": "fyt (Stirrup Yield)",
                            "rho_l": "ρl (Long. Reinforcement)",
                            "rho_t": "ρt (Transverse Stirrup)",
                            "rho_f": "ρf (FRP Volumetric Ratio)",
                            "Efeq": "Ef (FRP Modulus)",
                            "wf_sf": "wf/sf (FRP Continuity Ratio)"
                        }
                        
                        df_shap = pd.DataFrame({
                            "Feature": [feat_academic_labels.get(f, f) for f in FEATS],
                            "Value": [feat_vector[f] for f in FEATS],
                            "SHAP_Impact": shap_vals
                        }).sort_values("SHAP_Impact", ascending=True)
                        
                        df_shap["Color"] = df_shap["SHAP_Impact"].apply(lambda v: "#059669" if v >= 0 else "#dc2626")
                        df_shap["Label"] = df_shap.apply(lambda r: f"{r['Feature']} = {r['Value']:.3g}", axis=1)
                        
                        fig_shap = px.bar(
                            df_shap,
                            x="SHAP_Impact",
                            y="Label",
                            orientation="h",
                            title="Local Feature Attribution: Contribution to ln(Tu)",
                            color="Color",
                            color_discrete_map="identity"
                        )
                        fig_shap.update_layout(
                            height=320,
                            margin=dict(l=10, r=20, t=40, b=35),
                            xaxis_title="SHAP Value (Contribution to ln(Tu))",
                            yaxis_title="",
                            paper_bgcolor="#ffffff",
                            plot_bgcolor="#ffffff",
                            font_color="#0f2942",
                            xaxis=dict(gridcolor="#f1f5f9"),
                            title_font=dict(size=13.5, color='#0f2942', family='Inter, sans-serif')
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                    except Exception as e:
                        st.caption(f"SHAP explanation: {e}")
                else:
                    st.caption("SHAP feature attribution active when XGBoost model is loaded.")

            with sweep_t:
                sweep_var = st.selectbox(
                    "Parameter for Sensitivity Sweep:",
                    [
                        "FRP Layer Count (nf)",
                        "Concrete Strength (f'c)",
                        "Section Width (b)",
                        "Section Height (h)",
                        "FRP Elastic Modulus (Ef)"
                    ],
                    index=0
                )
                sweep_x, sweep_y = [], []
                current_val = 0
                model_sw = MODELS.get(chosen_model) or MODELS.get("Stacking Ensemble") or MODELS.get("CatBoost")
                
                if sweep_var == "FRP Layer Count (nf)":
                    sweep_vals = np.array([0, 1, 2, 3, 4, 5])
                    current_val = inp_nf
                    for v in sweep_vals:
                        vec = feat_vector.copy()
                        vec["rho_f"] = 0.0 if v == 0 else (2.0 * v * inp_tf * calc_wf * (inp_b + inp_h)) / (inp_b * inp_h * calc_sf)
                        vec["wf_sf"] = 0.0 if v == 0 else calc_wf_sf
                        sweep_y.append(float(np.exp(model_sw.predict(np.array([[vec[f] for f in FEATS]]))[0])))
                    sweep_x = sweep_vals
                    x_lbl = "Number of FRP Plies nf"
                elif sweep_var == "Concrete Strength (f'c)":
                    sweep_vals = np.linspace(20.0, 60.0, 25)
                    current_val = inp_fc
                    for v in sweep_vals:
                        vec = feat_vector.copy(); vec["fc"] = v
                        sweep_y.append(float(np.exp(model_sw.predict(np.array([[vec[f] for f in FEATS]]))[0])))
                    sweep_x = sweep_vals
                    x_lbl = "Concrete Cylinder Strength f'c (MPa)"
                elif sweep_var == "Section Width (b)":
                    sweep_vals = np.linspace(150, 350, 25)
                    current_val = inp_b
                    for v in sweep_vals:
                        vec = feat_vector.copy(); vec["b"] = v
                        sweep_y.append(float(np.exp(model_sw.predict(np.array([[vec[f] for f in FEATS]]))[0])))
                    sweep_x = sweep_vals
                    x_lbl = "Section Width b (mm)"
                elif sweep_var == "Section Height (h)":
                    sweep_vals = np.linspace(150, 700, 25)
                    current_val = inp_h
                    for v in sweep_vals:
                        vec = feat_vector.copy(); vec["h"] = v
                        sweep_y.append(float(np.exp(model_sw.predict(np.array([[vec[f] for f in FEATS]]))[0])))
                    sweep_x = sweep_vals
                    x_lbl = "Section Height h (mm)"
                else:
                    sweep_vals = np.linspace(120.0, 420.0, 25)
                    current_val = inp_Ef
                    for v in sweep_vals:
                        vec = feat_vector.copy(); vec["Efeq"] = v
                        sweep_y.append(float(np.exp(model_sw.predict(np.array([[vec[f] for f in FEATS]]))[0])))
                    sweep_x = sweep_vals
                    x_lbl = "FRP Elastic Modulus Ef (GPa)"

                fig_sw = go.Figure()
                fig_sw.add_trace(go.Scatter(
                    x=sweep_x, y=sweep_y, mode='lines+markers', name='Predicted Tu',
                    line=dict(color='#1e3a8a', width=2.5), marker=dict(size=6, color='#2563eb')
                ))
                fig_sw.add_trace(go.Scatter(
                    x=[current_val], y=[Tu_pred], mode='markers', name='Current Design Point',
                    marker=dict(color='#dc2626', size=13, symbol='star', line=dict(color='white', width=1.5))
                ))
                fig_sw.add_hline(y=inp_Td, line_dash="dash", line_color="#dc2626", annotation_text=f"Demand Td = {inp_Td:.1f} kN·m")
                fig_sw.update_layout(
                    title=f"Parametric Sensitivity: Tu vs {x_lbl}",
                    title_font={'size': 13.5, 'color': '#0f2942', 'family': 'Inter, sans-serif'},
                    height=320,
                    margin=dict(l=10, r=10, t=40, b=35),
                    xaxis=dict(title=x_lbl, gridcolor="#f1f5f9"),
                    yaxis=dict(title="Torsional Capacity Tu (kN·m)", gridcolor="#f1f5f9"),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font_color="#0f2942"
                )
                st.plotly_chart(fig_sw, use_container_width=True)

        except Exception as e:
            st.error(f"Inference error: {str(e)}")


# =============================================================================
# TAB 2: BATCH CSV INGESTION & EXPORT ENGINE
# =============================================================================

with tab_batch:
    st.markdown("### Batch Structural Evaluation Engine")
    st.caption("Download the standardized CSV schema, evaluate multi-specimen datasets with 95% split-conformal coverage bounds, and export verified capacity assessments.")
    
    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded_csv = st.file_uploader("Upload Beam Configurations CSV:", type=["csv"])
    with col_info:
        sample_template_df = pd.DataFrame([{
            'beam_id': 'Beam-001', 'fc': 35.0, 'b': 250, 'h': 400, 'fyl': 400.0, 'fyt': 400.0,
            'rho_l': 0.015, 'rho_t': 0.004, 'rho_f': 0.003, 'Efeq': 230.0, 'wf_sf': 1.0, 'Td': 40.0
        }, {
            'beam_id': 'Beam-002', 'fc': 45.0, 'b': 300, 'h': 500, 'fyl': 400.0, 'fyt': 400.0,
            'rho_l': 0.020, 'rho_t': 0.005, 'rho_f': 0.005, 'Efeq': 250.0, 'wf_sf': 0.67, 'Td': 65.0
        }])
        csv_buffer = io.StringIO()
        sample_template_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Standard CSV Template",
            data=csv_buffer.getvalue(),
            file_name="batch_beams_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    if uploaded_csv is not None:
        try:
            batch_df = pd.read_csv(uploaded_csv)
            st.success(f"Loaded {len(batch_df)} beam configurations.")
            missing_feats = [f for f in FEATS if f not in batch_df.columns]
            if missing_feats:
                st.warning(f"Missing required feature columns: {missing_feats}")
            else:
                model_b = MODELS.get("Stacking Ensemble") or MODELS.get("CatBoost") or MODELS.get("XGBoost")
                X_b = batch_df[FEATS].values.astype(float)
                preds_log = model_b.predict(X_b)
                batch_df["Tu_pred (kN·m)"] = np.exp(preds_log).round(2)
                
                q_95 = 0.06
                if META and "conformal_q" in META:
                    q_95 = META["conformal_q"].get("Stacking", {}).get("95", 0.06)
                batch_df["Tu_min_95%"] = np.exp(preds_log - q_95).round(2)
                batch_df["Tu_max_95%"] = np.exp(preds_log + q_95).round(2)
                
                if "Td" in batch_df.columns:
                    batch_df["Safety_Factor"] = (batch_df["Tu_pred (kN·m)"] / batch_df["Td"]).round(2)
                    batch_df["Status"] = batch_df["Safety_Factor"].apply(lambda v: "PASS" if v >= 1.0 else "FAIL")
                    
                st.dataframe(batch_df, use_container_width=True)
                
                res_buf = io.StringIO()
                batch_df.to_csv(res_buf, index=False)
                st.download_button(
                    label="Download Evaluated Results CSV",
                    data=res_buf.getvalue(),
                    file_name="batch_evaluated_results.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# =============================================================================
# TAB 3: PUBLICATION PLOT & FIGURE VISUALIZER (STRICT ARTICLE ORDER, NO DUPLICATES)
# =============================================================================

with tab_plots:
    st.markdown("### Manuscript Figure Gallery")
    st.caption("Canonical sequence of 24 publication figures arranged in manuscript progression (flowcharts excluded). Standalone high-resolution figures are prioritized with dual 600 DPI PNG and vector PDF downloads.")
    
    # Filter Controls
    section_options = ["All Manuscript Figures (Figs. 1 - 24)"] + sorted(list(set(item["section"] for item in CANONICAL_MANUSCRIPT_FIGURES)))
    col_sec, col_search, col_layout = st.columns([2, 1.8, 1])
    with col_sec:
        selected_sec = st.selectbox("Filter by Article Section:", section_options, index=0)
    with col_search:
        search_kw = st.text_input("Search Figures:", placeholder="e.g. monte carlo, shap, parity, correlation...")
    with col_layout:
        view_cols = st.radio("Grid Columns:", [1, 2], index=1, horizontal=True)

    # Filtered list
    display_figs = CANONICAL_MANUSCRIPT_FIGURES
    if selected_sec != "All Manuscript Figures (Figs. 1 - 24)":
        display_figs = [f for f in display_figs if f["section"] == selected_sec]
    if search_kw.strip():
        kw = search_kw.strip().lower()
        display_figs = [f for f in display_figs if kw in f["title"].lower() or kw in f["subtitle"].lower() or kw in f["rel_path"].lower()]

    st.markdown(f"**Displaying {len(display_figs)} figures** (out of 24 canonical figures in manuscript)")
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

    # Render Grid
    grid_c = st.columns(view_cols)
    for idx, f_item in enumerate(display_figs):
        full_png = os.path.join(WORKSPACE_DIR, f_item["rel_path"])
        full_pdf = os.path.splitext(full_png)[0] + ".pdf"
        has_pdf = os.path.exists(full_pdf)
        
        with grid_c[idx % view_cols]:
            st.markdown('<div class="fig-item-card">', unsafe_allow_html=True)
            
            # Figure Metadata Header
            st.markdown(f"<span class='fig-index-tag'>{f_item['section']}</span>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.02rem; font-weight:700; color:#0f2942; line-height:1.35;'>{f_item['title']}</div>", unsafe_allow_html=True)
            st.caption(f"{f_item['subtitle']}")
            
            # Display Image
            if os.path.exists(full_png):
                try:
                    img = Image.open(full_png)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Image load error: {e}")
                    
                meta = get_image_info(full_png)
                st.caption(f"Resolution: `{meta['dims']}` | Size: `{meta['size_str']}` | Timestamp: `{meta['mtime']}`")
                
                # Download Buttons Row
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    with open(full_png, "rb") as f_png:
                        st.download_button(
                            label="Download PNG (600 DPI)",
                            data=f_png.read(),
                            file_name=os.path.basename(full_png),
                            mime="image/png",
                            key=f"png_btn_{f_item['id']}",
                            use_container_width=True
                        )
                with btn_c2:
                    if has_pdf:
                        with open(full_pdf, "rb") as f_pdf:
                            st.download_button(
                                label="Download Vector PDF",
                                data=f_pdf.read(),
                                file_name=os.path.basename(full_pdf),
                                mime="application/pdf",
                                key=f"pdf_btn_{f_item['id']}",
                                use_container_width=True
                            )
                    else:
                        st.button("No PDF Available", disabled=True, key=f"nopdf_{f_item['id']}", use_container_width=True)
            else:
                st.warning(f"File not found on disk: {f_item['rel_path']}")
                
            st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 4: MODEL ZOO & BENCHMARKS
# =============================================================================

with tab_models:
    st.markdown("### Candidate Model Zoo & Validation Benchmarks")
    st.caption("Quantitative benchmarking across 10 surrogate learners evaluated on 10-Fold Cross-Validation (3,000 samples), Validation Set (1,000 samples), and Holdout Test Set (1,000 samples).")
    
    perf_csv = os.path.join(ML_OUT_DIR, "table_model_performance_cv_val_test.csv")
    if os.path.exists(perf_csv):
        df_perf = pd.read_csv(perf_csv)
        st.markdown("#### 1. Candidate Model Performance Benchmark")
        st.dataframe(df_perf, use_container_width=True, hide_index=True)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if "Test_R2" in df_perf.columns:
                fig_r2 = px.bar(
                    df_perf.sort_values("Test_R2"),
                    x="Test_R2", y="Model", orientation="h",
                    title="Holdout Test Set Accuracy R² (Higher is Better)",
                    color="Test_R2", color_continuous_scale="Blues", text_auto=".4f"
                )
                fig_r2.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font_color="#0f2942", xaxis=dict(gridcolor="#f1f5f9"))
                st.plotly_chart(fig_r2, use_container_width=True)
        with col_p2:
            if "Test_RMSE" in df_perf.columns:
                fig_rmse = px.bar(
                    df_perf.sort_values("Test_RMSE", ascending=False),
                    x="Test_RMSE", y="Model", orientation="h",
                    title="Holdout Test Set Error RMSE (kN·m, Lower is Better)",
                    color="Test_RMSE", color_continuous_scale="Reds", text_auto=".2f"
                )
                fig_rmse.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font_color="#0f2942", xaxis=dict(gridcolor="#f1f5f9"))
                st.plotly_chart(fig_rmse, use_container_width=True)
                
    st.divider()
    
    col_hpo, col_uq = st.columns(2)
    with col_hpo:
        hpo_csv = os.path.join(ML_OUT_DIR, "table_hyperparameters_optimization.csv")
        if os.path.exists(hpo_csv):
            st.markdown("#### 2. Optuna Hyperparameter Optimization Details")
            st.dataframe(pd.read_csv(hpo_csv), use_container_width=True, hide_index=True)
    with col_uq:
        uq_csv = os.path.join(ML_OUT_DIR, "uq_comparison.csv")
        if os.path.exists(uq_csv):
            st.markdown("#### 3. Uncertainty Quantification Comparison")
            st.dataframe(pd.read_csv(uq_csv), use_container_width=True)


# =============================================================================
# TAB 5: PIPELINE LAUNCHER & LIGHT CONSOLE
# =============================================================================

@st.fragment(run_every=2)
def render_light_console():
    jobs_list = job_manager.get_all_jobs()
    if not jobs_list:
        st.info("No background pipeline jobs logged.")
        return
        
    job_ids = [j["job_id"] for j in jobs_list]
    default_idx = 0
    if "selected_job_id" in st.session_state and st.session_state["selected_job_id"] in job_ids:
        default_idx = job_ids.index(st.session_state["selected_job_id"])
        
    def fmt_j(jid):
        j = next((x for x in jobs_list if x["job_id"] == jid), None)
        if not j: return jid
        ico = "Running" if j["status"] == "RUNNING" else ("Done" if j["status"] in ["COMPLETED", "FINISHED"] else "Stopped")
        return f"[{ico}] [{j['start_time'].split()[-1]}] {j['name']} (PID: {j.get('pid', 'N/A')})"
        
    sel_jid = st.selectbox("Select Background Job to Monitor:", job_ids, index=default_idx, format_func=fmt_j, key="log_sel_tab")
    st.session_state["selected_job_id"] = sel_jid
    job = job_manager.get_job(sel_jid)
    if not job: return
    
    st_col1, st_col2, st_col3 = st.columns([2, 2, 2])
    with st_col1:
        st.markdown(f"**Status:** `{job['status']}` (PID `{job.get('pid')}`)")
    with st_col2:
        st.markdown(f"**Started:** `{job['start_time']}`")
    with st_col3:
        if job["status"] == "RUNNING":
            if st.button("Stop Process", key=f"stop_{job['job_id']}"):
                job_manager.kill_job(job["job_id"])
                st.toast(f"Stopped PID {job['pid']}")
                st.rerun()
                
    raw_log = job_manager.read_log_tail(job.get("log_path", ""), max_lines=300)
    st.markdown(f'<div class="light-console">{raw_log if raw_log.strip() else "[Log is currently empty - process initializing...]"}</div>', unsafe_allow_html=True)
    st.caption("Console auto-refreshes periodically during background runs.")

with tab_console:
    st.markdown("### Pipeline Execution Console & Diagnostics")
    st.caption("Execute background figure generation suites, retrain models, or monitor real-time execution logs.")
    
    col_l1, col_l2 = st.columns([1, 1], gap="large")
    with col_l1:
        st.markdown("#### Dispatch Pipeline Task")
        p_mode = st.selectbox("Select Task Profile:", [
            "All Publication Figures (smmt_figures.py)",
            "Full ML Training Pipeline (smmt_ml_pipeline.py)"
        ])
        if st.button("Dispatch Background Task", type="primary", use_container_width=True):
            if "Figures" in p_mode:
                cmd = [PYTHON_EXE, "-u", "smmt_figures.py"]
                nm = "Publication Figures"
            else:
                cmd = [PYTHON_EXE, "-u", "smmt_ml_pipeline.py"]
                nm = "ML Pipeline"
            new_job = job_manager.launch_job(nm, cmd, p_mode)
            st.session_state["selected_job_id"] = new_job["job_id"]
            st.toast(f"Dispatched {nm}")
            st.rerun()
            
    with col_l2:
        st.markdown("#### Live Execution Console")
        render_light_console()
