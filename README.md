# MSMMT-FRP: Online Torsional Design & Surrogate Modeling Portal for FRP-Strengthened RC Beams

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SCIE Q1 Research](https://img.shields.io/badge/Research-SCIE%20Q1%20Peer--Reviewed-darkgreen.svg)](#citation)

---

## Overview

This repository hosts the official web application and surrogate design environment for predicting the ultimate torsional capacity ($T_u$) of reinforced concrete (RC) beams strengthened with externally bonded Fiber-Reinforced Polymer (FRP) composites.

The framework integrates:
1. **Mechanical Rigor**: Grounded in the **Modified Softened Membrane Model for Torsion (MSMMT-FRP)** incorporating cyclic concrete softening, rotating crack angle theory, and FRP stress-strain compatibility.
2. **High-Accuracy Surrogate Engine**: Benchmarked across 10 candidate machine learning algorithms on 5,000 high-fidelity SMMT solves, with the optimized **Stacking Ensemble** achieving $R^2 = 0.9965$ and $\text{RMSE} = 2.16\text{ kN}\cdot\text{m}$ on unseen test data.
3. **Split-Conformal Uncertainty Quantification (UQ)**: Distribution-free prediction intervals with mathematically guaranteed 90% and 95% coverage probabilities.
4. **Explainable AI (XAI)**: Real-time local SHAP (Shapley Additive exPlanations) attribution and continuous parametric sensitivity analysis.
5. **Publication Figure Visualizer**: Interactive access to all 24 canonical manuscript figures with dual high-resolution 600 DPI PNG and publication-ready vector PDF downloads.

---

## Live Web Application

The interactive web portal is deployed and publicly accessible via Streamlit Community Cloud:
👉 **[Launch SMMT-FRP Design Portal](https://share.streamlit.io)** *(Replace with your deployed URL)*

---

## Repository Structure

```text
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python package dependencies
├── job_manager.py              # Background execution and process monitor
├── batch_beams_template.csv    # Standard CSV schema for batch evaluations
├── .streamlit/
│   └── config.toml             # Academic theme & headless deployment configuration
├── ml_out/                     # Trained surrogate model artifacts & metadata
│   ├── stack_Tu.joblib         # Optimized Stacking Ensemble model
│   ├── catboost_Tu.cbm         # CatBoost model
│   ├── xgb_Tu.json             # XGBoost model
│   ├── lgb_Tu.joblib           # LightGBM model
│   ├── inference_meta.json     # Feature scaling, conformal quantiles & bounds
│   └── *.csv                   # Benchmark evaluation & HPO tables
├── figures/                    # 24 publication figures (600 DPI PNG & vector PDF)
└── README.md                   # Project documentation & citation guide
```

---

## Local Installation & Execution

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/smmt-frp-torsion-portal.git
cd smmt-frp-torsion-portal
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Deployment to Streamlit Community Cloud (1-Click)

1. Push this repository to your GitHub account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub credentials.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
5. Click **"Deploy"**. The app will automatically build and launch within 2–3 minutes.

---

## Input Parameter Specifications

| Parameter | Symbol | Unit | Physical Interpretation |
| :--- | :---: | :---: | :--- |
| Section Width | $b$ | mm | Total beam cross-section width |
| Section Height | $h$ | mm | Total beam cross-section depth |
| Concrete Strength | $f'_c$ | MPa | Standard concrete cylinder compressive strength |
| Long. Steel Yield | $f_{yl}$ | MPa | Yield strength of longitudinal reinforcing bars |
| Stirrup Steel Yield | $f_{yt}$ | MPa | Yield strength of transverse steel stirrups |
| Long. Reinforcement Ratio | $\rho_l$ | % | $A_{sl} / (b \cdot h)$ |
| Stirrup Ratio | $\rho_t$ | % | $(A_{st} \cdot p_{st}) / (b \cdot h \cdot s)$ |
| FRP Volumetric Ratio | $\rho_f$ | % | $(2 \cdot n_f \cdot t_f \cdot w_f \cdot (b+h)) / (b \cdot h \cdot s_f)$ |
| Equivalent FRP Modulus | $E_{feq}$ | GPa | Elastic tensile modulus of FRP composite |
| FRP Continuity Ratio | $w_f / s_f$ | - | Ratio of strip width to center-to-center spacing ($=1.0$ for continuous jacket) |

---

## Citation

If you use this portal or surrogate models in your academic research, please cite our manuscript:

```bibtex

```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
