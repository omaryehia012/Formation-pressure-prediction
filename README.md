## Kick Detection System: Formation Pressure Prediction and Safety Assessment

This repository contains a complete, end‑to‑end implementation of a machine learning system that predicts formation pressure from real‑time drilling and petrophysical parameters, visualizes safety margins against hydrostatic pressure, and supports kick detection decision‑making. It includes:

- A Streamlit application (`kick_detection_app_All.py`) that performs prediction, safety analysis, and interactive visualization
- Trained models and scalers saved under `models/`
- Notebooks used for data exploration, feature engineering, and model training under `Code For AI Models/`
- A curated set of figures and plots under `Graphs/` to document data analysis and model performance

The project is tailored to petroleum drilling operations with a focus on early kick detection by estimating formation pressure and comparing it to the computed hydrostatic pressure from mud weight and TVD. When hydrostatic pressure drops below formation pressure, a kick risk is flagged with actionable recommendations.


### Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement and Case Context](#2-problem-statement-and-case-context)
- [3. Domain Background: Formation Pressure and Kicks](#3-domain-background-formation-pressure-and-kicks)
- [4. Data Sources and Feature Dictionary](#4-data-sources-and-feature-dictionary)
- [5. Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
- [6. Modeling Strategy](#6-modeling-strategy)
- [7. Classification vs Regression: What and Why](#7-classification-vs-regression-what-and-why)
- [8. Model Zoo and Rationale](#8-model-zoo-and-rationale)
- [9. Training Workflow](#9-training-workflow)
- [10. Model Evaluation and Selection](#10-model-evaluation-and-selection)
- [11. Production App: Streamlit UI/UX](#11-production-app-streamlit-uiux)
- [12. Installation and Quick Start](#12-installation-and-quick-start)
- [13. Running the Application](#13-running-the-application)
- [14. Reproducibility: Notebooks and Seeds](#14-reproducibility-notebooks-and-seeds)
- [15. Configuration, Paths, and File Layout](#15-configuration-paths-and-file-layout)
- [16. Safety Analysis Logic](#16-safety-analysis-logic)
- [17. Practical Guidance for Engineers](#17-practical-guidance-for-engineers)
- [18. Troubleshooting](#18-troubleshooting)
- [19. Roadmap and Future Work](#19-roadmap-and-future-work)
- [20. FAQ](#20-faq)
- [21. Glossary](#21-glossary)
- [22. Citations and Further Reading](#22-citations-and-further-reading)
- [23. License and Disclaimer](#23-license-and-disclaimer)


## 1. Project Overview

The Kick Detection System estimates formation pressure using supervised machine learning trained on historical drilling and petrophysical logs. The core output is a numeric formation pressure estimate (PSI). The application also computes hydrostatic pressure from mud density and true vertical depth (TVD), and presents a clear “SAFE / CRITICAL” indicator for kick risk.

Key characteristics:

- Input parameters include drilling variables (ROP, WOB, RPM, Torque), surface pressures (Stand Pipe Pressure), flow and temperature data (Flow In, Temp Out), and petrophysical descriptors (Corrected Bulk Density, Deep Resistivity, Neutron Porosity).
- Model family evaluated includes Linear, Ridge, Lasso, Elastic Net, KNN, Decision Tree, Random Forest, SVR, and XGBoost. The deployed model in `kick_detection_app_All.py` loads `models/model_All.h5` with its corresponding scaler `models/scaler_All.h5`.
- UI/UX is built with Streamlit, offering structured parameter input, clear metrics, and an actionable, color‑coded safety summary.


## 2. Problem Statement and Case Context

In drilling operations, maintaining a safe margin between hydrostatic pressure in the wellbore and formation pressure is essential to prevent influx (kick) events. Underbalanced conditions, where hydrostatic pressure is less than formation pressure, can cause formation fluids to enter the wellbore. Rapid and reliable estimation of formation pressure helps engineers adjust mud weight and other parameters proactively.

This project targets:

- Predicting formation pressure from readily available measurements
- Providing a transparent safety margin and status indicator
- Serving as a decision aid for adjusting mud weight, flow, and operational parameters

The solution is research‑oriented (Master’s project) yet practical, designed for operational awareness, training, and as a foundation for integration into real‑time drilling advisory systems.


## 3. Domain Background: Formation Pressure and Kicks

Formation pressure is the pressure of fluids within the rock pores. In conventional overbalanced drilling, mud hydrostatic pressure intentionally exceeds formation pressure to avoid influx.

- Hydrostatic pressure (PSI) is estimated as:
  - P_h = 0.052 × MW(ppg) × TVD(ft)
- Kick risk increases if P_h < P_f (formation pressure).
- Operational mitigations include increasing mud weight (within fracture gradient limits), adjusting choke/flow, and controlling penetration rates.

The model does not replace well control procedures but augments situational awareness by providing a continuous estimate of formation pressure alongside clear visual cues.


## 4. Data Sources and Feature Dictionary

The app expects 13 input features matching model training. The features are entered through the sidebar in the Streamlit UI and are assembled into a `pandas.DataFrame` with the following schema:

- `TVD(ft)` — True Vertical Depth in feet
- `BITSIZE(in)` — Bit size in inches
- `NPHI(%)` — Neutron porosity percentage
- `Corrected Bulk Density(gm/cc)` — Corrected bulk density in g/cc
- `Deep Resistivity (Ohm)` — Deep resistivity in ohm‑m (Ohm)
- `ROP(M/hr)` — Rate of penetration in meters per hour
- `WOB(KLb)` — Weight on bit in Kilo‑pounds
- `RPM` — Rotations per minute
- `Torque(lb.F)` — Torque in pound‑feet
- `Stand Pipe Pressure(Psi)` — Standpipe pressure in PSI
- `Flow In(GPM)` — Flow rate in gallons per minute
- `Temp - Out` — Outlet temperature (°F)
- `Total Gas(PPM)` — Total gas concentration in parts per million

Additional runtime input (not fed to the ML model but used by the app):

- `Mud Density (PPG)` — Used to compute hydrostatic pressure for the safety analysis

Reference figures for the above features and their relationships can be found under `Graphs/`, including:

- `Graphs/Correlation Matrix.png`
- `Graphs/Pairplot.png`
- `Graphs/Features/` for per‑feature distributions
- `Graphs/Features with depth/` for trends vs depth


## 5. Exploratory Data Analysis (EDA)

The notebooks in `Code For AI Models/` document the EDA and model development process:

- `Kick Detection System All Data .ipynb`
- `Kick Detection System.ipynb`

Typical EDA steps include:

- Data cleaning and unit consistency checks
- Outlier identification for pressure and gas channels
- Correlation analysis between drilling parameters and formation pressure
- Trend analysis with depth and operational changes
- Feature scaling and transformation as needed for model families

Visual evidence and insights from the EDA are exported into the `Graphs/` directory, including per‑feature distributions, target relationships, and model performance visuals.


## 6. Modeling Strategy

The primary modeling objective is numeric estimation of formation pressure (regression). Multiple algorithms are benchmarked to balance bias/variance trade‑offs and operational interpretability. The final production model is chosen based on cross‑validated error metrics, robustness to outliers, and stability across operational regimes.

General approach:

1. Prepare feature matrix X and target y (formation pressure)
2. Split into train/validation/test sets
3. Scale features where applicable (e.g., StandardScaler, MinMaxScaler)
4. Tune models using cross‑validation or randomized/grid search
5. Compare models on MAE, RMSE, R² and operational error bands (e.g., ±250 PSI)
6. Select the best model for deployment and persist model + scaler with `joblib`

Persisted artifacts used by the app:

- `models/model_All.h5` — Trained regressor (e.g., XGBoost/ensemble)
- `models/scaler_All.h5` — Corresponding scaler for input standardization


## 7. Classification vs Regression: What and Why

Two common supervised learning paradigms apply to kick detection contexts:

- Classification models output categories (e.g., “Kick” vs “No Kick”). They are trained on labels derived from events or thresholds and evaluated with metrics like accuracy, precision/recall, F1, ROC‑AUC.
- Regression models output continuous values (e.g., formation pressure in PSI). They are evaluated with MAE, RMSE, MAPE, R², and operational error tolerances.

In this project, regression is preferred for the core task of estimating formation pressure because:

- It preserves quantitative information important for engineering judgment and safety margins.
- The safety decision can be derived deterministically by comparing P_h and P_f, avoiding information loss from early binarization.
- It enables graded recommendations (magnitude of underbalance) instead of a single class label.

However, classification can be layered on top by defining a risk label such as `KickRisk = 1 if P_f > P_h else 0`, enabling alerting, imbalanced learning techniques, and event prediction workflows. Both paradigms can coexist: regression for magnitude, classification for alerting.


## 8. Model Zoo and Rationale

Models explored in the notebooks and visualized under `Graphs/Models/` include:

- Linear Regression, Ridge, Lasso, Elastic Net — Baselines with different regularization profiles for interpretability and bias control
- K‑Nearest Neighbors (KNN) — Non‑parametric baseline that can capture local structure
- Decision Tree — High‑variance baseline, interpretable structure
- Random Forest — Ensemble of trees improving variance reduction and feature interaction capture
- Support Vector Regression (SVR) — Strong performance with kernels on structured, scaled data
- XGBoost — Gradient boosting tree ensemble with excellent performance on tabular data

Each model’s performance is tracked in terms of MAE, RMSE, R² and operational tolerance bands, with selections motivated by robustness and stability. The Streamlit app currently loads the best persisted “All Data” model and scaler.


## 9. Training Workflow

High‑level training process (see notebooks for full details):

1. Data ingestion and validation
2. Feature engineering and unit normalization
3. Train/validation/test split with temporal or depth‑aware logic when applicable
4. Scaling/standardization pipeline (persisted)
5. Model hyperparameter tuning (grid/random search or early stopping for boosters)
6. Cross‑validation and error analysis, including depth‑stratified checks
7. Final model fit on full training set and evaluation on held‑out test set
8. Persist artifacts with `joblib.dump` to `models/*.h5`
9. Export plots to `Graphs/Models/` and `Graphs/Features*/`


## 10. Model Evaluation and Selection

Common metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Mean Absolute Percentage Error (MAPE)
- Operational tolerance thresholds (e.g., ±250 PSI)

Diagnostic plots in `Graphs/Models/*` include per‑model error comparisons, predicted‑vs‑actual scatter, and residual distributions. The final selection balances raw accuracy, stability across ranges of TVD and mud weight, and resilience to outliers.


## 11. Production App: Streamlit UI/UX

The Streamlit application provides a polished interface for prediction and safety assessment.

Highlights:

- Sidebar input sections: Well Parameters, Formation Properties, Drilling Parameters, Mud & Flow Parameters
- Predict button triggers scaling + model inference and stores results in `st.session_state`
- Results area shows:
  - Kick Detection Result with color‑coded status
  - Metric cards for Formation Pressure, Hydrostatic Pressure, and Safety Margin
  - Plotly bar chart comparing pressures
  - Safety analysis box with percentage margin and recommendation
- Input summary table reflects entered parameters for record‑keeping

Visuals and theme:

- Custom CSS, gradients, and glassmorphism styling
- Background image `R.jfif`
- Consistent `Inter` font family and responsive layout


## 12. Installation and Quick Start

Prerequisites:

- Windows 10/11 with PowerShell
- Python 3.10+ recommended

Steps:

1. Clone or copy the repository directory to your machine
2. Create and activate a virtual environment
3. Install dependencies from `requirements.txt`
4. Run the Streamlit app

Windows PowerShell example:

```powershell
# Navigate to the project folder
cd "D:\Python ENV\Petroleum Projects\Formation Pressure Detection - Gendy"

# Create venv (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Launch the app
streamlit run kick_detection_app_All.py
```

If your default shell restricts script execution, you may need to allow running the activation script once:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```


## 13. Running the Application

At app startup, the following happens:

- Page is configured with wide layout and the “🛢️” icon
- Model artifacts load from `models/scaler_All.h5` and `models/model_All.h5`
- Custom CSS is injected to style the page

User flow:

1. Enter inputs in the sidebar
2. Click “Predict Formation Pressure”
3. Review results: numeric pressures, safety margin (PSI and %), and recommendations
4. Use the safety status (SAFE/CRITICAL) to guide operational consideration

Inputs expected by the model are strictly the 13 features listed in [Data Sources and Feature Dictionary](#4-data-sources-and-feature-dictionary). Mud Density is used only for hydrostatic pressure calculation.


## 14. Reproducibility: Notebooks and Seeds

Notebooks in `Code For AI Models/` are the ground truth for data preparation and model selection. To reproduce experiments:

1. Open the notebooks and execute cells in order
2. Ensure consistent package versions (see `requirements.txt`)
3. Use fixed random seeds where available
4. Export trained artifacts with `joblib.dump` to `models/*.h5`

Note: The `.h5` extension is used here for persisted `joblib` objects (not HDF5 neural nets). This improves portability across the app and notebooks.


## 15. Configuration, Paths, and File Layout

Relevant file tree excerpt:

```
Formation Pressure Detection - Gendy/
  kick_detection_app_All.py
  models/
    model_All.h5
    scaler_All.h5
  Code For AI Models/
    Kick Detection System All Data .ipynb
    Kick Detection System.ipynb
  Graphs/
    Correlation Matrix.png
    Pairplot.png
    Models/
      Models Performance.png
    Features/
    Features with depth/
  requirements.txt
  README.md
  R.jfif
```

If you change model filenames or paths, update the loader in `kick_detection_app_All.py` under the `load_model()` function accordingly.


## 16. Safety Analysis Logic

Core computations in the app:

- Formation pressure estimate: `P_f` from ML model output (PSI)
- Hydrostatic pressure: `P_h = 0.052 × MW(ppg) × TVD(ft)`
- Safety margin: `|P_h − P_f|` with SAFE if `P_h > P_f`, CRITICAL if `P_h <= P_f`

Recommendations:

- SAFE: Maintain parameters and continue monitoring
- CRITICAL: Increase mud weight (respecting fracture gradient and ECD limits) or reduce TVD and reassess

The app presents both PSI and percentage margin for quick interpretation. It is meant to complement, not replace, well control practices.


## 17. Practical Guidance for Engineers

Operational usage tips:

- Confirm units (PPG, PSI, ft) for consistency
- Re‑evaluate Mud Density if CRITICAL is reached with small margins
- Track trends with depth: if the model predicts rising formation pressure, plan ahead for mud weight adjustments
- Use the Input Summary table to record decisions

Integration ideas:

- Feed real‑time rig data streams into the app
- Add alarms when `P_f − P_h` exceeds a threshold for N seconds
- Couple with hydraulics module for dynamic ECD simulation


## 18. Troubleshooting

Common issues:

- Model load error: Ensure `models/model_All.h5` and `models/scaler_All.h5` exist and are compatible with the installed libraries
- Streamlit fails to start: Check Python version and reinstall requirements
- Prediction seems off: Verify input units and ranges, compare with notebook validation examples
- Background image not visible: Confirm `R.jfif` is present in the project root

Diagnostics:

- Launch Streamlit with `--server.enableXsrfProtection=false` if behind certain proxies (only if necessary)
- Print intermediate values in the app to debug inputs and scaling (temporary changes for debugging)


## 19. Roadmap and Future Work

- Add uncertainty quantification (prediction intervals)
- Implement hybrid approach: regression for magnitude + classification for alerts
- Expand feature set (ECD, annular pressure, MWD/LWD advanced logs)
- Continual learning with online updates as new wells are drilled
- Multi‑well generalization and domain adaptation
- Role‑based UI with operator and engineer modes


## 20. FAQ

- Why not predict a binary “kick/no‑kick” directly?
  - Predicting formation pressure preserves more information and supports nuanced decision‑making. A binary label can be derived post‑hoc from the pressure comparison.

- Can I swap in a different model?
  - Yes. Persist your new model and scaler to `models/` and ensure the input feature order matches. Update the loader if filenames change.

- Do I need GPUs?
  - No. Inference is CPU‑friendly; training boosters like XGBoost can benefit from larger CPUs/GPUs but is not required to run the app.

- How accurate is the model?
  - See `Graphs/Models/Models Performance.png` and notebook evaluations. Accuracy depends on data quality and representativeness.


## 21. Glossary

- Formation Pressure (P_f): Pressure of fluids in the formation pores
- Hydrostatic Pressure (P_h): Pressure exerted by the mud column at a given TVD
- Kick: Uncontrolled influx of formation fluids into the wellbore
- Mud Weight (MW): Density of drilling mud, usually in pounds per gallon (PPG)
- TVD: True Vertical Depth
- ROP: Rate of Penetration
- WOB: Weight on Bit
- RPM: Rotations Per Minute
- SVR: Support Vector Regression
- MAE/RMSE/R²: Common regression error metrics


## 22. Citations and Further Reading

- Bourgoyne, A. T., Millheim, K. K., Chenevert, M. E., Young, F. S. Applied Drilling Engineering. SPE.
- Rabia, H. Well Engineering & Construction.
- Montgomery, D. C., Runger, G. C. Applied Statistics and Probability for Engineers.
- XGBoost: Scalable Tree Boosting (`https://xgboost.readthedocs.io/`)
- Streamlit Docs (`https://docs.streamlit.io/`)
- scikit‑learn User Guide (`https://scikit-learn.org/stable/user_guide.html`)


## 23. License and Disclaimer

This project is provided for research and educational purposes. Operational use should be accompanied by proper well control procedures, engineering review, and adherence to safety standards. The authors and contributors are not liable for decisions made based on the outputs of this software.

Copyright © 2024 — Advanced Formation Pressure Analysis System

---

## Appendix A. Extended Overview of AI Models (Case‑Focused)

This appendix provides an in‑depth, practical summary of the model families evaluated for formation pressure prediction, with guidance on when to favor each in a drilling context.

### A.1 Linear Models (Linear, Ridge, Lasso, Elastic Net)

- What they are: Linear models assume a linear relationship between features and target. Ridge applies L2 regularization; Lasso applies L1; Elastic Net blends L1/L2.
- Strengths:
  - High interpretability; coefficients reflect directional influence
  - Fast to train and predict; robust baselines
  - Regularization controls overfitting with noisy sensors
- Limitations:
  - Struggle with strong nonlinear interactions (e.g., depth × resistivity effects)
  - Sensitive to multicollinearity if not regularized
- When to use:
  - Early feasibility checks; explainability prioritized
  - Stable regimes where relationships are near‑linear
- Notes for this case:
  - Useful as a sanity check and to detect unexpected sign directions on features (e.g., WOB, ROP)
  - Pair with polynomial features if nonlinearity is mild

### A.2 K‑Nearest Neighbors (KNN)

- What it is: Non‑parametric method predicting target from the average of the K closest training points in feature space.
- Strengths:
  - Captures local structure; simple and intuitive
- Limitations:
  - Sensitive to feature scaling and irrelevant features
  - Slower at inference for very large datasets
- When to use:
  - As a non‑linear baseline to validate other models
- Notes for this case:
  - Works if operational modes cluster (e.g., particular mud systems and bit types)

### A.3 Decision Trees

- What they are: Hierarchical splits on features, forming piecewise constant predictions
- Strengths:
  - Interpretability via rules; handles feature interactions
  - Handles missing values and mixed feature types (with care)
- Limitations:
  - High variance; prone to overfitting
- When to use:
  - Rule extraction; interpretable prototypes
- Notes for this case:
  - Good for understanding key thresholds (e.g., Stand Pipe Pressure breakpoints)

### A.4 Random Forest

- What it is: Ensemble of decision trees trained on bootstrapped samples with feature subsampling
- Strengths:
  - Strong performance with less tuning
  - Robust to noise; captures interactions
- Limitations:
  - Less interpretable than single trees; larger models
- When to use:
  - Solid tabular default when explainability is moderate priority
- Notes for this case:
  - Stable; good balance between bias/variance across depths and lithologies

### A.5 Support Vector Regression (SVR)

- What it is: Margin‑based regression using kernels to model nonlinear relationships
- Strengths:
  - Strong performance on medium‑sized, well‑scaled datasets
- Limitations:
  - Sensitive to hyperparameters (C, epsilon, kernel parameters)
  - Training time grows quickly with dataset size
- When to use:
  - High‑quality, scaled features with complex but smooth nonlinearities
- Notes for this case:
  - Consider RBF kernel; grid or Bayesian search for C and gamma

### A.6 XGBoost (Gradient Boosted Trees)

- What it is: Additive ensemble of shallow trees trained sequentially to correct prior residuals
- Strengths:
  - State‑of‑the‑art on many tabular tasks; handles nonlinearity and interactions well
  - Built‑in regularization and early stopping; robust feature importance tools
- Limitations:
  - More hyperparameters to tune; risk of overfitting with small datasets if not regularized
- When to use:
  - Primary production candidate for tabular drilling data
- Notes for this case:
  - Typically the strongest performer for formation pressure estimation when trained with careful CV


## Appendix B. Classification vs Regression in Kick Detection

### B.1 Formulations

- Regression target: `P_f` (formation pressure) in PSI
- Derived classification: `KickRisk = 1 if P_f > P_h else 0`

### B.2 Advantages and Trade‑Offs

- Regression advantages:
  - Preserves magnitude information for nuanced decisions
  - Enables sensitivity analyses and what‑if scenarios (e.g., changing MW)
- Classification advantages:
  - Direct alerts and thresholds for operations
  - Imbalanced learning techniques (focal loss, class weighting) for rare events
- Combined approach:
  - Use regression for magnitude + threshold‑based classification for alarm; calibrate thresholds by well control policy

### B.3 Metric Alignment

- Regression metrics tie to engineering tolerance (e.g., ±250 PSI acceptable band)
- Classification metrics align to operational risk (recall for kick events often prioritized over precision)

### B.4 Recommended Practice for This Case

1. Train and deploy regression for `P_f`
2. Compute `P_h` from MW and TVD
3. Apply policy thresholds to derive risk levels (SAFE/CAUTION/CRITICAL)


## Appendix C. Feature Engineering and Scaling in Practice

### C.1 Scaling Strategy

- Standardization for models sensitive to scale (SVR, KNN, linear models)
- Store scaler with `joblib` and apply identical transform in the app

### C.2 Feature Checks

- Units: Ensure PPG/PSI/ft consistency
- Range validation: Clamp or flag extreme values (e.g., gas spikes)
- Missing values: Impute or block predictions with warnings

### C.3 Interaction Features (Optional)

- Examples: `TVD × MW`, `ROP × WOB`, `RPM × Torque`
- Use cautiously; tree boosters often capture interactions without manual creation


## Appendix D. Hyperparameter Guidance per Model

Below are common hyperparameters and practical ranges to explore:

- Linear/Ridge/Lasso/Elastic Net:
  - `alpha` (regularization strength): 1e‑4 to 1e2
  - `l1_ratio` for Elastic Net: 0.1 to 0.9
- KNN:
  - `n_neighbors`: 3 to 31 (odd values)
  - `weights`: uniform, distance
  - `metric`: euclidean, manhattan
- Decision Tree:
  - `max_depth`: 3 to 20
  - `min_samples_split`: 2 to 50
  - `min_samples_leaf`: 1 to 20
- Random Forest:
  - `n_estimators`: 100 to 1000
  - `max_depth`: 5 to None
  - `max_features`: sqrt, log2, 0.3–0.8
- SVR:
  - `C`: 0.1 to 100
  - `epsilon`: 0.01 to 5
  - `gamma` (RBF): 1e‑4 to 1
- XGBoost:
  - `n_estimators`: 200 to 2000 (use early stopping)
  - `max_depth`: 3 to 10
  - `learning_rate`: 0.01 to 0.2
  - `subsample`: 0.6 to 1.0
  - `colsample_bytree`: 0.6 to 1.0
  - `reg_alpha` (L1): 0 to 5
  - `reg_lambda` (L2): 0.1 to 10


## Appendix E. Cross‑Validation and Evaluation Details

### E.1 Splitting Strategy

- Random K‑fold as a baseline
- Depth‑aware or well‑aware splits to avoid leakage across similar regimes
- Temporal splits if order matters (e.g., drilling sequence)

### E.2 Error Bands and Engineering Tolerances

- Track % of predictions within ±100, ±250, ±500 PSI
- Report MAE/RMSE/R² alongside band compliance

### E.3 Residual Diagnostics

- Residual vs predicted plots to detect heteroscedasticity
- Error vs TVD/MW to check regime bias
- Influence of outliers (e.g., high gas, tool malfunction)


## Appendix F. Interpretability and Diagnostics

### F.1 Global Importance

- Tree ensembles: gain/weight/cover importances
- Permutation importance across models

### F.2 Local Explanations

- SHAP values to attribute feature contributions per prediction
- Use case: Explain CRITICAL flags by highlighting which inputs push `P_f` above `P_h`

### F.3 Sanity Checks

- Directionality: e.g., higher MW should raise `P_h`, not necessarily `P_f`
- Known physics: resistivity/porosity trends should align with formation pressure expectations per basin


## Appendix G. Operationalization and Monitoring

### G.1 Data Validation in Production

- Schema checks: presence of 13 features with valid types
- Range checks and unit enforcement
- Drift detection: compare recent distributions to training baselines

### G.2 Performance Monitoring

- Log predictions and inputs (with timestamps and depth)
- Periodic back‑checks against measured pressures when available
- Track CRITICAL event recall and false alarm rate if labels are later available

### G.3 Model Refresh

- Schedule retraining as data accumulates
- Maintain versioned artifacts in `models/` with changelog


## Appendix H. Risk Management and Limitations

- The system provides estimates, not guarantees; use professional judgment
- Input quality is critical; sensor faults can degrade predictions
- Geology changes and abnormal pressure zones may require model retraining
- Always align with well control procedures and regulatory requirements


## Appendix I. What‑If Analysis Examples

Use the app to run scenario exploration:

1. Fix TVD and inputs; vary Mud Density (PPG) to find the minimum overbalanced condition
2. Evaluate sensitivity by changing ROP/WOB and observing any shifts in predicted `P_f`
3. Compare SAFE margin as the well deepens; anticipate mud program changes


## Appendix J. Extended Troubleshooting

- Mismatch in feature order:
  - Ensure the app’s `Inputs` list matches the training feature order exactly
- Different scaler used:
  - Regenerate and persist the correct scaler; keep model and scaler paired
- Version drift:
  - Align `scikit‑learn` and `xgboost` versions with those used in training
- Serialization errors:
  - Re‑dump artifacts with `joblib` and test a dry‑run load in a clean venv


## Appendix K. Contribution Guide (Internal)

1. Create a feature branch
2. Update notebooks/graphs and regenerate artifacts as needed
3. Verify app runs locally end‑to‑end (prediction + safety analysis)
4. Open a PR summarizing changes to data, features, models, and any UI updates


## Appendix L. Extended FAQ

- Can I add more features (e.g., ECD, flow out)?
  - Yes. Update training, retrain model, export new scaler + model, and sync the `Inputs` list in the app.
- How do I interpret a small positive margin (SAFE but close)?
  - Treat as caution; consider operational buffers and measurement uncertainty.
- Can I run headless or as a service?
  - Streamlit is interactive; for services, wrap the model in a FastAPI/Flask endpoint and reuse the same scaler/model artifacts.


## Appendix M. Change Log (Summary)

- v1.0: Initial research app with XGBoost model and Streamlit UI; style improvements; safety analysis and charts; notebooks and graphs included.


## Appendix N. Extended References

- SPE/IADC papers on well control best practices and kick detection
- Gradient boosting and interpretability literature (SHAP)
- Petroleum engineering textbooks on pore pressure prediction and drilling hydraulics


## Appendix O. Legal and Ethical Considerations

- Do not rely solely on software outputs for safety‑critical operations
- Maintain data privacy and compliance when logging rig data
- Ensure model governance and auditability of decisions


## Appendix P. Quick Code Examples

Below are minimal Python examples to illustrate how the persisted scaler and model could be invoked programmatically outside Streamlit (for documentation only):

```python
import joblib
import pandas as pd

Inputs = ['TVD(ft)', 'BITSIZE(in)', 'NPHI(%)',
          'Corrected Bulk Density(gm/cc)', 'Deep Resistivity (Ohm)', 'ROP(M/hr)',
          'WOB(KLb)', 'RPM', 'Torque(lb.F)', 'Stand Pipe Pressure(Psi)',
          'Flow In(GPM)', 'Temp - Out', 'Total Gas(PPM)']

scaler = joblib.load('models/scaler_All.h5')
model = joblib.load('models/model_All.h5')

row = {
  'TVD(ft)': 5000.0,
  'BITSIZE(in)': 8.5,
  'NPHI(%)': 15.0,
  'Corrected Bulk Density(gm/cc)': 2.65,
  'Deep Resistivity (Ohm)': 50.0,
  'ROP(M/hr)': 30.0,
  'WOB(KLb)': 20.0,
  'RPM': 120,
  'Torque(lb.F)': 2000.0,
  'Stand Pipe Pressure(Psi)': 2500.0,
  'Flow In(GPM)': 800.0,
  'Temp - Out': 150.0,
  'Total Gas(PPM)': 100.0,
}

df = pd.DataFrame([row], columns=Inputs)
pf = model.predict(scaler.transform(df))[0]

MW = 10.0  # PPG
TVD = row['TVD(ft)']  # ft
ph = 0.052 * MW * TVD

status = 'SAFE' if ph > pf else 'CRITICAL'
print({'formation_pressure': pf, 'hydrostatic_pressure': ph, 'status': status})
```


## Appendix Q. Extended Safety Notes

- Treat CRITICAL as a trigger for immediate review by the drilling team
- Validate with secondary indicators (gas, flow, pit gain, SP pressure trends)
- Avoid reactive changes without considering fracture gradient and ECD impacts


## Appendix R. Dataset Health Checklist

- Coverage across depth intervals and lithologies
- Balanced representation of operational regimes
- Documented preprocessing steps (unit conversions, clamps)
- Versioned dataset snapshots for auditability


## Appendix S. UI/UX Design Rationale

- Clear separation of inputs and outputs to reduce cognitive load
- Color‑coded status and concise recommendations
- Tables and charts for traceability and quick situational awareness


## Appendix T. Extending to Multi‑Task Learning (Optional)

- Jointly predict `P_f` and auxiliary targets (e.g., equivalent circulating density) to improve robustness
- Multi‑objective loss may regularize predictions in sparse regimes


## Appendix U. Calibration and Uncertainty (Future Work)

- Quantify uncertainty via quantile regression or conformal predictors
- Calibrate predictions post‑hoc using isotonic regression if needed


## Appendix V. Integration Patterns

- Batch scoring: CSV in/out with the same `Inputs` schema
- Real‑time: Adapter service that pulls rig parameters and feeds the model, returning `P_f`, `P_h`, and status
- Alerting: Hook status changes to notification systems


## Appendix W. Template for Field Deployment SOP (Example)

1. Validate inputs and units against rig logs
2. Run the app and capture baseline predictions at the start of each shift
3. Monitor status changes and margins; log any CRITICAL events
4. Conduct cross‑checks and agree operational actions in morning meetings


## Appendix X. Extended Glossary (Additions)

- ECD: Equivalent Circulating Density
- LWD/MWD: Logging While Drilling / Measurement While Drilling
- Overbalanced/Underbalanced: Relative pressure states vs formation
- SHAP: SHapley Additive exPlanations


## Appendix Y. Notes on Units and Conversions

- 1 PPG ≈ 0.052 psi/ft conversion factor in hydrostatics
- Confirm temperature channels (°F/°C) consistency with training data
- Resistivity units should match model expectations (Ohm‑m or Ohm as encoded)


## Appendix Z. Maintenance Checklist

- Verify artifact compatibility after dependency upgrades
- Periodic re‑evaluation of feature importances and drift
- Refresh documentation and screenshots under `Graphs/` as models evolve