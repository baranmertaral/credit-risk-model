# Credit Risk Prediction Model

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.85-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end machine learning pipeline that predicts the probability of a borrower experiencing serious financial distress within 2 years. Built on 150,000 real-world credit records, the model achieves **AUC 0.85** and includes full SHAP-based explainability.

🔴 **[Live Demo → credit-risk-model-from-baranmert.streamlit.app](https://credit-risk-model-from-baranmert.streamlit.app)**

---

## Problem Statement

Financial institutions need reliable tools to assess credit risk before extending loans. Misclassifying high-risk borrowers leads to significant losses, while being overly conservative limits revenue. This project builds a production-ready classification model to predict the probability of serious delinquency (90+ days past due) using behavioral and financial features.

---

## Dataset

- **Source:** [Give Me Some Credit – Kaggle](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)
- **Size:** 150,000 borrowers, 10 features
- **Target:** `SeriousDlqin2yrs` — binary (1 = default within 2 years)
- **Class imbalance:** 6.68% positive class (handled via `scale_pos_weight`)

---

## Project Structure
```
credit-risk-model/
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis & visualizations
│   ├── 02_preprocessing.ipynb      # Missing value imputation, outlier clipping, train/test split
│   └── 03_modeling.ipynb           # Model training, comparison, SHAP explainability
├── app/
│   └── app.py                      # Streamlit demo application
├── data/
│   ├── model.pkl                   # Trained XGBoost model
│   └── scaler.pkl                  # StandardScaler
├── environment.yml                 # Conda environment (Python 3.11)
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis
- Identified severe class imbalance (6.68% default rate)
- Found 19.8% missing values in `MonthlyIncome` and 2.6% in `NumberOfDependents`
- Detected extreme outliers in `DebtRatio` and `RevolvingUtilizationOfUnsecuredLines`

### 2. Preprocessing
- **Missing values:** Median imputation
- **Outlier clipping:** Age [18–100], DebtRatio [0–10], RevolvingUtilization [0–1], MonthlyIncome [0–50k]
- **Train/test split:** 80/20 with stratification to preserve class distribution
- **Scaling:** StandardScaler applied to all features

### 3. Model Training & Comparison

| Model | AUC | F1 Score |
|-------|-----|----------|
| Logistic Regression | 0.8343 | 0.2930 |
| Random Forest | 0.8328 | 0.2481 |
| **XGBoost** ✓ | **0.8499** | **0.3470** |

XGBoost was selected as the final model. Class imbalance was addressed using `scale_pos_weight = 14.0` (ratio of negative to positive samples).

### 4. Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to interpret model predictions at both global and individual levels.

**Key findings:**
- `NumberOfTimes90DaysLate` — strongest predictor; high values dramatically increase default probability
- `age` — older borrowers are significantly less likely to default
- `MonthlyIncome` — higher income reduces default risk
- `DebtRatio` — high debt burden increases risk

---

## Live Application

The Streamlit app allows users to:
- Input custom borrower profiles via interactive sliders
- Get real-time default probability predictions
- View SHAP waterfall plots explaining each individual prediction

🔴 **[Try it here](https://credit-risk-model-from-baranmert.streamlit.app)**

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Modeling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| App | Streamlit |
| Version Control | Git, GitHub |

---

## Author

**Baran Mert Aral** — Data Scientist  
[LinkedIn](https://linkedin.com/in/baranmertaral) · [GitHub](https://github.com/baranmertaral)