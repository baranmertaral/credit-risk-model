# Credit Risk Prediction Model

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.85-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)

End-to-end machine learning pipeline for credit default prediction using real-world financial data (150,000 records).

## Problem
Predict whether a borrower will experience serious financial distress (90+ days late on payment) within 2 years.

## Results
| Model | AUC | F1 |
|-------|-----|-----|
| Logistic Regression | 0.8343 | 0.2930 |
| Random Forest | 0.8328 | 0.2481 |
| **XGBoost** | **0.8499** | **0.3470** |

## Key Findings (SHAP)
- Number of times 90+ days late is the strongest predictor of default
- Younger age correlates with higher default risk
- Higher monthly income significantly reduces default probability

## Project Structure
```
credit-risk-model/
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_preprocessing.ipynb # Data cleaning & feature engineering
│   └── 03_modeling.ipynb     # Model training & SHAP explainability
├── app/
│   └── app.py                # Streamlit demo application
└── data/
    ├── model.pkl              # Trained XGBoost model
    └── scaler.pkl             # Feature scaler
```

## Tech Stack
- **Modeling:** scikit-learn, XGBoost, SHAP
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **App:** Streamlit

## Dataset
[Give Me Some Credit - Kaggle](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset)