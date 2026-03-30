import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

@st.cache_resource
def load_model():
    model = pickle.load(open('data/model.pkl', 'rb'))
    scaler = pickle.load(open('data/scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_model()

st.title("Kredi Riski Tahmin Modeli")
st.markdown("XGBoost tabanlı, SHAP açıklamalı kredi riski tahmin sistemi · AUC: 0.85")

st.sidebar.header("Müşteri Bilgileri")

age = st.sidebar.slider("Yaş", 18, 100, 45)
income = st.sidebar.number_input("Aylık Gelir ($)", 0, 50000, 5000, step=500)
debt_ratio = st.sidebar.slider("Borç Oranı", 0.0, 5.0, 0.3)
revolving = st.sidebar.slider("Kredi Kullanım Oranı", 0.0, 1.0, 0.3)
late_30_59 = st.sidebar.slider("30-59 Gün Gecikme", 0, 10, 0)
late_60_89 = st.sidebar.slider("60-89 Gün Gecikme", 0, 10, 0)
late_90 = st.sidebar.slider("90+ Gün Gecikme", 0, 10, 0)
open_credits = st.sidebar.slider("Açık Kredi Sayısı", 0, 30, 5)
real_estate = st.sidebar.slider("Gayrimenkul Kredisi", 0, 10, 1)
dependents = st.sidebar.slider("Bakmakla Yükümlü Kişi", 0, 10, 0)

input_data = pd.DataFrame([{
    'RevolvingUtilizationOfUnsecuredLines': revolving,
    'age': age,
    'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
    'DebtRatio': debt_ratio,
    'MonthlyIncome': income,
    'NumberOfOpenCreditLinesAndLoans': open_credits,
    'NumberOfTimes90DaysLate': late_90,
    'NumberRealEstateLoansOrLines': real_estate,
    'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
    'NumberOfDependents': dependents
}])

input_scaled = scaler.transform(input_data)
proba = model.predict_proba(input_scaled)[0][1]

st.header("Tahmin Sonucu")
col1, col2 = st.columns(2)

with col1:
    st.metric("Default Olasılığı", f"{proba:.1%}")

with col2:
    if proba < 0.1:
        st.success("Düşük Risk")
    elif proba < 0.3:
        st.warning("Orta Risk")
    else:
        st.error("Yüksek Risk")

st.progress(float(proba))

st.header("Feature Importance")
feature_names = list(input_data.columns)
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)
st.bar_chart(importance_df.set_index('Feature'))

st.header("Model Açıklaması (SHAP)")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_data.iloc[0],
        feature_names=input_data.columns.tolist()
    ), show=False)
    st.pyplot(fig)
    plt.close()
except Exception as e:
    st.info("SHAP grafiği yüklenemedi.")