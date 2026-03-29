import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
# Load model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "churn_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn probability.")

# Basic Info
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

# Account Info
tenure = st.number_input("Tenure (months)", 0, 72)
monthly = st.number_input("Monthly Charges", 0.0)

# Services
phone = st.selectbox("Phone Service", ["No", "Yes"])
lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
stream_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Billing
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Automatically compute TotalCharges
total = tenure * monthly

# Create DataFrame
input_data = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "PhoneService":[phone],
    "MultipleLines":[lines],
    "InternetService":[internet],
    "OnlineSecurity":[security],
    "OnlineBackup":[backup],
    "DeviceProtection":[device],
    "TechSupport":[tech],
    "StreamingTV":[stream_tv],
    "StreamingMovies":[stream_movies],
    "Contract":[contract],
    "PaperlessBilling":[paperless],
    "PaymentMethod":[payment],
    "MonthlyCharges":[monthly],
    "TotalCharges":[total]
})

# Prediction button
if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠ Customer likely to churn\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer likely to stay\n\nProbability: {probability:.2f} chance customer will churn")
    st.progress(probability)
    # Addition of feature importance
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']
    
    feature_names = preprocessor.get_feature_names_out()
    importance = classifier.feature_importances_
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })
    
    importance_df = importance_df.sort_values(by="importance", ascending=False).head(10)
    importance_df["feature"] = importance_df["feature"].str.replace("num__", "")
    importance_df["feature"] = importance_df["feature"].str.replace("cat__", "")
    st.subheader("Top 10 important features")
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    ax.barh(
        importance_df['feature'],
        (importance_df['importance']/importance_df['importance'].sum())*100
    )
    
    ax.set_xlabel('Importance of features (%)')
    ax.set_title("Feature Importance")
    ax.invert_yaxis()
    
    st.pyplot(fig)

