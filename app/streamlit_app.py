import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

#  LIVE API
API_URL = "https://telco-churn-api-iv8s.onrender.com/predict"

st.title("Customer Churn Prediction System")
st.write("Enter customer details to predict churn and understand reasons.")

# ---------------- INPUT ---------------- #

# Basic Info
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

# Account Info
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, value=70.5)

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

# AUTO CALCULATION 
total = tenure * monthly

st.info(f Estimated Total Charges: {round(total, 2)}")

# ---------------- PREDICTION ---------------- #

if st.button("Predict Churn"):

    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    try:
        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            result = response.json()

            prediction = result["prediction"]
            probability = result["pred_proba"]

            # ---------------- RESULT ---------------- #

            if prediction == 1:
                st.error(f" Customer likely to churn\n\nConfidence: {probability:.2f}")
            else:
                st.success(f" Customer likely to stay\n\nConfidence: {probability:.2f}")

            st.progress(probability)

            # ---------------- EXPLANATION ---------------- #

            st.subheader(" Top Reasons")

            for reason in result.get("explanation", []):
                st.write(f"• {reason}")

        else:
            st.error("API Error")

    except Exception as e:
        st.error(f"Error connecting to API: {e}")

# ---------------- STATIC FEATURE IMPORTANCE GRAPH ---------------- #

st.subheader("Global Feature Importance")

# Hardcoded (from model insights)
features = [
    "Contract", "Tenure", "MonthlyCharges", "TechSupport",
    "InternetService", "OnlineSecurity", "PaymentMethod",
    "PaperlessBilling", "DeviceProtection", "OnlineBackup"
]

importance = [20, 18, 15, 12, 10, 8, 6, 5, 3, 3]

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(features, importance)
ax.set_xlabel("Importance (%)")
ax.set_title("Top Features Affecting Churn")
ax.invert_yaxis()

st.pyplot(fig)