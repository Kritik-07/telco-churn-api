
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "churn_model.pkl")

model = joblib.load(MODEL_PATH)


columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

def predict_churn(data):
    df = pd.DataFrame([data], columns=columns)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if df["TotalCharges"].isna().any():
        raise ValueError("TotalCharges is invalid")

    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return df, prediction, proba