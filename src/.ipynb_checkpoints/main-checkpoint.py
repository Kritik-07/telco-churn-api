from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import shap
from src.predict import predict_churn
from typing import Literal, List

app = FastAPI()

# ---------------- LOAD MODEL ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "churn_model.pkl")

model = joblib.load(MODEL_PATH)

# ---------------- FEATURES ---------------- #

col = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

# ---------------- INPUT MODEL ---------------- #

class Customer(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., example="Male")
    SeniorCitizen: int = Field(..., example=0, description="0 = No, 1 = Yes")
    Partner: Literal["Yes", "No"] = Field(..., example="Yes")
    Dependents: Literal["Yes", "No"] = Field(..., example="No")
    tenure: int = Field(..., example=12)
    PhoneService: Literal["Yes", "No"] = Field(..., example="Yes")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., example="No")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., example="Fiber optic")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., example="Yes")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., example="Month-to-month")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., example="Yes")
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.5)
    TotalCharges: float = Field(..., example=845.5)

# ---------------- RESPONSE MODEL ---------------- #

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    explanation: List[str]

# ---------------- SHAP SETUP ---------------- #

preprocessor = model.named_steps['preprocessor']
lgbm_model = model.named_steps['classifier']
explainer = shap.TreeExplainer(lgbm_model)
feature_names = preprocessor.get_feature_names_out()

# ---------------- EXPLANATION FUNCTION ---------------- #

def human_explain(feature_name, impact):
    feature_name = feature_name.replace("cat__", "").replace("num__", "")

    if "_" in feature_name:
        name = feature_name.split("_")[0]
    else:
        name = feature_name

    if impact > 0:
        return f"{name} is increasing churn risk"
    else:
        return f"{name} is reducing churn risk"

# ---------------- ROOT ENDPOINT ---------------- #

@app.get("/")
def home():
    return {
        "message": "Telco Churn Prediction API",
        "docs": "/docs"
    }

# ---------------- PREDICT ENDPOINT ---------------- #

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "prediction": 0,
                        "label": "No Churn",
                        "probability": 0.198,
                        "explanation": [
                            "Contract is increasing churn risk",
                            "Tenure is reducing churn risk"
                        ]
                    }
                }
            }
        }
    }
)
def predict(customer: Customer = Body(...)):
    
    # Get prediction
    df, prediction, pred_proba = predict_churn(customer.dict())

    # Transform input
    processed_data = preprocessor.transform(df)

    # SHAP values
    shap_result = explainer(processed_data)
    shap_values = shap_result.values[0]

    # Top 5 important features
    top_indices = abs(shap_values).argsort()[-5:][::-1]

    explanation = [
        human_explain(feature_names[i], shap_values[i]) for i in top_indices
    ]

    return {
        "prediction": int(prediction),
        "label": "Churn" if prediction == 1 else "No Churn",
        "probability": round(pred_proba, 3),
        "explanation": explanation
    }


#conda install -c conda-forge shap

# shap_values
# shap_result.values =
# [
#   [0.2, -0.5, 0.1, ...]
# ]

# 👉 Each number = contribution of each processed feature

# ============================================================
# 🔥 PIPELINE + SHAP EXPLANATION FLOW (IMPORTANT)
# ============================================================

# Our saved model is NOT just a model — it's a Pipeline:
#
#   Pipeline
#    ├── preprocessor  → handles data cleaning & conversion
#    │       - OneHotEncoding (strings → numbers)
#    │       - Scaling (normalize numeric values)
#    │
#    └── classifier    → LGBM model (makes predictions)
#
# ------------------------------------------------------------
# ❌ Problem:
# SHAP cannot directly explain a Pipeline
#
#   shap.Explainer(model)  → ❌ ERROR
#
# ------------------------------------------------------------
# ✅ Solution:
# We SPLIT the pipeline into 2 parts:
#
#   1. Preprocessor  → prepares data (Converts:categories → numbers, scales numeric values)
#   2. Model         → makes prediction (SHAP works here)
#
# ------------------------------------------------------------
# STEP 1: Extract components
#
#   preprocessor = model.named_steps['preprocessor']
#   lgbm_model   = model.named_steps['classifier']
#
# ------------------------------------------------------------
# STEP 2: Create SHAP explainer ONLY for model
#
#   explainer = shap.TreeExplainer(lgbm_model)
#
#   (TreeExplainer is used because LGBM is a tree-based model)
#
# ------------------------------------------------------------
# STEP 3: During prediction
#
#   Raw Input (API JSON)
#           ↓
#   Convert to DataFrame
#           ↓
#   preprocessor.transform(data)
#           ↓
#   processed_data (NUMERIC ONLY)
#           ↓
#   model.predict(processed_data)
#           ↓
#   explainer(processed_data)
#           ↓
#   SHAP values (feature contributions)
#
# ------------------------------------------------------------
# ⚠️ IMPORTANT NOTE:
# SHAP now explains PROCESSED FEATURES (after encoding),
# not original features directly.
#
# That’s why we see:
#   feature_index = 32 (not actual feature name)
#
# ------------------------------------------------------------
# 🧠 Key Idea:
#   Preprocessor = "data converter"
#   Model        = "decision maker"
#   SHAP         = "decision explainer"
#
# ============================================================