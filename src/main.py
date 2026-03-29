from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import shap
from src.predict import predict_churn
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "churn_model.pkl")

model = joblib.load(MODEL_PATH)



col = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

class Customer(BaseModel):
    gender: str
    SeniorCitizen:int
    Partner: str
    Dependents: str
    tenure:int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges:float
    TotalCharges:float

preprocessor = model.named_steps['preprocessor']
lgbm_model = model.named_steps['classifier']
explainer  = shap.TreeExplainer(lgbm_model)
features_names = preprocessor.get_feature_names_out()

def human_explain(feature_name, impact):
    feature_name = feature_name.replace("cat__","").replace("num__","")
    if "_" in feature_name:
        name = feature_name.split("_")[0]
    else:
        name = feature_name
    if impact > 0:
        text = f"{name} increase the churn risk by {round(impact,3)}"
    else:
        text = f"{name} decrease the churn risk by {round(impact,3)}"
    return text
    
@app.post("/predict")
def predict(customer: Customer):
    df, prediction, pred_proba = predict_churn(customer.dict())
    data = pd.DataFrame([customer.dict()],columns = col)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors = 'coerce')


    #transform data using pipeline (Converts your raw input into model-ready numeric format)
    processed_data = preprocessor.transform(data)

    #get shap values
    shap_result = explainer(processed_data)

    #take first row
    shap_values = shap_result.values[0]

    top_indices = abs(shap_values).argsort()[-5:][::-1]

    explaination = [
        human_explain(features_names[i],shap_values[i]) for i in top_indices
        ]
    return {'prediction': int(prediction),
           'pred_proba': f"{round(pred_proba,3)}% it will be churn",
        'Explaination': explaination
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