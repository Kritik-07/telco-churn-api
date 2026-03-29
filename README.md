# Telco Customer Churn Prediction System

## Live API

- **FastAPI Backend:** https://telco-churn-api-iv8s.onrender.com  
- **API Docs:** https://telco-churn-api-iv8s.onrender.com/docs
---

An end-to-end Machine Learning system to predict customer churn and provide **human-readable explanations** using SHAP. The system includes a **FastAPI backend**, **Streamlit frontend**, and a **production-ready ML pipeline**.

---

## Problem Statement

Customer churn is a major challenge in the telecom industry. Retaining existing customers is more cost-effective than acquiring new ones.

This project aims to:

* Predict whether a customer will churn
* Identify key factors influencing churn
* Provide actionable insights for retention strategies

---

## Dataset Overview

This project uses the IBM Telco Customer Churn dataset 

### Dataset Details

* Total Records: **7043 customers**
* Features: **21 columns**
* Target Variable: **Churn**

  * No: 5174
  * Yes: 1869

---

### Feature Categories

* **Demographics**: gender, SeniorCitizen, Partner, Dependents
* **Services**: InternetService, OnlineSecurity, TechSupport, etc.
* **Account Info**: tenure, Contract, PaymentMethod
* **Billing**: MonthlyCharges, TotalCharges

---

## Key Insights from EDA

### Churn Behavior

* Customers with **short tenure churn significantly more**
* **Month-to-month contracts** have the highest churn rate
* **Higher monthly charges → higher churn probability**

---

### High Impact Features

* Contract type (**Month-to-month → highest churn**)
* Internet Service (**Fiber optic users churn more**)
* Tech Support (**No support → much higher churn**)
* Online Security (**Absence increases churn**)
* Payment Method (**Electronic check users churn more**)

---

### Weak Features

* Gender
* PhoneService
* MultipleLines

---

### Important Business Insight

> The highest churn risk customer typically has:

* Month-to-month contract
* Fiber optic internet
* No technical support
* Low tenure
* High monthly charges

---

## Machine Learning Pipeline

* **Preprocessing**

  * OneHotEncoding (categorical features)
  * StandardScaler (numerical features)

* **Models Used**

  * Logistic Regression
  * Random Forest
  * Gradient Boosting
  * XGBoost
  * LightGBM (final model)

* **Final Model**: LightGBM Classifier

---

##  Model Performance

* Cross-validation (5-fold)
* Accuracy: **~80.3%**
* ROC-AUC: **~0.845**

Indicates strong ability to distinguish churn vs non-churn

---

## Data Cleaning Insight

The `TotalCharges` column was initially of type `object` due to blank values.

### Fix applied:

```python
pd.to_numeric(df["TotalCharges"], errors="coerce")
```

Converts invalid values to NaN for proper handling

---

## Explainable AI (SHAP)

The system uses SHAP to:

* Identify top contributing features
* Explain predictions in human-readable format
* Improve trust and interpretability

### Example Output

```json
## 📌 Sample Response

```json
{
  "prediction": 0,
  "label": "No Churn",
  "probability": 0.198,
  "explanation": [
    "Contract is increasing churn risk by 0.24%",
    "Tenure is reducing churn risk 0.23%"
  ]
}
```

---

## System Architecture

```text
User (Streamlit UI)
        ↓
FastAPI Backend (src/)
        ↓
ML Pipeline (Preprocessing + Model)
        ↓
Prediction + SHAP Explanation
```

---

## Project Structure

```text
tel-churn-project/
│
├── app/                  # Streamlit frontend
├── data/                 # Raw & processed data
├── models/               # Saved ML model
├── notebooks/            # Training & EDA
├── src/                  # FastAPI backend
├── requirements.txt
├── README.md
```

---

##  Running Locally

### 1 Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2️ Run FastAPI

```bash
python -m uvicorn src.main:app --reload
```

Open: http://127.0.0.1:8000/docs

---

### 3️ Run Streamlit

```bash
streamlit run app/streamlit_app.py
```


---
## API Endpoint

### POST `/predict`

#### Input

```json
{
  "gender": "Male",
  "tenure": 12,
  "MonthlyCharges": 70.5,
  ...
}
```

#### Output

```json
{
  "churn": false,
  "confidence": 0.19,
  "top_reasons": [...]
}
```

---

## Key Learnings

* Handling real-world data issues (e.g., missing values in numeric columns)
* Building end-to-end ML pipelines
* Deploying ML models using FastAPI
* Implementing Explainable AI (SHAP)
* Designing modular and scalable ML systems

---

## Highlights

- Built and deployed an end-to-end ML system
- Integrated explainable AI (SHAP)
- Designed clean API with structured output

## Future Improvements

* Docker containerization
* Cloud deployment (AWS/GCP)
* Real-time monitoring
* Model retraining pipeline
* Advanced feature engineering

---

## Author

**Kritik Munot**

* GitHub: https://github.com/Kritik-07
* LinkedIn: https://www.linkedin.com/in/kritik-munot/

---

## If you found this useful

Give this project a ⭐ on GitHub!
