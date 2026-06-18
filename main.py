from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn based on their account features.",
    version="1.0.0"
)

MODEL_PATH = "model/churn_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, le=72, description="Months as a customer (0–72)")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly bill amount in USD")
    TotalCharges: float = Field(..., ge=0, description="Total amount charged to date")
    Contract: int = Field(..., ge=0, le=2, description="0=Month-to-month, 1=One year, 2=Two year")
    InternetService: int = Field(..., ge=0, le=2, description="0=DSL, 1=Fiber optic, 2=No internet")
    PaymentMethod: int = Field(..., ge=0, le=3, description="0=Bank transfer, 1=Credit card, 2=Electronic check, 3=Mailed check")
    OnlineSecurity: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    TechSupport: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 65.5,
                "TotalCharges": 780.0,
                "Contract": 0,
                "InternetService": 1,
                "PaymentMethod": 2,
                "OnlineSecurity": 0,
                "TechSupport": 0
            }
        }


class PredictionResponse(BaseModel):
    churn_prediction: bool
    churn_probability: float
    risk_level: str


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is live",
        "docs": "/docs",
        "status": "model loaded" if model else "model not found — run the notebook first"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running notebooks/analysis.ipynb"
        )

    features = np.array([[
        customer.tenure,
        customer.MonthlyCharges,
        customer.TotalCharges,
        customer.Contract,
        customer.InternetService,
        customer.PaymentMethod,
        customer.OnlineSecurity,
        customer.TechSupport
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if probability > 0.7:
        risk = "High"
    elif probability > 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "churn_prediction": bool(prediction),
        "churn_probability": round(float(probability), 3),
        "risk_level": risk
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}