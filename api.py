from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ===========================================
# Create FastAPI Application
# ===========================================

app = FastAPI(
    title="Customer Retention Prediction API",
    description="Predict customer retention probability using a trained Random Forest model.",
    version="1.0.0"
)

# ===========================================
# Load Trained Model
# ===========================================

model = joblib.load("models/retention_model.pkl")

# ===========================================
# Input Schema
# ===========================================

class CustomerData(BaseModel):
    recency_days: float
    total_spent: float
    avg_order_value: float

# ===========================================
# Home Endpoint
# ===========================================

@app.get("/")
def home():
    return {
        "message": "Customer Retention Prediction API",
        "status": "Running"
    }

# ===========================================
# Prediction Endpoint
# ===========================================

@app.post("/predict")
def predict(data: CustomerData):

    # Apply the same preprocessing used during training
    features = np.array([[
        data.recency_days,
        np.log1p(data.total_spent),
        data.avg_order_value
    ]])

    # Predict probability
    probability = model.predict_proba(features)[0][1]

    # Determine risk level and recommendation
    if probability >= 0.70:
        risk = "Low Risk"
        recommendation = (
            "Customer is highly likely to return. "
            "Maintain engagement through loyalty rewards and personalized offers."
        )

    elif probability >= 0.40:
        risk = "Medium Risk"
        recommendation = (
            "Customer has a moderate chance of returning. "
            "Consider sending promotional offers or follow-up messages."
        )

    else:
        risk = "High Risk"
        recommendation = (
            "Customer is unlikely to return. "
            "Launch a retention campaign with discounts or special incentives."
        )

    return {
        "retention_probability": round(float(probability), 4),
        "risk_level": risk,
        "recommendation": recommendation
    }