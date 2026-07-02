from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from prophet.serialize import model_from_json


# ===========================================
# Create FastAPI Application
# ===========================================

app = FastAPI(
    title="Customer Intelligence Platform API",
    description="Customer Intelligence Platform powered by Machine Learning.",
    version="2.1.0"
)

# ===========================================
# Load Trained Model
# ===========================================

model = joblib.load("models/retention_model.pkl")

# ===========================================
# Load Sales Forecast Model
# ===========================================

with open(
    "models/sales_forecast_model.json",
    "r"
) as file:

    sales_model = model_from_json(
        file.read()
    )

# ===========================================
# Load Customer Dataset
# ===========================================

customer_features = pd.read_csv("data/customer_features.csv")

# Recreate retained column
customer_features["retained"] = (
    customer_features["total_orders"] > 1
).astype(int)

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
        "platform": "Customer Intelligence Platform",
        "version": "2.1.0",
        "status": "Running"
    }

# ===========================================
# Health Endpoint
# ===========================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# ===========================================
# Dashboard Endpoint
# ===========================================

@app.get("/dashboard")
def dashboard():

    total_customers = len(customer_features)

    retained_customers = int(
        customer_features["retained"].sum()
    )

    retention_rate = round(
        customer_features["retained"].mean() * 100,
        2
    )

    average_spend = round(
        customer_features["total_spent"].mean(),
        2
    )

    average_orders = round(
        customer_features["total_orders"].mean(),
        2
    )

    return {

        "customers": total_customers,

        "retained_customers": retained_customers,

        "retention_rate": retention_rate,

        "average_spend": average_spend,

        "average_orders": average_orders

    }

# ===========================================
# Customer List Endpoint and Customer search Endpoint
# ===========================================

@app.get("/customers")
def customers(page: int = 1, limit: int = 100):

    start = (page - 1) * limit
    end = start + limit

    return customer_features.iloc[start:end][
        "customer_unique_id"
    ].tolist()

@app.get("/search-customers")
def search_customers(query: str):

    results = customer_features[
        customer_features["customer_unique_id"]
        .str.contains(query, case=False, na=False)
    ]

    return results[
        "customer_unique_id"
    ].head(20).tolist()

# ===========================================
# Customer Details Endpoint
# ===========================================

@app.get("/customer/{customer_id}")
def customer(customer_id: str):

    customer = customer_features[
        customer_features["customer_unique_id"] == customer_id
    ]

    if customer.empty:

        return {
            "error": "Customer not found"
        }

    customer = customer.iloc[0]

    return {

        "customer_id": customer["customer_unique_id"],

        "segment": customer["segment"],

        "total_orders": int(customer["total_orders"]),

        "total_spent": float(customer["total_spent"]),

        "avg_order_value": float(customer["avg_order_value"]),

        "recency_days": int(customer["recency_days"])

    }

# ===========================================
# Prediction Endpoint
# ===========================================

@app.post("/predict")
@app.post("/predict-retention")
def predict(data: CustomerData):

    # Apply preprocessing
    features = np.array([[
        data.recency_days,
        np.log1p(data.total_spent),
        data.avg_order_value
    ]])

    probability = model.predict_proba(features)[0][1]

    # Risk Level
    if probability >= 0.70:

        risk = "Low Risk"

        recommendation = (
            "Customer is highly likely to return. "
            "Maintain engagement through loyalty rewards "
            "and personalized offers."
        )

    elif probability >= 0.40:

        risk = "Medium Risk"

        recommendation = (
            "Customer has a moderate chance of returning. "
            "Consider sending promotional offers or "
            "follow-up messages."
        )

    else:

        risk = "High Risk"

        recommendation = (
            "Customer is unlikely to return. "
            "Launch a retention campaign with discounts "
            "or special incentives."
        )

    return {

        "retention_probability": round(
            float(probability),
            4
        ),

        "confidence": f"{probability*100:.2f}%",

        "risk_level": risk,

        "recommendation": recommendation

    }

# ===========================================
# Customer Segments Endpoint
# ===========================================

@app.get("/segments")
def segments():

    segment_counts = (
        customer_features["segment"]
        .value_counts()
        .to_dict()
    )

    return segment_counts

# ===========================================
# Model Information Endpoint
# ===========================================

@app.get("/model-info")
def model_info():

    return {

        "algorithm": "Random Forest Classifier",

        "dataset": "Brazilian Olist E-Commerce Dataset",

        "features": [

            "recency_days",

            "total_spent_log",

            "avg_order_value"

        ],

        "target": "retained"

    }

# ===========================================
# Sales Forecast Endpoint
# ===========================================

@app.get("/sales-forecast")
def sales_forecast():

    # Generate next 30 days
    future = sales_model.make_future_dataframe(
        periods=30
    )

    forecast = sales_model.predict(future)

    # Keep only future dates
    forecast = forecast.tail(30)

    forecast = forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ]

    forecast.columns = [
        "date",
        "predicted_sales",
        "lower_bound",
        "upper_bound"
    ]

    forecast["date"] = (
        forecast["date"]
        .astype(str)
    )

    return {
        "forecast": forecast.to_dict(
            orient="records"
        ),

        "total_predicted_sales": round(
            forecast["predicted_sales"].sum(),
            2
        ),

        "average_daily_sales": round(
            forecast["predicted_sales"].mean(),
            2
        ),

        "highest_day": round(
            forecast["predicted_sales"].max(),
            2
        ),

        "lowest_day": round(
            forecast["predicted_sales"].min(),
            2
        )
    }