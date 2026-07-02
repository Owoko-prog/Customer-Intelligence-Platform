# ==========================================================
# Customer Intelligence Platform
# Sales Forecast Model Training
# ==========================================================

# Data Manipulation
import pandas as pd
import numpy as np

# Prophet
from prophet import Prophet
from prophet.serialize import model_to_json

# Paths
from pathlib import Path

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# ==========================================================
# Create Required Directories
# ==========================================================

Path("models").mkdir(exist_ok=True)

Path("data").mkdir(exist_ok=True)
# ==========================================================
# Load Datasets
# ==========================================================

def load_data():

    print("Loading datasets...")

    orders = pd.read_csv(
        "data/olist_orders_dataset.csv"
    )

    payments = pd.read_csv(
        "data/olist_order_payments_dataset.csv"
    )

    print("Datasets loaded successfully.")

    return orders, payments


# ==========================================================
# Prepare Daily Sales Dataset
# ==========================================================

def prepare_sales_data(orders, payments):

    print("Preparing sales data...")

    # Keep delivered orders
    orders = orders[
        orders["order_status"] == "delivered"
    ]

    # Merge orders and payments
    sales = pd.merge(
        orders,
        payments,
        on="order_id",
        how="inner"
    )

    # Convert purchase timestamp
    sales["order_purchase_timestamp"] = pd.to_datetime(
        sales["order_purchase_timestamp"]
    )

    # Keep only required columns
    sales = sales[
        [
            "order_purchase_timestamp",
            "payment_value"
        ]
    ]

    # Aggregate daily sales
    daily_sales = (
        sales
        .groupby(
            sales["order_purchase_timestamp"].dt.date
        )["payment_value"]
        .sum()
        .reset_index()
    )

    daily_sales.columns = [
        "date",
        "sales"
    ]

    # Convert to datetime
    daily_sales["date"] = pd.to_datetime(
        daily_sales["date"]
    )

    # Fill missing dates
    daily_sales = daily_sales.set_index("date")

    daily_sales = daily_sales.asfreq("D")

    daily_sales["sales"] = (
        daily_sales["sales"]
        .fillna(0)
    )

    daily_sales = daily_sales.reset_index()

    # Remove incomplete end of dataset
    daily_sales = daily_sales[
        daily_sales["date"] < "2018-08-20"
    ]

    # Save processed dataset
    daily_sales.to_csv(
        "data/daily_sales.csv",
        index=False
    )

    print("Daily sales dataset created.")

    return daily_sales


# ==========================================================
# Test the Functions
# ==========================================================

orders, payments = load_data()

daily_sales = prepare_sales_data(
    orders,
    payments
)

print(daily_sales.head())

print("Daily Sales Shape:", daily_sales.shape)

# ==========================================================
# Train Prophet Model
# ==========================================================

def train_model(daily_sales):

    print("Training Prophet model...")

    # Prepare data for Prophet
    forecast_data = daily_sales.copy()

    forecast_data.columns = [
        "ds",
        "y"
    ]

    # Create model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    # Train model
    model.fit(forecast_data)

    print("Model trained successfully.")

    return model

# ==========================================================
# Save Model
# ==========================================================

def save_model(model):

    print("Saving model...")

    with open(
        "models/sales_forecast_model.json",
        "w"
    ) as file:

        file.write(
            model_to_json(model)
        )

    print("Model saved successfully.")

# ==========================================================
# Main Function
# ==========================================================

def main():

    orders, payments = load_data()

    daily_sales = prepare_sales_data(
        orders,
        payments
    )

    model = train_model(
        daily_sales
    )

    save_model(model)

    print("\n===================================")
    print("Sales Forecast Model Ready")
    print("===================================")
    print("Dataset : data/daily_sales.csv")
    print("Model   : models/sales_forecast_model.json")
    print("===================================")


if __name__ == "__main__":

    main()