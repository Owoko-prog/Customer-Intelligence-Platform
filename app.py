import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Customer Intelligence Platform", page_icon="📊", layout="wide")
# ==========================================
# API Endpoints
# ==========================================

BASE_URL = "https://customer-intelligence-platform-0sh3.onrender.com"

DASHBOARD_URL = f"{BASE_URL}/dashboard"

SEGMENTS_URL = f"{BASE_URL}/segments"

MODEL_INFO_URL = f"{BASE_URL}/model-info"

PREDICT_URL = f"{BASE_URL}/predict-retention"

# ==========================================
# Helper Functions
# ==========================================

def fetch_api_data(url):

    try:

        response = requests.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:

        st.error(f"API Error: {e}")

        return None
    
# Temporary (Version 2.0)
customer_features=pd.read_csv("data/customer_features.csv")
customer_features["retained"]=(customer_features["total_orders"]>1).astype(int)

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Dashboard",
        "👥 Customer Center",
        "🧠 AI Insights",
        "📋 Platform Overview"
    ]
)

if page == "🏠 Dashboard":

    st.title("📊 Executive Dashboard")

    dashboard = fetch_api_data(DASHBOARD_URL)

    segments = fetch_api_data(SEGMENTS_URL)

    if dashboard:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "👥 Customers",
            dashboard["customers"]
        )

        c2.metric(
            "🔄 Retention Rate",
            f"{dashboard['retention_rate']:.2f}%"
        )

        c3.metric(
            "💰 Average Spend",
            f"${dashboard['average_spend']:.2f}"
        )

        c4.metric(
            "🛒 Average Orders",
            dashboard["average_orders"]
        )

    st.divider()

    if segments:

        seg = pd.DataFrame(
            segments.items(),
            columns=["Segment", "Customers"]
        )

        fig = px.pie(
            seg,
            names="Segment",
            values="Customers",
            hole=0.45,
            title="Customer Segments"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        "Customer analytics are now being served directly from the FastAPI backend."
    )

elif page == "👥 Customer Center":

    st.title("👥 Customer Center")

    st.caption(
        "View customer information and predict the likelihood of customer retention."
    )

    # Temporary (Version 2.0)
    # Customer search will move to the API in Version 2.2
    cid = st.selectbox(
        "Select Customer",
        customer_features["customer_unique_id"]
    )

    selected = customer_features[
        customer_features["customer_unique_id"] == cid
    ].iloc[0]

    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🛒 Total Orders",
            int(selected["total_orders"])
        )

        st.metric(
            "💰 Total Spend",
            f"${selected['total_spent']:.2f}"
        )

        st.metric(
            "🏷 Customer Segment",
            selected["segment"]
        )

    with col2:

        st.metric(
            "💵 Average Order Value",
            f"${selected['avg_order_value']:.2f}"
        )

        st.metric(
            "📅 Days Since Last Purchase",
            int(selected["recency_days"])
        )

    st.divider()

    if st.button("🚀 Predict Retention"):

        payload = {

            "recency_days": float(selected["recency_days"]),

            "total_spent": float(selected["total_spent"]),

            "avg_order_value": float(selected["avg_order_value"])

        }

        with st.spinner("Predicting customer retention..."):

            response = requests.post(
                PREDICT_URL,
                json=payload,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            st.subheader("Prediction Results")

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Retention Probability",
                    f"{result['retention_probability']:.1%}"
                )

            with c2:

                st.metric(
                    "Confidence",
                    result["confidence"]
                )

            st.progress(
                result["retention_probability"]
            )

            st.metric(
                "Risk Level",
                result["risk_level"]
            )

            st.success(
                result["recommendation"]
            )

        else:

            st.error(
                "Prediction service is currently unavailable."
            )

elif page == "🧠 AI Insights":

    st.title("🧠 AI Insights")

    st.caption(
        "Machine Learning model information and feature importance."
    )

    model = fetch_api_data(MODEL_INFO_URL)

    if model:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Algorithm",
                model["algorithm"]
            )

        with col2:

            st.metric(
                "Prediction Target",
                model["target"]
            )

        st.subheader("Model Features")

        for feature in model["features"]:

            st.write(f"✅ {feature}")

    st.divider()

    st.subheader("Feature Importance")

    st.image(
        "models/feature_importance.png",
        use_container_width=True
    )

    feature_importance = pd.read_csv(
        "models/feature_importance.csv"
    )

    st.dataframe(
        feature_importance,
        use_container_width=True
    )

    st.info(
        "Feature importance shows which variables have the greatest influence on customer retention predictions."
    )

elif page == "📋 Platform Overview":

    st.title("📋 Platform Overview")

    st.markdown("""
# Customer Intelligence Platform

### Turning Customer Data into Business Intelligence

The Customer Intelligence Platform is an end-to-end machine learning application that helps businesses understand customer behaviour, identify valuable customer segments, and predict customer retention using Artificial Intelligence.

---

## 🚀 Platform Capabilities

### 📊 Executive Dashboard

Monitor key customer metrics including:

- Customer Count
- Retention Rate
- Average Spend
- Customer Segments

---

### 👥 Customer Center

Explore customer profiles and predict customer retention using the trained Machine Learning model.

---

### 🧠 AI Insights

Understand how the Random Forest model makes predictions through feature importance and model metadata.

---

### 🌐 REST API

A cloud-hosted FastAPI backend deployed on Render provides prediction and analytics services for the application.

---

## 🛠 Technology Stack

| Component | Technology |
|------------|------------|
| Frontend | Streamlit |
| Backend | FastAPI |
| Machine Learning | Scikit-learn |
| Classification | Random Forest |
| Clustering | K-Means |
| Visualization | Plotly |
| Data Processing | Pandas & NumPy |
| Deployment | Render |

---

## 📂 Dataset

Brazilian Olist E-Commerce Dataset

- 96,000+ Customers
- 100,000+ Orders
- Customer Purchase History
- Transaction Behaviour
- Spending Patterns

---

## 🎯 Business Value

The platform enables businesses to:

- Improve customer retention
- Identify high-value customers
- Support marketing campaigns
- Understand customer behaviour
- Make data-driven decisions

---

## 🚀 Future Roadmap

✅ Sales Forecasting

✅ Customer Lifetime Value Prediction

✅ PostgreSQL Database

✅ Authentication & User Accounts

✅ CRM Integration

✅ Executive Reporting Dashboard

---

## 👨‍💻 Developer

**Shelton Owoko**

Machine Learning • Data Science • Business Intelligence
""")

    st.success(
        "Customer Intelligence Platform Version 2.0"
    )
