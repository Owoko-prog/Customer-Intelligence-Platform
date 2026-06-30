import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Customer Intelligence Platform", page_icon="📊", layout="wide")
API_URL="https://customer-intelligence-platform-0sh3.onrender.com/predict"

customer_features=pd.read_csv("data/customer_features.csv")
customer_features["retained"]=(customer_features["total_orders"]>1).astype(int)

st.sidebar.title("📊 Navigation")
page=st.sidebar.radio("Go To",["Dashboard","Customer Lookup","Model Insights","About"])

if page=="Dashboard":
    st.title("📊 Customer Intelligence Platform")
    retention_rate=customer_features["retained"].mean()*100
    c1,c2,c3,c4=st.columns(4)
    c1.metric("👥 Customers",len(customer_features))
    c2.metric("🔄 Retention",f"{retention_rate:.2f}%")
    c3.metric("💰 Avg Spend",f"${customer_features['total_spent'].mean():.2f}")
    c4.metric("🛒 Avg Orders",round(customer_features["total_orders"].mean(),2))
    st.divider()
    seg=customer_features["segment"].value_counts().reset_index()
    seg.columns=["Segment","Customers"]
    st.plotly_chart(px.pie(seg,names="Segment",values="Customers",hole=.45),use_container_width=True)
    st.subheader("🔥 High Risk Customers")
    risk=customer_features.sort_values("recency_days",ascending=False).head(10)
    st.dataframe(risk[["customer_unique_id","segment","total_spent","recency_days"]],use_container_width=True)

elif page=="Customer Lookup":
    st.title("🔍 Customer Center")
    cid=st.selectbox("Customer",customer_features["customer_unique_id"])
    selected=customer_features[customer_features["customer_unique_id"]==cid].iloc[0]
    a,b,c=st.columns(3)
    a.metric("Orders",int(selected["total_orders"]))
    b.metric("Spend",f"${selected['total_spent']:.2f}")
    c.metric("Segment",selected["segment"])
    d,e=st.columns(2)
    d.metric("Average Order",f"${selected['avg_order_value']:.2f}")
    e.metric("Recency (days)",int(selected["recency_days"]))
    if st.button("Predict Retention"):
        with st.spinner("Predicting..."):
            r=requests.post(API_URL,json={
                "recency_days":float(selected["recency_days"]),
                "total_spent":float(selected["total_spent"]),
                "avg_order_value":float(selected["avg_order_value"])
            },timeout=30)
        if r.status_code==200:
            res=r.json()
            st.metric("Retention Probability",f"{res['retention_probability']:.1%}")
            st.progress(res["retention_probability"])
            st.metric("Risk Level",res["risk_level"])
            st.success(res["recommendation"])
        else:
            st.error("Prediction API unavailable.")

elif page=="Model Insights":
    st.title("🧠 Model Insights")
    st.image("models/feature_importance.png",use_container_width=True)
    st.dataframe(pd.read_csv("models/feature_importance.csv"),use_container_width=True)

else:
    st.title("ℹ️ About")
    st.write("Customer Intelligence Platform powered by Streamlit, FastAPI and Machine Learning.")
