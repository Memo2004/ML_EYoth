import streamlit as st
import numpy as np
import joblib
import os
from pathlib import Path

from prophet.plot import plot_plotly

# --- Sidebar Navigation ---
st.sidebar.title("🚀 Model Selection")

page = st.sidebar.radio("Navigate to:", ["🏠 Home", "🔍 Customer Segmentation", "📈 Sales Forecasting"])

# --- Home Page ---
if page == "🏠 Home":
    st.title("📊 Multi-Model Prediction App")
    st.markdown("""
    Welcome to the Multi-Model Prediction App!  
    This app helps you with:
    
    - 🔍 Predicting customer segments using the KMeans algorithm.  
    - 📈 Forecasting future sales using the Prophet model.
    
    Use the sidebar to select which model you want to explore.
    """)
   


# --- Customer Segmentation ---
elif page == "🔍 Customer Segmentation":
    
    st.title("🔍 Predict Customer Segment")
    project_root = Path(__file__).parent  
    model1_path = project_root / 'kmeans_model.pkl'
    # Load the trained model
    with open(model1_path, 'rb') as f:
        model = joblib.load(f)
 
    scaler_path = project_root / 'scaler.pkl'
    # Load the trained model
    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)

    #model = joblib.load(r"C:\Users\hp\Desktop\ML_EYoth\kmeans_model.pkl")
    #scaler = joblib.load(r"C:\Users\hp\Desktop\ML_EYoth\scaler.pkl")

    cluster_names = {
        0: "Loyal Premium Customers",
        1: "At-Risk / Low-Value Users",
        2: "High-Spenders with High Ratings",
        3: "Frequent but Low-Spending Buyers"
    }

    st.markdown("Enter customer attributes below:")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            orders = st.number_input("📦 Number of Orders", min_value=0)
            spend = st.number_input("💰 Total Spend", min_value=0.0)

        with col2:
            rating = st.slider("⭐ Average Rating", 0.0, 5.0, step=0.1)
            payment_method = st.selectbox("💳 Preferred Payment Method", ["Credit Card", "PayPal", "Cash", "Debit Card"])

    payment_dict = {'Cash': 0, 'Credit Card': 1, 'Debit Card': 2, 'PayPal': 3}
    payment_encoded = payment_dict.get(payment_method, 0)

    if st.button("🔍 Predict Segment"):
        user_input = np.array([[orders, spend, rating, payment_encoded]])
        scaled_input = scaler.transform(user_input)
        cluster = model.predict(scaled_input)[0]
        segment = cluster_names.get(cluster, "Unknown Segment")
        st.success(f"🎯 Predicted Customer Segment: **{segment}**")

# --- Sales Forecasting ---
elif page == "📈 Sales Forecasting":
    st.title("📈 Daily Sales Forecast")
    
    project_root = Path(__file__).parent  
    model2_path = project_root / 'sales_forecast_model.pkl'
    # Load the trained model
    with open(model2_path, 'rb') as f:
        model = joblib.load(f)

    n_days = st.slider("Select the number of days to forecast:", 7, 90, 30)

    future = model.make_future_dataframe(periods=n_days, freq='D')
    forecast = model.predict(future)

    st.subheader("📊 Forecast Chart")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig)

    st.subheader("📋 Forecast Data Table")
    st.dataframe(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days).rename(columns={
            'ds': 'Date',
            'yhat': 'Predicted Sales',
            'yhat_lower': 'Lower Estimate',
            'yhat_upper': 'Upper Estimate'
        })
    )
