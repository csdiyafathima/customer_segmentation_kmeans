import streamlit as st
import joblib
import pandas as pd
kmeans = joblib.load("customer_segmentation_model.pkl")

st.title("Customer Cluster Prediction")

annual_income = st.number_input(
    "Enter Annual Income (INR)",
    min_value=0.0,
    value=50000.0
)

monthly_spending = st.number_input(
    "Enter Monthly Spending (INR)",
    min_value=0.0,
    value=5000.0
)

visits = st.number_input(
    "Enter Visits Per Month",
    min_value=0.0,
    value=8.0
)

if st.button("Predict Cluster"):

    new_customer = pd.DataFrame({
        "Annual_Income_INR": [annual_income],
        "Monthly_Spending_INR": [monthly_spending],
        "Visits_Per_Month": [visits]
    })

    cluster = kmeans.predict(new_customer)

    st.success(f"Customer belongs to Cluster {cluster[0]}")
