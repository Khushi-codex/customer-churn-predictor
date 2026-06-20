import streamlit as st
import joblib
import numpy as np
import os

model = joblib.load(os.path.join(os.path.dirname(__file__), "model/churn_model.pkl"))

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.caption("Enter customer details to predict churn risk")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)

def encode(val, mapping):
    return mapping[val]

gender_map = {"Female": 0, "Male": 1}
yn = {"No": 0, "Yes": 1}
senior_map = {"No": 0, "Yes": 1}
multi_map = {"No": 0, "No phone service": 1, "Yes": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
nis_map = {"No": 0, "No internet service": 1, "Yes": 2}
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {"Bank transfer (automatic)": 0, "Credit card (automatic)": 1, "Electronic check": 2, "Mailed check": 3}

if st.button("Predict Churn", type="primary"):
    features = np.array([[
        encode(gender, gender_map),
        senior_map[senior],
        yn[partner],
        yn[dependents],
        tenure,
        yn[phone_service],
        encode(multiple_lines, multi_map),
        encode(internet, internet_map),
        encode(online_security, nis_map),
        encode(online_backup, nis_map),
        encode(device_protection, nis_map),
        encode(tech_support, nis_map),
        encode(streaming_tv, nis_map),
        encode(streaming_movies, nis_map),
        encode(contract, contract_map),
        yn[paperless],
        encode(payment, payment_map),
        monthly,
        total
    ]])

    prob = model.predict_proba(features)[0][1]
    pred = prob > 0.5

    st.divider()
    if pred:
        st.error(f"⚠️ High Churn Risk — {prob:.0%} probability")
    else:
        st.success(f"✅ Low Churn Risk — {prob:.0%} probability")

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Probability", f"{prob:.0%}")
    c2.metric("Risk Level", "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low")
    c3.metric("Prediction", "Will Churn" if pred else "Will Stay")