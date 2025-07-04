import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os
import gdown

# --- Load model, encoders, and scaler ---
model_path = Path('Models/best_random_forest_model.pkl')
encoders_path = Path('Models/label_encoders.pkl')
scaler_path = Path('Models/scaler.pkl')

if not model_path.exists():
    url = 'https://drive.google.com/uc?id=1CtweaxrA8UGT6smSgC53298h7Drqb4Du'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(model_path), quiet=False)

loaded_model = joblib.load(model_path)
encoders = joblib.load(encoders_path)
scaler_data = joblib.load(scaler_path)

def predict_customer_churn(user_input):
    customer_df = pd.DataFrame([user_input])
    for feature in encoders:
        customer_df[feature] = encoders[feature].transform(customer_df[feature])
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    customer_df[num_features] = scaler_data.transform(customer_df[num_features])
    prediction_result = loaded_model.predict(customer_df)[0]
    prediction_score = loaded_model.predict_proba(customer_df)[0][1]
    status = "Churn" if prediction_result == 1 else "No Churn"
    return status, prediction_score

# --- Streamlit UI ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="💡", layout="centered")
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {background-color: #4CAF50; color: white; font-weight: bold;}
    .st-bb {background-color: #e3fcec;}
    </style>
    """, unsafe_allow_html=True)

st.title("🌟 Customer Churn Prediction App")
st.markdown("""
Welcome! Enter customer details below to predict the likelihood of churn. 

:star2: **Our model helps you keep your customers happy!** :star2:
""")

with st.form("churn_form"):
    st.subheader("Enter Customer Information:")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=0.0)
    submit = st.form_submit_button("Predict Churn")

user_input = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

if 'submit' in locals() and submit:
    status, score = predict_customer_churn(user_input)
    if status == "No Churn":
        st.success(f"🎉 The customer is likely to stay! (Probability of churn: {score:.2%})")
    else:
        st.warning(f"⚠️ The customer may churn. (Probability of churn: {score:.2%})")
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#4CAF50; font-size:18px;'>Thank you for using our Churn Predictor! Keep delighting your customers! 😊</div>",
        unsafe_allow_html=True
    )
