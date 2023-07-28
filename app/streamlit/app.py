import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000/predict"

st.title("Churn Prediction Web App")

st.write("Enter the following information:")

# Group 1: Number Vmail and International
st.subheader("Number Vmail and International")
number_vmail_messages = st.number_input("Number of Voice Mail Messages", min_value=0, step=1)
international_plan = st.selectbox("International Plan", ["no", "yes"])
voice_mail_plan = st.selectbox("Voice Mail Plan", ["no", "yes"])

# Group 2: Total Charges
st.subheader("Total Charges")
total_day_charge = st.number_input("Total Day Charge", min_value=0.0, step=0.01)
total_eve_charge = st.number_input("Total Evening Charge", min_value=0.0, step=0.01)
total_night_charge = st.number_input("Total Night Charge", min_value=0.0, step=0.01)
total_intl_charge = st.number_input("Total International Charge", min_value=0.0, step=0.01)

# Group 3: Total Calls
st.subheader("Total Calls")
total_day_calls = st.number_input("Total Day Calls", min_value=0, step=1)
total_eve_calls = st.number_input("Total Evening Calls", min_value=0, step=1)
total_night_calls = st.number_input("Total Night Calls", min_value=0, step=1)
total_intl_calls = st.number_input("Total International Calls", min_value=0, step=1)
number_customer_service_calls = st.number_input("Number of Customer Service Calls", min_value=0, step=1)

# Group 4: Total Minutes
st.subheader("Total Minutes")
total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, step=0.1)
total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, step=0.1)
total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, step=0.1)
total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, step=0.1)

if st.button("Predict"):
    data = {
        "number_vmail_messages": number_vmail_messages,
        "international_plan": international_plan,
        "voice_mail_plan": voice_mail_plan,
        "total_day_charge": total_day_charge,
        "total_eve_charge": total_eve_charge,
        "total_night_charge": total_night_charge,
        "total_intl_charge": total_intl_charge,
        "total_day_calls": total_day_calls,
        "total_eve_calls": total_eve_calls,
        "total_night_calls": total_night_calls,
        "total_intl_calls": total_intl_calls,
        "number_customer_service_calls": number_customer_service_calls,
        "total_day_minutes": total_day_minutes,
        "total_eve_minutes": total_eve_minutes,
        "total_night_minutes": total_night_minutes,
        "total_intl_minutes": total_intl_minutes,
    }

    response = requests.post(BACKEND_URL, json=data)
    prediction = response.json()["prediction"]
    st.subheader("Prediction Result:")
    st.text(f"Churn Prediction: {prediction}")
