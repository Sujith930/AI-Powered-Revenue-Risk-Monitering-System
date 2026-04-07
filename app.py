import streamlit as st
import joblib
import numpy as np

st.title("AI Revenue Risk Monitoring System")

# Load model
model = joblib.load("revenue_model.pkl")

st.markdown("Enter transaction details below:")

discount = st.slider("Discount", 0.0, 1.0, 0.1)
quantity = st.number_input("Quantity", min_value=1, value=5)
shipping_cost = st.number_input("Shipping Cost", min_value=0.0, value=50.0)
month = st.slider("Month", 1, 12, 6)
year = st.selectbox("Year", [2012, 2013, 2014, 2015])

if st.button("Predict Revenue Risk"):
    input_data = np.array([[discount, quantity, shipping_cost, month, year]])
    prediction = model.predict(input_data)[0]

    threshold = 1128

    if prediction < threshold:
        risk = "High Risk"
    else:
        risk = "Normal"

    st.subheader("AI Prediction Output")
    st.success(f"Predicted Sales: {prediction:.2f}")
    st.write(f"Risk Level: {risk}")

