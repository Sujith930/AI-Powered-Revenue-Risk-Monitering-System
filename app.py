import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from groq import Groq

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="AI Revenue System", layout="wide")

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
    AI Revenue Risk Monitoring System
    </h1>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("revenue_model.pkl")

# ---------------------------
# INPUT SECTION
# ---------------------------
st.sidebar.header("📥 Enter Transaction Details")

discount = st.sidebar.slider("Discount (%)", 0.0, 1.0, 0.1)
quantity = st.sidebar.number_input("Quantity", 1, 20, 5)
shipping_cost = st.sidebar.number_input("Shipping Cost", 0.0, 1000.0, 50.0)

# Business-friendly input
season = st.sidebar.selectbox(
    "Business Period",
    ["Regular", "Festive Season", "Off Season"]
)

# Convert season → month
if season == "Festive Season":
    month = 12
elif season == "Off Season":
    month = 6
else:
    month = 3

year = 2015  # hidden constant for model

# ---------------------------
# MAIN PREDICTION
# ---------------------------
prediction = None

if st.sidebar.button("🚀 Predict Revenue"):

    input_data = np.array([[discount, quantity, shipping_cost, month, year]])
    prediction = model.predict(input_data)[0]

    threshold = 1128

    if prediction < threshold:
        risk = "🔴 High Risk"
        insight = "High discount or low quantity is reducing expected revenue."
        recommendation = "Reduce discount or increase quantity."
    else:
        risk = "🟢 Normal"
        insight = "Transaction is expected to generate stable revenue."
        recommendation = "Maintain current pricing strategy."

    # ---------------------------
    # OUTPUT SECTION
    # ---------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric("💰 Predicted Revenue", f"{prediction:.2f}")
        st.metric("⚠️ Risk Level", risk)

    with col2:
        st.subheader("💡 Insight")
        st.write(insight)

        st.subheader("📌 Recommendation")
        st.write(recommendation)

    # ---------------------------
    # GRAPH SECTION
    # ---------------------------
    st.subheader("📊 Revenue vs Threshold")

    fig, ax = plt.subplots()
    ax.bar(["Predicted Revenue", "Threshold"], [prediction, threshold])
    st.pyplot(fig)

    # ---------------------------
    # EXTRA ANALYSIS
    # ---------------------------
    st.subheader("📈 Transaction Analysis")

    if discount > 0.3:
        st.warning("High discount detected — may reduce profit.")
    elif quantity > 10:
        st.success("Bulk order — good revenue potential.")
    else:
        st.info("Normal transaction pattern.")

# ---------------------------
# AI CHAT
# ---------------------------
st.markdown("---")
st.subheader("🤖 AI Business Assistant")

import os
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

user_query = st.text_input("Ask business questions:")

if st.button("Get AI Insight"):
    if prediction is None:
        st.warning("⚠️ Run prediction first")
    else:
        prompt = f"""
        You are a business analyst AI.

        Transaction Details:
        Discount: {discount}
        Quantity: {quantity}
        Shipping Cost: {shipping_cost}
        Period: {season}
        Predicted Revenue: {prediction}

        User Question: {user_query}

        Provide:
        - Insight
        - Risk explanation
        - Recommendation
        """

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        st.write(response.choices[0].message.content)
