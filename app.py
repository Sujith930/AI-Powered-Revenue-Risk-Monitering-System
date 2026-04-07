import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from groq import Groq
import os

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="AI Revenue System", layout="wide")

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
# SESSION STATE
# ---------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ---------------------------
# SIDEBAR INPUTS
# ---------------------------
st.sidebar.header("📥 Enter Transaction Details")

discount = st.sidebar.slider("Discount (%)", 0.0, 1.0, 0.1)
quantity = st.sidebar.number_input("Quantity", 1, 20, 5)
shipping_cost = st.sidebar.number_input("Shipping Cost", 0.0, 1000.0, 50.0)

season = st.sidebar.selectbox(
    "Business Period",
    ["Regular", "Festive Season", "Off Season"]
)

# Convert business period → month
if season == "Festive Season":
    month = 12
elif season == "Off Season":
    month = 6
else:
    month = 3

year = 2015  # hidden (for model compatibility)

# ---------------------------
# PREDICTION BUTTON
# ---------------------------
if st.sidebar.button("🚀 Predict Revenue"):

    input_data = np.array([[discount, quantity, shipping_cost, month, year]])
    st.session_state.prediction = model.predict(input_data)[0]

# ---------------------------
# DISPLAY RESULTS
# ---------------------------
if st.session_state.prediction is not None:

    prediction = st.session_state.prediction
    threshold = 1128

    if prediction < threshold:
        risk = "🔴 High Risk"
        insight = "High discount or low quantity is reducing expected revenue."
        recommendation = "Reduce discount or increase quantity."
    else:
        risk = "🟢 Normal"
        insight = "Transaction is expected to generate stable revenue."
        recommendation = "Maintain current pricing strategy."

    col1, col2 = st.columns(2)

    with col1:
        st.metric("💰 Predicted Revenue", f"{prediction:.2f}")
        st.metric("⚠️ Risk Level", risk)

    with col2:
        st.subheader("💡 Insight")
        st.write(insight)

        st.subheader("📌 Recommendation")
        st.write(recommendation)

    st.success("✅ Prediction ready — you can now ask AI for insights")

    # ---------------------------
    # GRAPH (DISCOUNT IMPACT)
    # ---------------------------
    st.subheader("📊 Revenue Sensitivity Analysis (Discount Impact)")

    discount_range = np.linspace(0, 1, 10)
    revenues = []

    for d in discount_range:
        temp_input = np.array([[d, quantity, shipping_cost, month, year]])
        rev = model.predict(temp_input)[0]
        revenues.append(rev)

    fig, ax = plt.subplots()
    ax.plot(discount_range, revenues, marker='o')
    ax.set_xlabel("Discount")
    ax.set_ylabel("Predicted Revenue")
    ax.set_title("Impact of Discount on Revenue")

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
        st.info("Normal transaction pattern")

# ---------------------------
# AI CHAT
# ---------------------------
st.markdown("---")
st.subheader("🤖 AI Business Assistant")

user_query = st.text_input("Ask business questions:")

if st.button("Get AI Insight"):

    if st.session_state.prediction is None:
        st.warning("⚠️ Please run prediction first")

    else:
        prediction = st.session_state.prediction
        
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

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
        - Clear business insight
        - Risk explanation
        - Actionable recommendation
        Keep it concise and practical.
        """

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        st.subheader("🤖 AI Response")
        st.write(response.choices[0].message.content)
