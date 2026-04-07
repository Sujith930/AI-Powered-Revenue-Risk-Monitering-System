import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Revenue Risk System", layout="wide")

st.title("📊 AI Revenue Risk Monitoring System")

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("revenue_model.pkl")

# -------------------------------
# SESSION STATE
# -------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# -------------------------------
# INPUTS
# -------------------------------
st.subheader("Enter Business Inputs")

col1, col2 = st.columns(2)

with col1:
    discount = st.slider("Discount (%)", 0, 50, 10) / 100
    quantity = st.number_input("Quantity Sold", min_value=1, value=50)

    if quantity > 1000:
        st.warning("⚠️ Large quantity — prediction may be less accurate")

with col2:
    shipping_cost = st.number_input("Shipping Cost", min_value=0.0, value=50.0)
    season = st.selectbox("Business Period", ["Off-Season", "Regular", "Peak"])

# -------------------------------
# MODEL COMPATIBILITY
# -------------------------------
season_to_month = {
    "Off-Season": 2,
    "Regular": 6,
    "Peak": 11
}

month = season_to_month[season]
year = 2015  # fixed

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Revenue"):

    input_data = np.array([[discount, quantity, shipping_cost, month, year]])

    prediction = model.predict(input_data)[0]
    st.session_state.prediction = prediction

    threshold = 1128

    st.subheader("📈 Prediction Result")
    st.write(f"Estimated Revenue: ₹ {prediction:.2f}")

    if prediction < threshold:
        st.error("⚠️ High Risk: Revenue is below expected level")
    else:
        st.success("✅ Healthy Revenue Expected")

# -------------------------------
# BUSINESS GRAPH
# -------------------------------
if st.session_state.prediction is not None:

    st.subheader("📊 Pricing Strategy Insight")

    discounts = np.linspace(0, 0.5, 20)
    revenues = []

    for d in discounts:
        temp_input = np.array([[d, quantity, shipping_cost, month, year]])
        revenues.append(model.predict(temp_input)[0])

    fig, ax = plt.subplots()
    ax.plot(discounts * 100, revenues)
    ax.axhline(y=1128, linestyle='--')

    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Predicted Revenue")
    ax.set_title("Impact of Discount on Revenue")

    st.pyplot(fig)

    st.info("💡 Insight: Increasing discounts beyond a point reduces revenue. Identify optimal pricing.")

# -------------------------------
# TRANSACTION ANALYSIS
# -------------------------------
if st.session_state.prediction is not None:

    st.subheader("📌 Transaction Insights")

    if quantity > 100:
        st.success("Bulk order → Strong revenue opportunity")
    elif discount > 0.3:
        st.warning("High discount → Profit margin risk")
    elif shipping_cost > 100:
        st.warning("High shipping cost → Optimize logistics")
    else:
        st.info("Balanced transaction")

# -------------------------------
# AI BUSINESS ASSISTANT
# -------------------------------
st.subheader("🤖 AI Business Assistant")

user_query = st.text_input("Ask business questions:")

if st.button("💡 Get AI Insight"):

    if st.session_state.prediction is None:
        st.warning("⚠️ Please run prediction first")

    elif not user_query:
        st.warning("⚠️ Please enter a question")

    else:
        try:
            from groq import Groq

            client = Groq(api_key=st.secrets["GROQ_API_KEY"])

            prediction = st.session_state.prediction

            prompt = f"""
You are a business analyst AI.

Transaction Details:
- Discount: {discount}
- Quantity: {quantity}
- Shipping Cost: {shipping_cost}
- Season: {season}
- Predicted Revenue: {prediction}

User Question:
{user_query}

Provide clear, simple, and actionable business insights.
"""

            response = client.chat.completions.create(
                model="llama3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )

            st.subheader("💡 AI Insights")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"AI Error: {e}")
