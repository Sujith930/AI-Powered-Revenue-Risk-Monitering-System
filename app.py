import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from groq import Groq

# -------------------------
# Load model
# -------------------------
model = joblib.load("revenue_model.pkl")

# -------------------------
# App Title
# -------------------------
st.set_page_config(page_title="AI Revenue System", layout="wide")
st.title("📊 AI Revenue Risk Monitoring System")

# -------------------------
# Inputs
# -------------------------
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    discount = st.slider("Discount", 0.0, 1.0, 0.1)
    quantity = st.number_input("Quantity", min_value=1, value=5)  # ✅ FIXED (no max limit)

with col2:
    shipping_cost = st.number_input("Shipping Cost", min_value=0.0, value=50.0)
    month = st.slider("Month", 1, 12, 6)

# -------------------------
# Season Mapping
# -------------------------
if month in [12, 1, 2]:
    season = "Winter"
elif month in [3, 4, 5]:
    season = "Summer"
elif month in [6, 7, 8]:
    season = "Monsoon"
else:
    season = "Autumn"

# -------------------------
# Prediction
# -------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.button("Predict Revenue"):

    input_data = np.array([[discount, quantity, shipping_cost, month, 2024]])
    prediction = model.predict(input_data)[0]

    st.session_state.prediction = prediction

    threshold = 1128

    if prediction < threshold:
        risk = "High Risk"
        st.error(f"⚠️ High Risk | Predicted Revenue: {prediction:.2f}")
    else:
        risk = "Normal"
        st.success(f"✅ Normal | Predicted Revenue: {prediction:.2f}")

    # -------------------------
    # Graph
    # -------------------------
    st.subheader("📈 Revenue vs Threshold")

    fig, ax = plt.subplots()
    ax.bar(["Predicted Revenue", "Threshold"], [prediction, threshold])
    ax.set_ylabel("Revenue")
    st.pyplot(fig)

    # -------------------------
    # Basic Insights
    # -------------------------
    st.subheader("📊 Transaction Analysis")

    if quantity > 10:
        st.success("Bulk order — good revenue potential.")
    elif discount > 0.5:
        st.warning("High discount — may reduce profit margins.")
    else:
        st.info("Normal transaction pattern.")

# -------------------------
# AI SECTION
# -------------------------
st.subheader("🤖 AI Business Assistant")

user_query = st.text_input("Ask business questions:")

if st.button("Get AI Insight"):

    if st.session_state.prediction is None:
        st.warning("⚠️ Please run prediction first")

    elif not user_query:
        st.warning("⚠️ Please enter a business question")

    else:
        prediction = st.session_state.prediction

        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        prompt = f"""
        You are a business analyst AI.

        Transaction Details:
        Discount: {discount}
        Quantity: {quantity}
        Shipping Cost: {shipping_cost}
        Season: {season}
        Predicted Revenue: {prediction}

        User Question: {user_query}

        Provide clear, practical, and actionable business insights.
        """

        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}]
            )

            st.subheader("💡 AI Insights")
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error("AI service error. Check API key or usage limits.")
