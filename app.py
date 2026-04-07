import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="AI Revenue Risk System", layout="wide")

st.title("📊 AI-Powered Revenue Risk Monitoring System")

# ------------------------------
# LOAD MODEL
# ------------------------------
model = joblib.load("revenue_model.pkl")

# ------------------------------
# SESSION STATE
# ------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ------------------------------
# INPUTS
# ------------------------------
st.subheader("🧾 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    discount = st.slider("Discount", 0.0, 1.0, 0.1)
    quantity = st.number_input("Quantity", min_value=1, value=50)

with col2:
    shipping_cost = st.number_input("Shipping Cost", min_value=0.0, value=50.0)
    season = st.selectbox("Season", ["Off-Season", "Regular", "Peak"])

# Convert season to numeric (model compatibility)
season_map = {"Off-Season": 2, "Regular": 6, "Peak": 11}
month = season_map[season]
year = 2015

# ------------------------------
# PREDICTION
# ------------------------------
if st.button("🔍 Predict Revenue"):
    input_data = np.array([[discount, quantity, shipping_cost, month, year]])
    prediction = model.predict(input_data)[0]
    st.session_state.prediction = prediction

# ------------------------------
# RESULTS
# ------------------------------
if st.session_state.prediction is not None:

    prediction = st.session_state.prediction
    threshold = 1128

    st.subheader("📈 Prediction Result")
    st.write(f"Predicted Revenue: ₹ {prediction:.2f}")

    # ------------------------------
    # RISK SCORE
    # ------------------------------
    risk_score = max(0, min(100, int((threshold - prediction) / threshold * 100)))
    st.metric("⚠️ Revenue Risk Score", f"{risk_score}%")

    # ------------------------------
    # GRAPH
    # ------------------------------
    st.subheader("📊 Revenue vs Target")

    values = [prediction, threshold]
    labels = ["Predicted Revenue", "Target"]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Revenue")
    st.pyplot(fig)

    # ------------------------------
    # TRANSACTION INSIGHTS
    # ------------------------------
    st.subheader("📌 Transaction Insights")

    if quantity > 100:
        st.success("Bulk order → Strong revenue opportunity")
    elif discount > 0.3:
        st.warning("High discount → Profit margin risk")
    elif shipping_cost > 100:
        st.warning("High shipping cost → Optimize logistics")
    else:
        st.info("Balanced transaction")

    # ------------------------------
    # STRATEGIC RECOMMENDATION
    # ------------------------------
    st.subheader("📊 Strategic Recommendation")

    if prediction < threshold:
        st.error("Focus on improving pricing, reducing discounts, or increasing volume.")
    else:
        st.success("Current strategy is effective. Consider scaling this approach.")

# ------------------------------
# HYBRID AI SECTION
# ------------------------------
st.subheader("🤖 Business Insights Engine")

user_query = st.text_input("Ask a business question:")

# Toggle for LLM
use_llm = st.checkbox("Enable Advanced AI Insights (LLM)")

if st.button("💡 Generate Insights"):

    if st.session_state.prediction is None:
        st.warning("⚠️ Please run prediction first")

    elif not user_query:
        st.warning("⚠️ Please enter a question")

    else:
        prediction = st.session_state.prediction
        threshold = 1128

        # ------------------------------
        # RULE-BASED INSIGHTS (ALWAYS RUN)
        # ------------------------------
        st.subheader("📌 Rule-Based Insights")

        insights = []

        if discount > 0.3:
            insights.append("Reduce discount levels to protect profit margins.")

        if quantity > 100:
            insights.append("Focus on bulk buyers and B2B opportunities.")

        if shipping_cost > 100:
            insights.append("Optimize logistics to reduce shipping costs.")

        if prediction < threshold:
            insights.append("Revenue is below target. Improve pricing or increase sales volume.")
        else:
            insights.append("Revenue is healthy. Consider scaling this strategy.")

        query = user_query.lower()

        if "increase revenue" in query:
            insights.append("Optimize discounts, target high-volume customers, and improve pricing.")

        if "profit" in query:
            insights.append("Reduce discounts and control operational costs to improve margins.")

        if "cost" in query:
            insights.append("Focus on reducing shipping and operational expenses.")

        for insight in insights:
            st.write("•", insight)

        # ------------------------------
        # OPTIONAL LLM INSIGHTS
        # ------------------------------
        if use_llm:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

                prompt = f"""
You are a business analyst.

Transaction Details:
- Discount: {discount}
- Quantity: {quantity}
- Shipping Cost: {shipping_cost}
- Predicted Revenue: {prediction}

User Question:
{user_query}

Provide advanced business insights.
"""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )

                st.subheader("🤖 Advanced AI Insights")
                st.write(response.choices[0].message.content)

            except Exception:
                st.warning("⚠️ LLM not available. Showing rule-based insights only.")
