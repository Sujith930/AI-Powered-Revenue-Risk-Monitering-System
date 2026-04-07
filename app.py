import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="AI Revenue Risk System", layout="wide")

st.title("📊 AI-Powered Revenue Risk Monitoring System")
st.caption("AI-powered system combining predictive analytics and business intelligence for revenue optimization")

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

# Convert season to model-compatible values
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

    # ------------------------------
    # KPI DASHBOARD
    # ------------------------------
    st.subheader("📊 Key Business Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Revenue", f"₹ {prediction:.2f}")
    col2.metric("Target Revenue", f"₹ {threshold}")
    col3.metric("Revenue Gap", f"₹ {threshold - prediction:.2f}")

    # ------------------------------
    # RISK ANALYSIS
    # ------------------------------
    st.subheader("⚠️ Risk Analysis")

    if prediction >= threshold:
        risk_score = 0
        st.metric("Revenue Risk", "LOW", delta="0%")
        st.success("🟢 Business is in a healthy state")

    elif prediction >= threshold * 0.7:
        risk_score = int(((threshold - prediction) / threshold) * 100)
        st.metric("Revenue Risk", "MEDIUM", delta=f"{risk_score}%")
        st.warning("🟡 Moderate risk – monitor pricing and costs")

    else:
        risk_score = int(((threshold - prediction) / threshold) * 100)
        st.metric("Revenue Risk", "HIGH", delta=f"{risk_score}%")
        st.error("🔴 High risk – immediate action required")

    # ------------------------------
    # GRAPH
    # ------------------------------
    st.subheader("📈 Revenue Performance")

    fig, ax = plt.subplots()
    labels = ["Predicted", "Target"]
    values = [prediction, threshold]
    ax.bar(labels, values)
    ax.set_ylabel("Revenue")
    ax.set_title("Revenue vs Target")
    st.pyplot(fig)

    st.info("💡 Insight: Compare predicted revenue against target to evaluate performance.")

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

# Divider
st.markdown("---")

# ------------------------------
# HYBRID AI SECTION
# ------------------------------
st.subheader("🧠 Intelligent Decision Support System")

user_query = st.text_input("Ask a business question:")
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
        # ADVANCED RULE-BASED INSIGHTS
        # ------------------------------
        st.subheader("📌 Rule-Based Insights")

        insights = []

        # Pricing
        if discount > 0.4:
            insights.append("Excessive discounting is eroding revenue. Revise pricing strategy.")
        elif discount > 0.2:
            insights.append("Moderate discounting detected. Ensure it drives sufficient volume.")

        # Volume
        if quantity > 200:
            insights.append("High-volume transaction → Opportunity for long-term B2B deals.")
        elif quantity < 5:
            insights.append("Low volume → Consider upselling or bundling strategies.")

        # Cost
        if shipping_cost > 150:
            insights.append("Shipping cost is very high → Optimize logistics or vendors.")
        elif shipping_cost > 80:
            insights.append("Shipping cost moderately high → Monitor efficiency.")

        # Revenue
        if prediction < threshold * 0.7:
            insights.append("Revenue critically low → Immediate action required.")
        elif prediction < threshold:
            insights.append("Revenue below target → Improve pricing or sales volume.")
        else:
            insights.append("Revenue healthy → Consider scaling this strategy.")

        # Combined logic
        if discount > 0.3 and quantity < 10:
            insights.append("High discount + low volume → Inefficient strategy causing losses.")

        if shipping_cost > 100 and prediction < threshold:
            insights.append("High cost + low revenue → Urgent cost optimization needed.")

        # Query-based
        query = user_query.lower()

        if "increase revenue" in query:
            insights.append("Increase revenue by optimizing pricing and targeting high-volume customers.")

        if "profit" in query:
            insights.append("Improve profit by reducing discounts and controlling costs.")

        if "cost" in query:
            insights.append("Reduce logistics and operational costs to improve margins.")

        for insight in insights:
            st.write("•", insight)

        # ------------------------------
        # OPTIONAL LLM
        # ------------------------------
        if use_llm:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

                prompt = f"""
You are a business analyst.

Discount: {discount}
Quantity: {quantity}
Shipping Cost: {shipping_cost}
Predicted Revenue: {prediction}

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
