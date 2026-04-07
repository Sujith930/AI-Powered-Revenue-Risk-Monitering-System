# AI-Powered Revenue Risk Monitoring System

## Overview
This project predicts sales revenue and identifies high-risk transactions using machine learning. It combines predictive modeling with business analytics to support data-driven decision-making.

## Key Features
- Revenue prediction using Linear Regression (R² ≈ 0.52)
- Identification of high-risk transactions
- Discount-profit analysis for business insights
- Interactive Streamlit application
- Dashboard visualization using Power BI

## Dataset
- Global Superstore dataset (~1000 records)
- Features: Discount, Quantity, Shipping Cost, Month, Year

## Model Details
- Algorithm: Linear Regression
- Evaluation Metrics:
  - R² Score: ~0.52
  - MAE: ~580
- Compared multiple models before selecting final model

## Tech Stack
- Python (Pandas, NumPy, Scikit-learn)
- Streamlit
- Power BI
- Joblib

## Deployment
Deployed using Streamlit Cloud: https://ai-powered-revenue-risk-monitering-system.streamlit.app/

## How to Run Locally
pip install -r requirements.txt  
streamlit run app.py
