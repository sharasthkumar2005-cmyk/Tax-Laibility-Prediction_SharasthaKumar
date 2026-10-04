"""
app.py
======
Tax Liability Prediction — Streamlit Application
-------------------------------------------------
This is the user-facing web application.

It:
  1. Loads the pre-trained ML pipeline (tax_liability_model.pkl)
  2. Presents a clean form for the user to enter their financial details
  3. Passes the inputs through the same pipeline used during training
  4. Displays the predicted tax liability

IMPORTANT: Run train_model.py first to generate the model file.
Then launch this app with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import json
import os

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tax Liability Prediction",
    page_icon="💰",
    layout="centered",
)

# ─────────────────────────────────────────────
# Helper: Load Model & Metrics
# ─────────────────────────────────────────────
MODEL_PATH   = "tax_liability_model.pkl"
METRICS_PATH = "model_metrics.json"


@st.cache_resource
def load_model():
    """
    Load the saved ML pipeline.
    @st.cache_resource ensures the model is loaded only once,
    not on every user interaction — this makes the app faster.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_metrics():
    """Load the saved model evaluation metrics."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def format_inr(amount: float) -> str:
    """Format a number as Indian Rupee currency string."""
    return f"₹{amount:,.0f}"


# ─────────────────────────────────────────────
# Load Resources
# ─────────────────────────────────────────────
model   = load_model()
metrics = load_metrics()

# ─────────────────────────────────────────────
# App Header
# ─────────────────────────────────────────────
st.title("💰 Tax Liability Prediction")
st.markdown("**Estimate your tax liability using Machine Learning**")
st.markdown("---")

# Show a warning if the model hasn't been trained yet
if model is None:
    st.error(
        "⚠️ Model file not found. Please run the training script first:\n\n"
        "```\npython train_model.py\n```"
    )
    st.stop()

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
st.subheader("📋 Enter Taxpayer Information")
st.info(
    "Fill in the details below and click **Predict Tax Liability**. "
    "All amounts are in Indian Rupees (₹)."
)

# ── Section 1: Personal Information ─────────
st.markdown("#### 👤 Section 1 — Personal Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        help="Your current age in years (18–100)",
    )

with col2:
    employment_type = st.selectbox(
        "Employment Type",
        options=["Salaried", "Self-Employed", "Business"],
        help="Select the category that best describes your employment",
    )

# ── Section 2: Income Details ────────────────
st.markdown("#### 💼 Section 2 — Income Details")
col3, col4 = st.columns(2)

with col3:
    annual_income = st.number_input(
        "Annual Income (₹)",
        min_value=0,
        max_value=100_000_000,
        value=500_000,
        step=10_000,
        help="Total salary or business income earned during the year",
    )

with col4:
    other_income = st.number_input(
        "Other Income (₹)",
        min_value=0,
        max_value=10_000_000,
        value=0,
        step=1_000,
        help="Income from rent, interest, freelance, or any other source",
    )

# ── Section 3: Deductions & Investments ──────
st.markdown("#### 📉 Section 3 — Deductions & Investments")
col5, col6 = st.columns(2)
col7, col8 = st.columns(2)

with col5:
    deductions = st.number_input(
        "Deductions (₹)",
        min_value=0,
        max_value=5_000_000,
        value=50_000,
        step=5_000,
        help="Standard deduction or Section 80C eligible deductions",
    )

with col6:
    investments = st.number_input(
        "Investments (₹)",
        min_value=0,
        max_value=5_000_000,
        value=50_000,
        step=5_000,
        help="Tax-saving investments (e.g., PPF, ELSS, NSC)",
    )

with col7:
    home_loan_interest = st.number_input(
        "Home Loan Interest (₹)",
        min_value=0,
        max_value=5_000_000,
        value=0,
        step=5_000,
        help="Annual interest paid on home loan (Section 24 deduction)",
    )

with col8:
    health_insurance = st.number_input(
        "Health Insurance Premium (₹)",
        min_value=0,
        max_value=500_000,
        value=0,
        step=1_000,
        help="Health insurance premium paid (Section 80D deduction)",
    )

# ── Section 4: Previous Tax ──────────────────
st.markdown("#### 🧾 Section 4 — Previous Tax")
previous_tax = st.number_input(
    "Previous Tax Paid (₹)",
    min_value=0,
    max_value=10_000_000,
    value=0,
    step=1_000,
    help="TDS already deducted or advance tax already paid this year",
)

st.markdown("---")

# ─────────────────────────────────────────────
# Prediction Button & Logic
# ─────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Tax Liability", type="primary", use_container_width=True)

if predict_clicked:
    # ── Input Validation ──────────────────────
    errors = []

    if annual_income == 0 and other_income == 0:
        errors.append("Please enter at least one income value (Annual Income or Other Income).")

    total_income = annual_income + other_income
    total_deductions = deductions + investments + home_loan_interest + health_insurance

    if total_deductions > total_income:
        errors.append(
            f"Total deductions ({format_inr(total_deductions)}) cannot exceed "
            f"total income ({format_inr(total_income)}). Please review your inputs."
        )

    if errors:
        for err in errors:
            st.warning(f"⚠️ {err}")
    else:
        # ── Build Input DataFrame ─────────────
        # The column names and order MUST match exactly what was used during training.
        input_data = pd.DataFrame({
            "Age":                    [age],
            "Employment_Type":        [employment_type],
            "Annual_Income_INR":      [annual_income],
            "Other_Income_INR":       [other_income],
            "Deductions_INR":         [deductions],
            "Investments_INR":        [investments],
            "Home_Loan_Interest_INR": [home_loan_interest],
            "Health_Insurance_INR":   [health_insurance],
            "Previous_Tax_Paid_INR":  [previous_tax],
        })

        # ── Generate Prediction ───────────────
        try:
            predicted_liability = model.predict(input_data)[0]
            # Ensure prediction is non-negative (tax can't be negative)
            predicted_liability = max(0.0, predicted_liability)

            # ── Display Results ───────────────
            st.markdown("---")
            st.success("✅ Prediction Complete!")
            st.markdown("### 📊 Prediction Results")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.metric(
                    label="💰 Estimated Tax Liability",
                    value=format_inr(predicted_liability),
                )

            # Estimated tax payable = predicted liability - tax already paid
            tax_payable = max(0.0, predicted_liability - previous_tax)
            with res_col2:
                st.metric(
                    label="📤 Estimated Tax Payable",
                    value=format_inr(tax_payable),
                    help="Predicted liability minus Previous Tax Paid",
                )

            # ── Breakdown Summary ─────────────
            with st.expander("📋 View Input Summary"):
                summary_df = pd.DataFrame({
                    "Field": [
                        "Age", "Employment Type",
                        "Annual Income", "Other Income",
                        "Deductions", "Investments",
                        "Home Loan Interest", "Health Insurance",
                        "Previous Tax Paid",
                    ],
                    "Value": [
                        age, employment_type,
                        format_inr(annual_income), format_inr(other_income),
                        format_inr(deductions), format_inr(investments),
                        format_inr(home_loan_interest), format_inr(health_insurance),
                        format_inr(previous_tax),
                    ],
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # ── Academic Disclaimer ───────────
            st.info(
                "📌 **Academic Note:** The *Estimated Tax Payable* is calculated by "
                "subtracting the previous tax paid from the predicted liability. "
                "This is a simplified academic calculation and **should not be "
                "treated as official tax advice.**"
            )

        except Exception:
            st.error(
                "❌ An error occurred while generating the prediction. "
                "Please check your inputs and try again."
            )

# ─────────────────────────────────────────────
# About the Model Section
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🤖 About the Model")

if metrics:
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Model Used", metrics.get("model_name", "—").replace(" Regressor", ""))
    with m_col2:
        st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
    with m_col3:
        st.metric("MAE", format_inr(metrics.get("mae", 0)))
    with m_col4:
        st.metric("Training Records", metrics.get("train_records", "—"))

    st.markdown(
        f"**Input Features ({metrics.get('num_features', 9)}):** "
        + ", ".join(
            f"`{f}`" for f in metrics.get("feature_names", [])
        )
    )
else:
    st.warning("Model metrics not available. Please run `train_model.py` first.")

# ─────────────────────────────────────────────
# Disclaimer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This application is an academic Machine Learning project "
    "using synthetic data. The predicted amount is an estimate and should not be "
    "considered official tax advice. For actual tax filing, please consult a "
    "certified tax professional or use the official Income Tax Department portal."
)
