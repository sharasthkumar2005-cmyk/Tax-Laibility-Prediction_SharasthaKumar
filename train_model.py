"""
train_model.py
==============
Tax Liability Prediction — Model Training Script
-------------------------------------------------
This script handles the complete ML workflow:
  1. Load the dataset
  2. Inspect & clean the data
  3. Preprocess features (encoding, scaling)
  4. Train & compare two models (Linear Regression vs Random Forest)
  5. Evaluate using MAE, MSE, RMSE, R²
  6. Save the best model pipeline with joblib

Run this script ONCE before launching the Streamlit app:
  python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# STEP 1 — Load Dataset
# ─────────────────────────────────────────────
DATA_PATH = "data/tax_liability_prediction_dataset.csv"
MODEL_PATH = "tax_liability_model.pkl"
METRICS_PATH = "model_metrics.json"

print("=" * 55)
print("  Tax Liability Prediction — Model Training")
print("=" * 55)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# STEP 2 — Basic Data Checks
# ─────────────────────────────────────────────
print(f"\n[2] Missing values : {df.isnull().sum().sum()}")
print(f"    Duplicate rows  : {df.duplicated().sum()}")

# Drop any duplicates just in case
df = df.drop_duplicates()

# ─────────────────────────────────────────────
# STEP 3 — Define Features & Target
# ─────────────────────────────────────────────
# Taxpayer_ID is just an identifier — NOT a useful feature for prediction.
# We drop it before training.
TARGET = "Tax_Liability_INR"

FEATURE_COLS = [
    "Age",
    "Employment_Type",
    "Annual_Income_INR",
    "Other_Income_INR",
    "Deductions_INR",
    "Investments_INR",
    "Home_Loan_Interest_INR",
    "Health_Insurance_INR",
    "Previous_Tax_Paid_INR",
]

X = df[FEATURE_COLS]
y = df[TARGET]

print(f"\n[3] Features : {FEATURE_COLS}")
print(f"    Target   : {TARGET}")
print(f"    Employment_Type categories: {df['Employment_Type'].unique().tolist()}")

# ─────────────────────────────────────────────
# STEP 4 — Train / Test Split
# ─────────────────────────────────────────────
# random_state=42 makes results reproducible — same split every time.
# test_size=0.2 means 80% training, 20% testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[4] Train size: {len(X_train)} | Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 5 — Build Preprocessing Pipeline
# ─────────────────────────────────────────────
# We use a ColumnTransformer so that:
#   • Numerical columns → StandardScaler (normalises range)
#   • Categorical column (Employment_Type) → OneHotEncoder
#
# Wrapping everything in a Pipeline guarantees the SAME preprocessing
# is applied during both training AND prediction — avoiding data leakage.

numerical_features = [
    "Age",
    "Annual_Income_INR",
    "Other_Income_INR",
    "Deductions_INR",
    "Investments_INR",
    "Home_Loan_Interest_INR",
    "Health_Insurance_INR",
    "Previous_Tax_Paid_INR",
]
categorical_features = ["Employment_Type"]

numerical_transformer = StandardScaler()

# handle_unknown='ignore' ensures unseen categories don't crash the app.
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ─────────────────────────────────────────────
# STEP 6 — Train & Evaluate Both Models
# ─────────────────────────────────────────────

def evaluate_model(name, pipeline, X_tr, y_tr, X_te, y_te):
    """Train a pipeline and return evaluation metrics."""
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_te)
    mae  = mean_absolute_error(y_te, preds)
    mse  = mean_squared_error(y_te, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_te, preds)
    print(f"\n  -- {name} --")
    print(f"    MAE  : ₹{mae:,.2f}")
    print(f"    MSE  : ₹{mse:,.2f}")
    print(f"    RMSE : ₹{rmse:,.2f}")
    print(f"    R²   : {r2:.4f}")
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


print("\n[5] Model Comparison")

# ── Model A: Linear Regression ──────────────
# Simple, fast, and interpretable.
# Works well when the relationship between features and target is roughly linear.
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])
lr_metrics = evaluate_model("Linear Regression", lr_pipeline, X_train, y_train, X_test, y_test)

# ── Model B: Random Forest Regressor ────────
# An ensemble of decision trees that captures non-linear relationships.
# Generally more accurate on real-world financial data.
# n_estimators=100 means 100 trees; random_state ensures reproducibility.
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
])
rf_metrics = evaluate_model("Random Forest Regressor", rf_pipeline, X_train, y_train, X_test, y_test)

# ─────────────────────────────────────────────
# STEP 7 — Select Best Model
# ─────────────────────────────────────────────
# We select the model with the higher R² score.
# R² (R-squared) measures how well the model explains variance in the data.
# Closer to 1.0 is better.

print("\n[6] Model Selection")
if rf_metrics["r2"] >= lr_metrics["r2"]:
    best_pipeline = rf_pipeline
    best_name = "Random Forest Regressor"
    best_metrics = rf_metrics
    print("    ✓ Random Forest selected (higher R²)")
    print("      Reason: Random Forest handles non-linear relationships in")
    print("      financial data better than Linear Regression. It is also")
    print("      robust to outliers, which are common in income/tax data.")
else:
    best_pipeline = lr_pipeline
    best_name = "Linear Regression"
    best_metrics = lr_metrics
    print("    ✓ Linear Regression selected (higher R²)")

# ─────────────────────────────────────────────
# STEP 8 — Save Model & Metrics
# ─────────────────────────────────────────────
joblib.dump(best_pipeline, MODEL_PATH)
print(f"\n[7] Model saved → {MODEL_PATH}")

# Save metrics so the Streamlit app can display them without retraining
metrics_to_save = {
    "model_name": best_name,
    "r2":  round(best_metrics["r2"],  4),
    "mae": round(best_metrics["mae"], 2),
    "rmse": round(best_metrics["rmse"], 2),
    "train_records": len(X_train),
    "num_features": len(FEATURE_COLS),
    "feature_names": FEATURE_COLS,
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics_to_save, f, indent=2)
print(f"    Metrics saved → {METRICS_PATH}")

print("\n" + "=" * 55)
print(f"  Training complete!")
print(f"  Best Model : {best_name}")
print(f"  R² Score   : {best_metrics['r2']:.4f}")
print(f"  MAE        : ₹{best_metrics['mae']:,.2f}")
print(f"  RMSE       : ₹{best_metrics['rmse']:,.2f}")
print("=" * 55)
print("\n  Run the app with:  streamlit run app.py\n")
