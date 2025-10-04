import streamlit as st
import mlflow
import pandas as pd

# Set MLflow tracking URI to match training
mlflow.set_tracking_uri("file:///Users/lilyle/Documents/datascience-demo/training_pipeline/mlruns")

# Define features (should match training)
numeric_features = [
    "first_purchase_day", "days_since_install", "sum_cumulative_levels_day1_1",
    "sum_cumulative_levels_day1_3", "sum_cumulative_levels_day1_7", "sum_cumulative_levels_day1_14",
    "avg_cumulative_levels_day1_3", "avg_cumulative_levels_day1_7", "avg_cumulative_levels_day1_14",
    "completion_rate_day1_1", "completion_rate_day1_3", "completion_rate_day1_7", "completion_rate_day1_14",
    "max_level_reach_day1_1", "max_level_reach_day1_3", "max_level_reach_day1_7", "max_level_reach_day1_14",
    "revenue_day1_1", "revenue_day1_3", "revenue_day1_7"
]
categorical_features = [
    "is_android_user", "is_return_next_day", "is_female", "is_age_30"
]
all_features = numeric_features + categorical_features

# Load latest model from MLflow Model Registry
model_name = "xgboost_optuna_best_model"
model_uri = f"models:/{model_name}/latest"
model = mlflow.pyfunc.load_model(model_uri)

st.title("Real-Time LTV Prediction")
st.write("Enter customer features to get a prediction from the latest trained model.")

with st.form("inference_form"):
    input_data = {}
    for col in numeric_features:
        input_data[col] = st.number_input(col, value=0)
    for col in categorical_features:
        input_data[col] = st.selectbox(col, [0.0, 1.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
    submitted = st.form_submit_button("Predict")

if submitted:
    df = pd.DataFrame([input_data])
    # Cast integer columns to int for MLflow model signature
    int_columns = [
        "first_purchase_day", "days_since_install", "sum_cumulative_levels_day1_1",
        "sum_cumulative_levels_day1_3", "sum_cumulative_levels_day1_7", "sum_cumulative_levels_day1_14",
        "max_level_reach_day1_1", "max_level_reach_day1_3", "max_level_reach_day1_7", "max_level_reach_day1_14"
    ]
    for col in int_columns:
        df[col] = df[col].astype(int)
    df = df[all_features]
    pred = model.predict(df)
    st.success(f"Predicted LTV: {pred[0]:.2f}")
