import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn

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

# FastAPI app
app = FastAPI()

class InferenceRequest(BaseModel):
    first_purchase_day: float
    days_since_install: float
    sum_cumulative_levels_day1_1: float
    sum_cumulative_levels_day1_3: float
    sum_cumulative_levels_day1_7: float
    sum_cumulative_levels_day1_14: float
    avg_cumulative_levels_day1_3: float
    avg_cumulative_levels_day1_7: float
    avg_cumulative_levels_day1_14: float
    completion_rate_day1_1: float
    completion_rate_day1_3: float
    completion_rate_day1_7: float
    completion_rate_day1_14: float
    max_level_reach_day1_1: float
    max_level_reach_day1_3: float
    max_level_reach_day1_7: float
    max_level_reach_day1_14: float
    revenue_day1_1: float
    revenue_day1_3: float
    revenue_day1_7: float
    is_android_user: float
    is_return_next_day: float
    is_female: float
    is_age_30: float

@app.post("/predict")
def predict(request: InferenceRequest):
    try:
        data = pd.DataFrame([request.dict()])
        # Cast columns to correct types for MLflow model signature
        int_columns = [
            "first_purchase_day", "days_since_install", "sum_cumulative_levels_day1_1",
            "sum_cumulative_levels_day1_3", "sum_cumulative_levels_day1_7", "sum_cumulative_levels_day1_14",
            "max_level_reach_day1_1", "max_level_reach_day1_3", "max_level_reach_day1_7", "max_level_reach_day1_14"
        ]
        for col in int_columns:
            data[col] = data[col].astype(int)
        data = data[all_features]  # Ensure column order
        prediction = model.predict(data)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
