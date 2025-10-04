import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from google.cloud import bigquery
import mlflow

# Ensure MLflow uses the correct tracking URI (local mlruns directory)
mlflow.set_tracking_uri("file:///Users/lilyle/Documents/datascience-demo/training_pipeline/mlruns")
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Initialize BigQuery client
client = bigquery.Client(project="tactile-471816")

# Define features
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
target = "revenue_day1_20"

# Table
table_id = "tactile-471816.data_analyst_test_local.train_test_split"
feature_str = ", ".join([f"`{col}`" for col in all_features + [target]])

# Read train and test sets
df_train = client.query(f"SELECT {feature_str} FROM `{table_id}` WHERE split_set = 'train'").to_dataframe()
df_test = client.query(f"SELECT {feature_str} FROM `{table_id}` WHERE split_set = 'test'").to_dataframe()

X_train = df_train[all_features].copy()
y_train = df_train[target].copy()
X_test = df_test[all_features].copy()
y_test = df_test[target].copy()


# Convert categorical features to float to avoid MLflow integer/NaN warning
for cat_col in categorical_features:
    X_train[cat_col] = X_train[cat_col].astype(float)
    X_test[cat_col] = X_test[cat_col].astype(float)



# Set MLflow experiment
mlflow.set_experiment("xgboost_optuna_experiment")
mlflow_client = MlflowClient()
parent_run = mlflow.start_run(run_name="optuna_parent_run")
parent_run_id = parent_run.info.run_id

# Optuna hyperparameter tuning
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
        "random_state": 42,
    }
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True) as child_run:
        mlflow_client.set_tag(child_run.info.run_id, "mlflow.parentRunId", parent_run_id)
        mlflow.log_params(params)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        mlflow.log_metrics({
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mse": mse
        })
        # Log model with input example and signature in the child run (for registry and later use)
        input_example = X_train.dropna().head(1)
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )
        return mse

 
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=3)


# Register the best model in the MLflow Model Registry

# Find best child run by trial number and parentRunId
best_trial_number = study.best_trial.number
best_run_id = None
experiment_id = mlflow.get_experiment_by_name("xgboost_optuna_experiment").experiment_id
for run in mlflow_client.search_runs([experiment_id]):
    run_name = run.data.tags.get("mlflow.runName")
    parent_id = run.data.tags.get("mlflow.parentRunId")
    if run_name == f"trial_{best_trial_number}" and parent_id == parent_run_id:
        best_run_id = run.info.run_id
        break
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri, "xgboost_optuna_best_model")
else:
    pass

mlflow.end_run()  # End parent run

