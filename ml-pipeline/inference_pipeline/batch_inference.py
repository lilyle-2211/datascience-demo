import mlflow

# Ensure MLflow uses the correct tracking URI (local mlruns directory)
mlflow.set_tracking_uri("file:///Users/lilyle/Documents/datascience-demo/training_pipeline/mlruns")
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

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

# 1. Load best model from MLflow Model Registry (latest version)
model_name = "xgboost_optuna_best_model"
model_uri = f"models:/{model_name}/latest"
model = mlflow.pyfunc.load_model(model_uri)

# 2. Create sample inference data (as pandas DataFrame)
sample_data = pd.DataFrame({
    "first_purchase_day": [1, 2],
    "days_since_install": [10, 20],
    "sum_cumulative_levels_day1_1": [5, 6],
    "sum_cumulative_levels_day1_3": [10, 12],
    "sum_cumulative_levels_day1_7": [20, 22],
    "sum_cumulative_levels_day1_14": [30, 32],
    "avg_cumulative_levels_day1_3": [3.3, 4.0],
    "avg_cumulative_levels_day1_7": [6.6, 7.3],
    "avg_cumulative_levels_day1_14": [10.0, 10.5],
    "completion_rate_day1_1": [0.8, 0.9],
    "completion_rate_day1_3": [0.7, 0.8],
    "completion_rate_day1_7": [0.6, 0.7],
    "completion_rate_day1_14": [0.5, 0.6],
    "max_level_reach_day1_1": [2, 3],
    "max_level_reach_day1_3": [4, 5],
    "max_level_reach_day1_7": [6, 7],
    "max_level_reach_day1_14": [8, 9],
    "revenue_day1_1": [1.0, 2.0],
    "revenue_day1_3": [2.0, 3.0],
    "revenue_day1_7": [3.0, 4.0],
    "is_android_user": [1.0, 0.0],
    "is_return_next_day": [0.0, 1.0],
    "is_female": [1.0, 0.0],
    "is_age_30": [0.0, 1.0]
})

# 3. Start Spark session
spark = SparkSession.builder.appName("BatchInference").getOrCreate()
df_spark = spark.createDataFrame(sample_data)

# 4. Define Pandas UDF for inference
@pandas_udf(DoubleType())
def predict_udf(*cols):
    X = pd.concat(cols, axis=1)
    X.columns = all_features
    preds = model.predict(X)
    return pd.Series(preds)

# 5. Apply UDF for scalable inference
# (order columns to match training)
df_spark = df_spark.select(*all_features)
df_spark.show(1, vertical=True)

df_pred = df_spark.withColumn("prediction", predict_udf(*[df_spark[c] for c in all_features]))

df_pred.show(1, vertical=True)

# Stop Spark session
spark.stop()
