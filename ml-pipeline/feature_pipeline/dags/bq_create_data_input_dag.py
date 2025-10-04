from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime
import os



# Read SQL from external files
DATA_PREPROCESSING_SQL_PATH = os.path.join(os.path.dirname(__file__), '../data_preprocessing.sql')
with open(DATA_PREPROCESSING_SQL_PATH, 'r') as f:
    data_preprocessing_sql = f.read()

TRAIN_TEST_SPLIT_SQL_PATH = os.path.join(os.path.dirname(__file__), '../train_test_split.sql')
with open(TRAIN_TEST_SPLIT_SQL_PATH, 'r') as f:
    train_test_split_sql = f.read()

default_args = {
    'start_date': datetime(2023, 1, 1),
}



dag = DAG(
    dag_id='bq_create_data_preprocessing_and_split',
    default_args=default_args,
    schedule=None,
    catchup=False,
)



create_data_preprocessing = BigQueryInsertJobOperator(
    task_id='create_data_preprocessing',
    configuration={
        "query": {
            "query": data_preprocessing_sql,
            "useLegacySql": False,
        }
    },
    project_id="tactile-471816",
    location="EU",  # Change to your dataset location
    dag=dag,
)

create_train_test_split = BigQueryInsertJobOperator(
    task_id='create_train_test_split',
    configuration={
        "query": {
            "query": train_test_split_sql,
            "useLegacySql": False,
        }
    },
    project_id="tactile-471816",
    location="EU",  # Change to your dataset location
    dag=dag,
)

# Set dependency: train_test_split runs after data_preprocessing
create_data_preprocessing >> create_train_test_split