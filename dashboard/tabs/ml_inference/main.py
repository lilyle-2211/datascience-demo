"""ML Model Inference tab main module."""

import streamlit as st
from ...utils.styling import create_section_header

def render_ml_inference_tab():
    """Render the ML Model Inference tab with batch and real-time inference subtabs."""
    
    st.markdown(create_section_header("ML Model Inference"), unsafe_allow_html=True)
    
    # Create subtabs for batch and real-time inference
    inference_tabs = st.tabs(["Batch Inference", "Real-time Inference"])
    
    with inference_tabs[0]:  # Batch Inference
        render_batch_inference()
    
    with inference_tabs[1]:  # Real-time Inference
        render_realtime_inference()

def render_batch_inference():
    """Render the batch inference section."""
    
    st.markdown("## Batch Inference")
    st.markdown("Process large volumes of data in scheduled batches for predictions.")
    
    st.markdown("""
    <div class="success-card">
    <h4>Batch Inference Characteristics</h4>
    <ul>
    <li><strong>High Throughput:</strong> Process thousands/millions of records efficiently</li>
    <li><strong>Scheduled Processing:</strong> Run on regular intervals (hourly, daily, weekly)</li>
    <li><strong>Cost-Effective:</strong> Optimize resource usage for large datasets</li>
    <li><strong>Latency Tolerant:</strong> Results don't need to be immediate</li>
    <li><strong>Quality Assurance:</strong> Time for validation and quality checks</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Cases
    with st.expander("Common Use Cases", expanded=False):
        st.markdown("""
        **Financial Services**
        - Credit risk scoring for loan portfolios
        - Fraud detection on transaction batches
        - Customer segmentation for marketing campaigns
        
        **E-commerce**
        - Product recommendation updates
        - Demand forecasting for inventory
        - Customer lifetime value calculations
        
        **Healthcare**
        - Medical image analysis for diagnostics
        - Drug discovery compound screening
        - Population health analytics
        
        **Manufacturing**
        - Predictive maintenance schedules
        - Quality control analysis
        - Supply chain optimization
        """)
    
    # Implementation Example
    with st.expander("Implementation Example - Python/Spark", expanded=False):
        st.code("""
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
import mlflow

def batch_inference_pipeline():
    # Initialize Spark session
    spark = SparkSession.builder \\
        .appName("BatchInference") \\
        .config("spark.sql.adaptive.enabled", "true") \\
        .getOrCreate()
    
    # Load model from MLflow
    model_uri = "models:/customer_churn_model/production"
    model = mlflow.spark.load_model(model_uri)
    
    # Load batch data
    batch_data = spark.read.parquet("s3://data-lake/customer_features/2024-01-01/")
    
    # Data preprocessing
    feature_cols = ["age", "tenure", "monthly_charges", "total_charges"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    batch_data_processed = assembler.transform(batch_data)
    
    # Generate predictions
    predictions = model.transform(batch_data_processed)
    
    # Select relevant columns
    results = predictions.select(
        "customer_id", 
        "prediction", 
        "probability",
        "features"
    )
    
    # Save results
    results.write \\
        .mode("overwrite") \\
        .parquet("s3://predictions/customer_churn/2024-01-01/")
    
    # Generate summary statistics
    summary = results.groupBy("prediction").count()
    summary.show()
    
    spark.stop()

# Schedule with Apache Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

dag = DAG(
    'batch_inference_pipeline',
    default_args={
        'owner': 'data-science-team',
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    },
    description='Daily batch inference for customer churn',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False
)

inference_task = PythonOperator(
    task_id='run_batch_inference',
    python_callable=batch_inference_pipeline,
    dag=dag
)
        """, language='python')
    
    # Architecture Patterns
    with st.expander("Architecture Patterns", expanded=False):
        st.markdown("**ETL/ELT Pipeline Pattern**")
        st.code("""
# Extract -> Transform -> Load -> Predict pattern
1. Extract: Pull data from sources (databases, APIs, files)
2. Transform: Clean, feature engineer, prepare data
3. Load: Store processed data in feature store
4. Predict: Run ML model inference
5. Store: Save predictions to database/data lake
        """)
        
        st.markdown("**Lambda Architecture Pattern**")
        st.code("""
# Batch + Speed Layer for comprehensive processing
Batch Layer: 
- Historical data processing with Spark/Hadoop
- High accuracy, complete data processing
- Results stored in batch views

Speed Layer:
- Real-time stream processing with Kafka/Storm
- Lower latency, approximate results
- Results stored in real-time views

Serving Layer:
- Combines batch and real-time views
- Provides unified query interface
        """)
    
    # Performance Optimization
    with st.expander("Performance Optimization", expanded=False):
        st.markdown("""
        **Data Partitioning**
        - Partition by date, region, or other logical boundaries
        - Enable parallel processing across partitions
        - Reduce data scanning and improve performance
        
        **Caching Strategies**
        - Cache frequently accessed features
        - Use in-memory stores like Redis for hot data
        - Implement smart cache invalidation policies
        
        **Resource Management**
        - Scale compute resources based on data volume
        - Use spot instances for cost optimization
        - Implement auto-scaling for variable workloads
        
        **Model Optimization**
        - Use model compression techniques
        - Implement model ensembling for better accuracy
        - Consider quantization for faster inference
        """)

def render_realtime_inference():
    """Render the real-time inference section."""
    
    st.markdown("## Real-time Inference")
    st.markdown("Serve predictions with low latency for immediate decision making.")
    
    st.markdown("""
    <div class="warning-card">
    <h4>Real-time Inference Characteristics</h4>
    <ul>
    <li><strong>Low Latency:</strong> Response times in milliseconds</li>
    <li><strong>High Availability:</strong> 99.9%+ uptime requirements</li>
    <li><strong>Scalable:</strong> Handle varying request volumes</li>
    <li><strong>Immediate Results:</strong> Predictions needed for real-time decisions</li>
    <li><strong>Resource Intensive:</strong> Always-on infrastructure costs</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Use Cases
    with st.expander("Common Use Cases", expanded=False):
        st.markdown("""
        **Financial Services**
        - Real-time fraud detection during transactions
        - Algorithmic trading decisions
        - Credit approval at point of sale
        
        **E-commerce**
        - Product recommendations during browsing
        - Dynamic pricing based on demand
        - Cart abandonment prevention
        
        **Digital Advertising**
        - Real-time bid optimization in ad auctions
        - Content personalization
        - A/B test assignment
        
        **Gaming & Entertainment**
        - Player behavior prediction
        - Content recommendation engines
        - Cheat detection systems
        """)
    
    # Implementation Example
    with st.expander("Implementation Example - FastAPI/Docker", expanded=False):
        st.code("""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
from typing import List
import logging

# Initialize FastAPI app
app = FastAPI(title="Real-time ML Inference API")

# Load model at startup
model = None
feature_names = None

@app.on_event("startup")
async def load_model():
    global model, feature_names
    try:
        model = joblib.load("models/customer_churn_model.pkl")
        feature_names = ["age", "tenure", "monthly_charges", "total_charges"]
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")

# Request/Response models
class PredictionRequest(BaseModel):
    age: int
    tenure: int
    monthly_charges: float
    total_charges: float

class PredictionResponse(BaseModel):
    customer_id: str = None
    prediction: int
    probability: float
    confidence: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Prepare features
        features = np.array([[
            request.age,
            request.tenure, 
            request.monthly_charges,
            request.total_charges
        ]])
        
        # Generate prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        # Determine confidence level
        confidence = "high" if probability > 0.8 else "medium" if probability > 0.6 else "low"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    try:
        # Process multiple requests efficiently
        features_batch = np.array([[
            req.age, req.tenure, req.monthly_charges, req.total_charges
        ] for req in requests])
        
        predictions = model.predict(features_batch)
        probabilities = model.predict_proba(features_batch).max(axis=1)
        
        return [
            PredictionResponse(
                prediction=int(pred),
                probability=float(prob),
                confidence="high" if prob > 0.8 else "medium" if prob > 0.6 else "low"
            )
            for pred, prob in zip(predictions, probabilities)
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
        """, language='python')
    
    # Docker Configuration
    with st.expander("Docker & Kubernetes Deployment", expanded=False):
        st.markdown("**Dockerfile**")
        st.code("""
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
        """, language='dockerfile')
        
        st.markdown("**Kubernetes Deployment**")
        st.code("""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference-api
  template:
    metadata:
      labels:
        app: ml-inference-api
    spec:
      containers:
      - name: api
        image: ml-inference-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: MODEL_PATH
          value: "/models/customer_churn_model.pkl"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-inference-service
spec:
  selector:
    app: ml-inference-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
        """, language='yaml')
    
    # Performance Considerations
    with st.expander("Performance & Scaling Strategies", expanded=False):
        st.markdown("""
        **Latency Optimization**
        - Model optimization (quantization, pruning)
        - Caching frequent predictions
        - Asynchronous processing where possible
        - Connection pooling and keep-alive
        
        **Scaling Strategies**
        - Horizontal Pod Autoscaling (HPA) based on CPU/memory
        - Custom metrics scaling (request rate, latency)
        - Blue-green deployments for zero-downtime updates
        - Circuit breakers for fault tolerance
        
        **Monitoring & Observability**
        - Request/response logging
        - Latency and throughput metrics
        - Model drift detection
        - Error rate monitoring
        - Distributed tracing
        """)
        
        st.code("""
# Example monitoring with Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest
import time

# Metrics
prediction_counter = Counter('ml_predictions_total', 'Total predictions made')
prediction_latency = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
error_counter = Counter('ml_prediction_errors_total', 'Total prediction errors')

@app.post("/predict")
async def predict_with_monitoring(request: PredictionRequest):
    start_time = time.time()
    
    try:
        # Your prediction logic here
        result = await make_prediction(request)
        prediction_counter.inc()
        return result
        
    except Exception as e:
        error_counter.inc()
        raise e
        
    finally:
        prediction_latency.observe(time.time() - start_time)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
        """, language='python')