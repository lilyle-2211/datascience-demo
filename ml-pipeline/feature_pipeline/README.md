# ML Feature Pipeline with Airflow Astronomer

## Overview

This project uses **Apache Airflow with Astronomer** to create a robust data aggregation and feature engineering pipeline for machine learning use cases. The pipeline transforms raw data into meaningful features that can be consumed by ML models.

## Purpose

- **Data Aggregation**: Collect and combine data from multiple sources
- **Feature Engineering**: Transform raw data into meaningful features for ML models
- **Data Quality**: Ensure data consistency and quality for downstream ML processes
- **Automation**: Scheduled and orchestrated data processing workflows

## Pipeline Components

### Data Processing Steps

1. **Data Preprocessing** (`data_preprocessing.sql`)
   - Clean and standardize raw data
   - Handle missing values and outliers
   - Create base feature transformations

2. **Train/Test Split** (`train_test_split.sql`)
   - Split processed data into training and testing sets
   - Maintain data integrity across splits
   - Prepare data for ML model training

## Project Structure

### Core Files

- **`dags/`**: Contains Airflow DAG definitions for data processing workflows
  - Data aggregation and transformation pipelines
  - Feature engineering workflows
  - ML data preparation tasks

- **`data_preprocessing.sql`**: SQL queries for data cleaning and initial feature creation
- **`train_test_split.sql`**: SQL logic for splitting data into training and testing sets

### Configuration Files

- **`Dockerfile`**: Astro Runtime Docker image with ML-specific dependencies
- **`requirements.txt`**: Python packages for data processing and ML libraries
- **`packages.txt`**: OS-level packages needed for data operations
- **`airflow_settings.yaml`**: Airflow connections and variables for data sources

### Supporting Directories

- **`include/`**: Additional data processing utilities and helper functions
- **`plugins/`**: Custom Airflow operators for ML-specific tasks
- **`tests/`**: Unit tests for data processing logic

## Key Features

✅ **Automated Data Pipeline**: Scheduled data processing and feature generation
✅ **BigQuery Integration**: Native support for Google BigQuery data operations
✅ **ML-Ready Output**: Data formatted and split for immediate ML model consumption
✅ **Data Quality Checks**: Built-in validation and monitoring
✅ **Scalable Architecture**: Designed to handle large-scale data processing

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Astronomer CLI (`astro`)
- Access to Google BigQuery (for data storage)

### Local Development

1. **Start the Pipeline**:
   ```bash
   astro dev start
   ```

2. **Access Airflow UI**:
   - Navigate to http://localhost:8080/
   - Username: `admin` / Password: `admin`

3. **Monitor Data Processing**:
   - View DAG execution in the Airflow UI
   - Check data quality metrics
   - Monitor feature generation progress

### Data Pipeline Workflow

```
Raw Data → Data Preprocessing → Feature Engineering → Train/Test Split → ML-Ready Data
```

### Container Architecture

The pipeline runs on five Docker containers:

- **Postgres**: Airflow metadata and workflow state storage
- **Scheduler**: Monitors and triggers data processing tasks
- **DAG Processor**: Parses and validates data pipeline definitions
- **API Server**: Serves the Airflow UI and REST API
- **Triggerer**: Handles deferred and sensor-based tasks

## ML Use Cases

This feature pipeline supports various ML scenarios:

- **Predictive Analytics**: Customer behavior prediction, sales forecasting
- **Classification Models**: User segmentation, fraud detection
- **Regression Analysis**: Revenue prediction, demand forecasting
- **Time Series**: Trend analysis, seasonal pattern detection

## Deployment

### Local Testing
```bash
# Start local development environment
astro dev start

# Test DAGs
astro dev pytest

# Stop environment
astro dev stop
```

### Production Deployment

For production deployment to Astronomer Cloud:

1. **Setup Astronomer Account**: Create account at [astronomer.io](https://www.astronomer.io)
2. **Deploy Pipeline**:
   ```bash
   astro deploy
   ```
3. **Configure Data Connections**: Set up BigQuery and other data source connections
4. **Monitor Production**: Use Astronomer Cloud monitoring and alerting

## Data Sources Integration

- **Google BigQuery**: Primary data warehouse for feature storage
- **External APIs**: Real-time data ingestion capabilities
- **CSV/Parquet Files**: Batch data processing support
- **Database Connections**: PostgreSQL, MySQL, and other RDBMS sources

## Next Steps

1. **Configure Data Sources**: Update `airflow_settings.yaml` with your data connections
2. **Customize Features**: Modify SQL files to match your specific ML requirements
3. **Add Validation**: Implement data quality checks and monitoring
4. **Scale Pipeline**: Configure resource allocation for large datasets

---

**Maintained by**: Data Science Team
**Documentation**: [Internal ML Pipeline Docs](link-to-docs)
**Support**: Contact data engineering team for pipeline issues
