# ML Feature Pipeline with Airflow Astronomer

## Overview

This project uses **Apache Airflow with Astronomer** to create a robust data aggregation and feature engineering pipeline for machine learning use cases. The pipeline transforms raw data into meaningful features that can be consumed by ML models.


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
