"""Regression model implementations."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb

def train_linear_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Linear Regression model.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        
    Returns:
        tuple: (model, scaler, train_metrics, test_metrics)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_metrics = {
        'mse': mean_squared_error(y_train, y_pred_train),
        'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'mae': mean_absolute_error(y_train, y_pred_train),
        'mape': mean_absolute_percentage_error(y_train, y_pred_train),
        'r2': r2_score(y_train, y_pred_train),
        'predictions': y_pred_train
    }
    
    test_metrics = {
        'mse': mean_squared_error(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae': mean_absolute_error(y_test, y_pred_test),
        'mape': mean_absolute_percentage_error(y_test, y_pred_test),
        'r2': r2_score(y_test, y_pred_test),
        'predictions': y_pred_test
    }
    
    return model, scaler, train_metrics, test_metrics

def train_xgboost_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate XGBoost model.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        
    Returns:
        tuple: (model, train_metrics, test_metrics)
    """
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = {
        'mse': mean_squared_error(y_train, y_pred_train),
        'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'mae': mean_absolute_error(y_train, y_pred_train),
        'mape': mean_absolute_percentage_error(y_train, y_pred_train),
        'r2': r2_score(y_train, y_pred_train),
        'predictions': y_pred_train
    }
    
    test_metrics = {
        'mse': mean_squared_error(y_test, y_pred_test),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae': mean_absolute_error(y_test, y_pred_test),
        'mape': mean_absolute_percentage_error(y_test, y_pred_test),
        'r2': r2_score(y_test, y_pred_test),
        'predictions': y_pred_test
    }
    
    return model, train_metrics, test_metrics