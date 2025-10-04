"""XGBoost regression tab implementation."""

import streamlit as st
from .models import train_xgboost_model
from .components import (
    display_metrics_comparison, 
    create_regression_plots, 
    display_business_context, 
    display_feature_importance_xgboost
)

def render_xgboost_regression_tab(X_train, X_test, y_train, y_test):
    """Render the XGBoost LTV tab."""
    st.subheader("XGBoost Model for LTV Prediction")
    
    with st.spinner("Training XGBoost model..."):
        xgb_model, xgb_train_metrics, xgb_test_metrics = train_xgboost_model(
            X_train, X_test, y_train, y_test
        )
    
    # Display metrics
    display_metrics_comparison(xgb_train_metrics, xgb_test_metrics, "XGBoost")
    
    # Business context
    display_business_context(xgb_test_metrics, y_test)
    
    # Plots
    st.subheader("Model Evaluation Plots")
    fig = create_regression_plots(y_test, xgb_test_metrics['predictions'], "XGBoost")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    feature_names = X_train.columns
    display_feature_importance_xgboost(xgb_model, feature_names)