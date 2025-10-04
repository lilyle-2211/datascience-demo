"""Linear regression tab implementation."""

import streamlit as st
from .models import train_linear_regression
from .components import (
    display_metrics_comparison, 
    create_regression_plots, 
    display_business_context, 
    display_feature_importance_linear
)

def render_linear_regression_tab(X_train, X_test, y_train, y_test):
    """Render the Linear Regression LTV tab."""
    st.subheader("Linear Regression Model for LTV Prediction")
    
    with st.spinner("Training Linear Regression model..."):
        lr_model, scaler, lr_train_metrics, lr_test_metrics = train_linear_regression(
            X_train, X_test, y_train, y_test
        )
    
    # Display metrics
    display_metrics_comparison(lr_train_metrics, lr_test_metrics, "Linear Regression")
    
    # Business context
    display_business_context(lr_test_metrics, y_test)
    
    # Plots
    st.subheader("Model Evaluation Plots")
    fig = create_regression_plots(y_test, lr_test_metrics['predictions'], "Linear Regression")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (coefficients)
    feature_names = X_train.columns
    display_feature_importance_linear(lr_model, feature_names)