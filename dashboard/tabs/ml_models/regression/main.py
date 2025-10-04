"""Main regression tab module."""

import streamlit as st
from .linear_regression import render_linear_regression_tab
from .xgboost_regression import render_xgboost_regression_tab
from .evaluation_guide import render_evaluation_guide_tab
from ....utils.data_utils import prepare_ml_data
from ....config.settings import ML_DEFAULTS

def render_regression_tab():
    """Render the complete regression tab."""
    st.subheader("Regression Models - Customer Lifetime Value Prediction")
    
    # Regression sub-tabs
    regression_tabs = st.tabs(["Linear Regression LTV", "XGBoost LTV", "Model Evaluation Guide"])
    
    # Load and prepare data (cached for performance)
    @st.cache_data
    def load_data():
        return prepare_ml_data(
            sample_size=ML_DEFAULTS["sample_size"],
            test_size=ML_DEFAULTS["test_size"],
            random_state=ML_DEFAULTS["random_state"]
        )
    
    df, X_train, X_test, y_train, y_test = load_data()
    
    with regression_tabs[0]:  # Linear Regression LTV
        render_linear_regression_tab(X_train, X_test, y_train, y_test)
    
    with regression_tabs[1]:  # XGBoost LTV
        render_xgboost_regression_tab(X_train, X_test, y_train, y_test)
        
    with regression_tabs[2]:  # Evaluation Guide
        render_evaluation_guide_tab()