"""Dashboard configuration settings."""

import streamlit as st

# Page configuration
PAGE_CONFIG = {
    "page_title": "Data Science Demo Dashboard",
    "layout": "wide",
    "initial_sidebar_state": "collapsed"
}

# Dashboard metadata
DASHBOARD_INFO = {
    "title": "DATA SCIENCE DEMO DASHBOARD",
    "creator": "Lily Le",
    "version": "1.0.0"
}

# Default values for ML models
ML_DEFAULTS = {
    "sample_size": 10000,
    "test_size": 0.2,
    "random_state": 42
}

# A/B Test default parameters
AB_TEST_DEFAULTS = {
    "alpha": 0.05,
    "power": 0.80,
    "daily_users": 1000,
    "treatment_allocation": 0.50,
    "current_rate": 0.15,
    "mde_absolute": 0.02,
    "mde_relative": 10.0
}

# Styling configuration
COLORS = {
    "primary": "#ff8c00",
    "secondary": "#2E86C1",
    "gradient_start": "#2E86C1",
    "gradient_end": "#5DADE2",
    "light_gray": "#f8f9fa",
    "dark_gray": "#6c757d",
    "border": "#dee2e6"
}