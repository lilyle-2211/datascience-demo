"""Main ML models tab module."""

import streamlit as st
from .regression.main import render_regression_tab
from .classification.main import render_classification_tab  
from .ranking.main import render_ranking_tab

def render_ml_models_tab():
    """Render the complete ML models tab."""
    st.header("Machine Learning Models")
    
    # ML sub-tabs
    ml_tabs = st.tabs(["Classification", "Regression", "Ranking"])
    
    with ml_tabs[0]:  # Classification
        render_classification_tab()
    
    with ml_tabs[1]:  # Regression
        render_regression_tab()
        
    with ml_tabs[2]:  # Ranking
        render_ranking_tab()