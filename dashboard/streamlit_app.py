"""
Data Science Demo Dashboard

A comprehensive dashboard showcasing A/B testing, machine learning models,
data preprocessing, and generative AI capabilities.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import warnings

# Import dashboard modules
from dashboard.config.settings import PAGE_CONFIG, DASHBOARD_INFO
from dashboard.utils.styling import apply_main_styling, render_dashboard_header, render_creator_info
from dashboard.tabs.abtest.main import render_abtest_tab
from dashboard.tabs.ml_models.main import render_ml_models_tab
from dashboard.tabs.ml_inference.main import render_ml_inference_tab
from dashboard.tabs.preprocessing.main import render_preprocessing_tab
from dashboard.tabs.genai.main import render_genai_tab
from dashboard.tabs.sql.main import render_sql_tab
from dashboard.tabs.python_tips.main import render_python_tips_tab
from dashboard.tabs.system_design.main import render_system_design_tab

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    """Main application entry point."""
    # Configure Streamlit page
    st.set_page_config(**PAGE_CONFIG)
    
    # Apply styling
    apply_main_styling()
    
    # Render header and creator info
    render_dashboard_header()
    render_creator_info()
    
    # Main navigation tabs
    main_tabs = st.tabs(["A/B TEST", "ML MODEL EXPERIMENT", "ML MODEL INFERENCE", "DATA PRE-PROCESSING", "GENAI", "SQL", "PYTHON TIPS", "SYSTEM DESIGN"])
    
    with main_tabs[0]:  # A/B Test
        render_abtest_tab()
    
    with main_tabs[1]:  # ML Model Experiment
        render_ml_models_tab()
    
    with main_tabs[2]:  # ML Model Inference
        render_ml_inference_tab()
        
    with main_tabs[3]:  # Data Pre-processing
        render_preprocessing_tab()
    
    with main_tabs[4]:  # GenAI
        render_genai_tab()
    
    with main_tabs[5]:  # SQL
        render_sql_tab()
    
    with main_tabs[6]:  # Python Tips
        render_python_tips_tab()
    
    with main_tabs[7]:  # System Design
        render_system_design_tab()

if __name__ == "__main__":
    main()