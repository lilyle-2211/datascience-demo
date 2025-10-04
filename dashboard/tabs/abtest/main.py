"""Main A/B testing tab module."""

import streamlit as st
from .components.goal_metrics import render_goal_metrics
from .components.sample_size_calculator import render_sample_size_calculator

def render_abtest_tab():
    """Render the complete A/B testing tab."""
    st.header("A/B Testing Framework")
    
    # A/B Test sub-tabs
    ab_tabs = st.tabs(["1. Set Goal Metrics", "2. Sample Size Calculator"])
    
    with ab_tabs[0]:
        render_goal_metrics()
        
    with ab_tabs[1]:
        render_sample_size_calculator()