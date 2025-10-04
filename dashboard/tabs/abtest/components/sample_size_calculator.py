"""Sample size calculator component for A/B testing."""

import streamlit as st
from ....utils.styling import create_section_header
from ..calculators.ab_test_calculator import (
    render_ab_test_configuration,
    render_ab_test_traffic_timeline, 
    render_ab_test_metric_parameters,
    render_ab_test_results
)

def render_sample_size_calculator():
    """Render the sample size calculator component."""
    # Initialize session state for calculator
    if "ab_params" not in st.session_state:
        st.session_state.ab_params = {}
        
    # Section 1: Simple A/B Test (1 Control vs 1 Treatment)
    st.markdown(
        create_section_header("1 Control vs 1 Treatment"),
        unsafe_allow_html=True,
    )

    # Render A/B test calculator in 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        render_ab_test_configuration()
    with col2:
        render_ab_test_traffic_timeline()
    with col3:
        render_ab_test_metric_parameters()

    render_ab_test_results()