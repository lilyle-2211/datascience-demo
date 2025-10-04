"""Styling utilities for the dashboard."""

import streamlit as st
from ..config.settings import COLORS

def apply_main_styling():
    """Apply main dashboard styling."""
    tab_styles = f"""
    <style>
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        justify-content: center;
        margin: 20px 0;
        background: transparent;
    }}

    .stTabs [data-baseweb="tab-list"] button {{
        height: 60px;
        padding: 12px 24px;
        background: {COLORS['light_gray']};
        color: {COLORS['dark_gray']};
        border: 2px solid {COLORS['border']};
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        min-width: 180px;
    }}

    .stTabs [data-baseweb="tab-list"] button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        background: #e9ecef;
        border-color: #adb5bd;
    }}

    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        background: {COLORS['primary']} !important;
        color: white !important;
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.4) !important;
        transform: translateY(-2px);
    }}

    .stTabs [data-baseweb="tab-list"] button p {{
        margin: 0;
        font-size: 18px;
        font-weight: bold;
    }}
    
    .section-header {{
        background: linear-gradient(90deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .section-divider {{
        margin: 40px 0;
        border-top: 2px solid #E8F4FD;
    }}
    
    .metric-card {{
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {COLORS['secondary']};
        margin: 0.5rem 0;
    }}
    
    .warning-card {{
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }}
    
    .success-card {{
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }}
    </style>
    """
    st.markdown(tab_styles, unsafe_allow_html=True)

def render_dashboard_header():
    """Render the main dashboard header."""
    st.markdown(
        f"""
        <h1 style='text-align: center; margin-top: 0; margin-bottom: 2rem;'>DATA SCIENCE DEMO DASHBOARD</h1>
        """,
        unsafe_allow_html=True,
    )

def render_creator_info():
    """Render creator information in top right."""
    col_spacer, col_creator = st.columns([6, 2])
    with col_creator:
        st.markdown(
            "<div style='text-align: right; font-weight: 500;'>Creator: Lily Le</div>",
            unsafe_allow_html=True,
        )

def create_section_header(title: str):
    """Create a styled section header."""
    return f'<div class="section-header">{title}</div>'