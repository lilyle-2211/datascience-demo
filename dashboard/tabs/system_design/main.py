"""System Design tab main module."""

import streamlit as st
from .system_design_guide import render_system_design_guide

def render_system_design_tab():
    """Render the System Design tab."""
    render_system_design_guide()