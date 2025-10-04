"""Python Tips tab main module."""

import streamlit as st
from .python_guide import render_python_guide

def render_python_tips_tab():
    """Render the Python tips tab."""
    render_python_guide()