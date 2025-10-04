"""Classification models main module."""

import streamlit as st
from .guide import render_classification_guide_tab

def render_classification_tab():
    """Render the classification models tab."""
    render_classification_guide_tab()