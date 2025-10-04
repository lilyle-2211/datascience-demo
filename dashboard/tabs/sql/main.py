"""SQL tab main module."""

import streamlit as st
from .sql_guide import render_sql_guide

def render_sql_tab():
    """Render the SQL concepts tab."""
    render_sql_guide()