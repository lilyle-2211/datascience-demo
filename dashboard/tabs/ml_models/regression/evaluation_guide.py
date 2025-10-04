"""Model evaluation guide tab implementation."""

import streamlit as st
from pathlib import Path

def render_evaluation_guide_tab():
    """Render the Regression Guide tab."""
    st.subheader("Regression Evaluation Metrics")
    guide_path = Path(__file__).parent / "regression_guide.md"
    if guide_path.exists():
        with open(guide_path, "r", encoding="utf-8") as f:
            guide_md = f.read()
        st.markdown(guide_md)
    else:
        st.warning("Regression guide markdown not found.")