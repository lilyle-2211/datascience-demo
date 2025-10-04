import streamlit as st
from pathlib import Path

def render_classification_guide_tab():
    """Render the Classification Guide tab."""
    
    # Read markdown content from the guide file (copied into the project for dashboard use)
    guide_path = Path(__file__).parent / "classification_guide.md"
    if guide_path.exists():
        with open(guide_path, "r", encoding="utf-8") as f:
            guide_md = f.read()
        st.markdown(guide_md)
    else:
        st.warning("Classification guide markdown not found.")
