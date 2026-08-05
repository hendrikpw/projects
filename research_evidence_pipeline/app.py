"""Standalone Streamlit entrypoint for the research evidence project."""

from __future__ import annotations

import streamlit as st

from portfolio_pages.design import inject_design_system
from research_evidence_pipeline.ui import render_dashboard


st.set_page_config(
    page_title="Research Evidence Pipeline",
    page_icon="⌁",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_design_system()
render_dashboard()

