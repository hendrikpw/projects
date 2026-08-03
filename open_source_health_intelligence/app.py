"""Standalone entrypoint for Open Source Repository Health Intelligence."""

from __future__ import annotations

import streamlit as st

from open_source_health_intelligence.ui import render_dashboard
from portfolio_pages.design import inject_design_system


st.set_page_config(
    page_title="Open Source Repository Health Intelligence",
    page_icon="⌁",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_design_system()
render_dashboard()
