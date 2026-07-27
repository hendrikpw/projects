"""Standalone Streamlit entrypoint for Global Seismic Activity Intelligence."""

import streamlit as st

from earthquake_intelligence.ui import render_dashboard
from portfolio_pages.design import inject_design_system


st.set_page_config(
    page_title="Global Seismic Activity Intelligence",
    page_icon="◉",
    layout="wide",
)
inject_design_system()
render_dashboard()
