"""Standalone Streamlit entrypoint for Energy Transition Intelligence."""

import streamlit as st

from energy_transition_intelligence.ui import render_dashboard
from portfolio_pages.design import inject_design_system


st.set_page_config(
    page_title="Energy Transition Intelligence",
    page_icon="◌",
    layout="wide",
)
inject_design_system()
render_dashboard()
