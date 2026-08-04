"""Standalone Streamlit entrypoint for London Cycle Rebalancing Intelligence."""

from __future__ import annotations

import streamlit as st

from london_bike_rebalancing_intelligence.ui import render_dashboard
from portfolio_pages.design import inject_design_system


st.set_page_config(
    page_title="London Cycle Rebalancing Intelligence",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_design_system()
render_dashboard()
