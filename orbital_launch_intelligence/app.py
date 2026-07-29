"""Standalone entrypoint for Orbital Launch Reliability Intelligence."""

import streamlit as st

from orbital_launch_intelligence.ui import render_dashboard
from portfolio_pages.design import inject_design_system


st.set_page_config(page_title="Orbital Launch Intelligence", page_icon="◈", layout="wide")
inject_design_system()
render_dashboard()
