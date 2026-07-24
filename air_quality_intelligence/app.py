"""Standalone entrypoint for the European Air Quality Intelligence app."""

import streamlit as st

from air_quality_intelligence.ui import render_dashboard
from portfolio_pages.design import inject_design_system


st.set_page_config(page_title="European Air Quality Intelligence", page_icon="◎", layout="wide")
inject_design_system()
render_dashboard()
