"""Standalone entrypoint for Biodiversity Observation Intelligence."""

import streamlit as st

from biodiversity_observation_intelligence.ui import render_dashboard
from portfolio_pages.design import inject_design_system


st.set_page_config(page_title="Biodiversity Observation Intelligence", page_icon="◈", layout="wide")
inject_design_system()
render_dashboard()
