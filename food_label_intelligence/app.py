"""Standalone entrypoint for Food Label & Product Choice Intelligence."""

import streamlit as st

from food_label_intelligence.ui import render_dashboard
from portfolio_pages.design import inject_design_system


st.set_page_config(page_title="Food Label Intelligence", page_icon="◈", layout="wide")
inject_design_system()
render_dashboard()
