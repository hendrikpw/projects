"""Standalone entrypoint for EU Inflation & Household Basket Intelligence."""

import streamlit as st

from eu_inflation_intelligence.ui import render_dashboard
from portfolio_pages.design import inject_design_system


st.set_page_config(page_title="EU Inflation Intelligence", page_icon="◈", layout="wide")
inject_design_system()
render_dashboard()
