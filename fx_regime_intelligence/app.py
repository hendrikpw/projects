"""Standalone entrypoint for FX Market Regime Intelligence."""

import streamlit as st

from fx_regime_intelligence.ui import render_dashboard
from portfolio_pages.design import inject_design_system


st.set_page_config(page_title="FX Market Regime Intelligence", page_icon="◈", layout="wide")
inject_design_system()
render_dashboard()
