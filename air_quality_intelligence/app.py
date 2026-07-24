"""Standalone entrypoint for the European Air Quality Intelligence app."""

import streamlit as st

from air_quality_intelligence.ui import render_dashboard


st.set_page_config(page_title="European Air Quality Intelligence", page_icon="◎", layout="wide")
render_dashboard()
