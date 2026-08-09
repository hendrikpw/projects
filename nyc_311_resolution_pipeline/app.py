"""Standalone Streamlit entrypoint."""

import streamlit as st

from nyc_311_resolution_pipeline.ui import render_dashboard


st.set_page_config(page_title="NYC 311 Resolution Pipeline", page_icon="⌂", layout="wide")
render_dashboard()
