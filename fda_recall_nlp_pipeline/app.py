"""Standalone Streamlit entrypoint."""

import streamlit as st

from fda_recall_nlp_pipeline.ui import render_dashboard


st.set_page_config(page_title="FDA Recall Triage Pipeline", page_icon="△", layout="wide")
render_dashboard()
