"""Standalone Streamlit entrypoint."""

import streamlit as st

from clinical_trial_ops_pipeline.ui import render_dashboard


st.set_page_config(page_title="Clinical Trial Operations ML", page_icon="◫", layout="wide")
render_dashboard()
