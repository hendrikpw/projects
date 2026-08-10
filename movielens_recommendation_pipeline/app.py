"""Standalone Streamlit entrypoint."""

import streamlit as st

from movielens_recommendation_pipeline.ui import render_dashboard


st.set_page_config(page_title="MovieLens Recommendation Pipeline", page_icon="◉", layout="wide")
render_dashboard()
