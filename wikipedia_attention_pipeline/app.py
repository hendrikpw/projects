"""Standalone Streamlit entrypoint."""

import streamlit as st

from wikipedia_attention_pipeline.ui import render_dashboard


st.set_page_config(page_title="Wikipedia Attention Pipeline", page_icon="⌁", layout="wide")
render_dashboard()
