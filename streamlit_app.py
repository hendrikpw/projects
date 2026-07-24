"""Central entrypoint for the hosted Data Science Portfolio."""

from __future__ import annotations

import streamlit as st

from air_quality_intelligence.ui import render_dashboard as render_air_quality
from portfolio_pages.job_market import render_job_market


st.set_page_config(
    page_title="Hendrik's Data Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --bg: #07111f;
  --panel: rgba(16, 30, 49, .82);
  --text: #f3f7fb;
  --muted: #9cb0c8;
  --accent: #5eead4;
  --border: rgba(148, 163, 184, .18);
}
html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
h1, h2, h3 { font-family: "Space Grotesk", sans-serif !important; letter-spacing: -.035em; }
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 15% -10%, rgba(37, 99, 235, .20), transparent 34rem),
    radial-gradient(circle at 85% 0%, rgba(20, 184, 166, .16), transparent 30rem),
    var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: rgba(7, 17, 31, .88);
  border-right: 1px solid var(--border);
}
.block-container { max-width: 1380px; padding-top: 2rem; padding-bottom: 4rem; }
.portfolio-hero {
  padding: 2.2rem 2.4rem;
  border: 1px solid var(--border);
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(30, 58, 90, .90), rgba(10, 24, 42, .88));
  box-shadow: 0 22px 70px rgba(0, 0, 0, .28);
  margin-bottom: 1.4rem;
}
.eyebrow { color: var(--accent); text-transform: uppercase; letter-spacing: .14em; font-weight: 700; font-size: .76rem; }
.portfolio-hero h1 { font-size: clamp(2.4rem, 5vw, 4.6rem); margin: .45rem 0 .65rem; line-height: .98; }
.portfolio-hero p { color: var(--muted); max-width: 740px; font-size: 1.05rem; margin: 0; }
.project-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin-top: 1rem; }
.project-card {
  padding: 1.35rem;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--panel);
  min-height: 190px;
}
.project-card h3 { margin: .35rem 0 .5rem; }
.project-card p { color: var(--muted); }
.tag { display: inline-block; color: #b7fff4; background: rgba(45, 212, 191, .11); border: 1px solid rgba(45, 212, 191, .22); border-radius: 999px; padding: .22rem .55rem; font-size: .72rem; margin: .1rem; }
@media (max-width: 720px) {
  .project-grid { grid-template-columns: 1fr; }
  .portfolio-hero { padding: 1.5rem; }
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_home() -> None:
    st.markdown(
        """
        <section class="portfolio-hero">
          <div class="eyebrow">Data Science · Analytics · AI</div>
          <h1>Hendrik's<br>Data Lab</h1>
          <p>
            A growing collection of interactive, source-backed data products.
            Each project turns public data into a focused decision tool—with
            transparent methods, reproducible code and a polished user experience.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("## Explore the portfolio")
    st.markdown(
        """
        <div class="project-grid">
          <article class="project-card">
            <div class="eyebrow">Live environmental analytics</div>
            <h3>European Air Quality Intelligence</h3>
            <p>Compare cities, track the seven-day European AQI forecast and identify the pollutants driving risk.</p>
            <span class="tag">Open-Meteo</span><span class="tag">CAMS</span>
            <span class="tag">Plotly</span><span class="tag">Streamlit</span>
          </article>
          <article class="project-card">
            <div class="eyebrow">Labor-market analytics</div>
            <h3>Job Market Analytics</h3>
            <p>Explore current job listings by company, location, remote status and in-demand skills.</p>
            <span class="tag">Arbeitnow API</span><span class="tag">NLP</span>
            <span class="tag">Pandas</span><span class="tag">Plotly</span>
          </article>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("### ◈ DATA LAB")
    st.caption("Interactive portfolio")
    page = st.radio(
        "Navigate",
        ["Portfolio home", "Air Quality Intelligence", "Job Market Analytics"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Python · Streamlit · Public APIs")
    st.link_button("View source on GitHub", "https://github.com/hendrikpw/projects", use_container_width=True)


if page == "Air Quality Intelligence":
    render_air_quality()
elif page == "Job Market Analytics":
    render_job_market()
else:
    render_home()
