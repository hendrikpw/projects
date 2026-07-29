"""Central entrypoint for the hosted Data Science Portfolio."""

from __future__ import annotations

import streamlit as st

from air_quality_intelligence.ui import render_dashboard as render_air_quality
from earthquake_intelligence.ui import render_dashboard as render_earthquakes
from energy_transition_intelligence.ui import render_dashboard as render_energy_transition
from eu_inflation_intelligence.ui import render_dashboard as render_eu_inflation
from nyc_collision_intelligence.ui import render_dashboard as render_nyc_collisions
from orbital_launch_intelligence.ui import render_dashboard as render_orbital_launches
from portfolio_pages.design import inject_design_system
from portfolio_pages.job_market import render_job_market


st.set_page_config(
    page_title="Hendrik's Data Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_design_system()


def render_home() -> None:
    st.markdown(
        """
        <section class="editorial-hero">
          <div class="brand-line">Hendrik / Data Portfolio</div>
          <div>
            <h1>Data,<br>made useful.</h1>
            <p>
              Interactive products that turn public data into clear decisions.
              Transparent methods, reproducible code and interfaces designed
              with restraint.
            </p>
          </div>
          <div class="hero-footer">
            <span>Data Science · Analytics · AI</span>
            <span>Stuttgart / 2026</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Selected work / 01–07</div>
          <h2>Seven live products.<br>One growing system.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="project-grid">
          <article class="project-card project-card--air">
            <div class="project-index">01 / ENVIRONMENTAL INTELLIGENCE</div>
            <div>
              <h3>European Air Quality</h3>
              <p>Compare cities, track the seven-day European AQI forecast and identify the pollutants driving risk.</p>
              <div class="project-meta">
                <span>Open-Meteo</span><span>CAMS</span><span>Plotly</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--jobs">
            <div class="project-index">02 / LABOR-MARKET ANALYTICS</div>
            <div>
              <h3>Job Market Analytics</h3>
              <p>Explore current job listings by company, location, remote status and in-demand skills.</p>
              <div class="project-meta">
                <span>Arbeitnow</span><span>NLP</span><span>Pandas</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--energy">
            <div class="project-index">03 / TRANSITION INTELLIGENCE</div>
            <div>
              <h3>Energy Transition</h3>
              <p>Benchmark European transition profiles, discover peer groups and simulate measurable improvements.</p>
              <div class="project-meta">
                <span>World Bank</span><span>K-means</span><span>PCA</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--collision">
            <div class="project-index">04 / GEOSPATIAL SAFETY</div>
            <div>
              <h3>NYC Collision Risk</h3>
              <p>Map collision concentrations, detect unusual days and compare reported factors by volume and severity.</p>
              <div class="project-meta">
                <span>NYC Open Data</span><span>Geoanalytics</span><span>MAD</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--seismic">
            <div class="project-index">05 / SEISMIC INTELLIGENCE</div>
            <div>
              <h3>Global Seismic Activity</h3>
              <p>Explore recent earthquakes through spatial sequences, focal depth, energy and magnitude-frequency behavior.</p>
              <div class="project-meta">
                <span>USGS</span><span>DBSCAN</span><span>GeoJSON</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--inflation">
            <div class="project-index">06 / HOUSEHOLD ECONOMICS</div>
            <div>
              <h3>EU Inflation & Household Basket</h3>
              <p>Compare harmonised European inflation and decompose a transparent estimate for your own spending mix.</p>
              <div class="project-meta">
                <span>Eurostat</span><span>HICP</span><span>Basket model</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--launch">
            <div class="project-index">07 / SPACEFLIGHT OPERATIONS</div>
            <div>
              <h3>Orbital Launch Reliability</h3>
              <p>Benchmark launch providers with uncertainty-aware reliability, cadence, concentration and a live mission board.</p>
              <div class="project-meta">
                <span>Launch Library 2</span><span>Wilson interval</span><span>HHI</span>
              </div>
            </div>
          </article>
        </div>
        <section class="statement-panel">
          <div class="section-kicker">Design principle</div>
          <h2>Evidence before aesthetics.<br><strong>Clarity before noise.</strong></h2>
        </section>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("### H / DATA LAB")
    st.caption("Interactive portfolio · 2026")
    page = st.radio(
        "Navigate",
        [
            "Portfolio home",
            "Air Quality Intelligence",
            "Job Market Analytics",
            "Energy Transition Intelligence",
            "NYC Collision Risk Intelligence",
            "Global Seismic Activity Intelligence",
            "EU Inflation & Household Basket",
            "Orbital Launch Reliability Intelligence",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Python / Streamlit / Public APIs")
    st.link_button(
        "View source",
        "https://github.com/hendrikpw/projects",
        width="stretch",
    )


if page == "Air Quality Intelligence":
    render_air_quality()
elif page == "Job Market Analytics":
    render_job_market()
elif page == "Energy Transition Intelligence":
    render_energy_transition()
elif page == "NYC Collision Risk Intelligence":
    render_nyc_collisions()
elif page == "Global Seismic Activity Intelligence":
    render_earthquakes()
elif page == "EU Inflation & Household Basket":
    render_eu_inflation()
elif page == "Orbital Launch Reliability Intelligence":
    render_orbital_launches()
else:
    render_home()
