"""Central entrypoint for the hosted Data Science Portfolio."""

from __future__ import annotations

import streamlit as st

from air_quality_intelligence.ui import render_dashboard as render_air_quality
from biodiversity_observation_intelligence.ui import render_dashboard as render_biodiversity
from clinical_trial_ops_pipeline.ui import render_dashboard as render_clinical_trials
from cyber_vulnerability_intelligence.ui import render_dashboard as render_cyber_vulnerabilities
from earthquake_intelligence.ui import render_dashboard as render_earthquakes
from energy_transition_intelligence.ui import render_dashboard as render_energy_transition
from eu_inflation_intelligence.ui import render_dashboard as render_eu_inflation
from fda_recall_nlp_pipeline.ui import render_dashboard as render_fda_recalls
from food_label_intelligence.ui import render_dashboard as render_food_labels
from fx_regime_intelligence.ui import render_dashboard as render_fx_regimes
from london_bike_rebalancing_intelligence.ui import render_dashboard as render_london_cycles
from movielens_recommendation_pipeline.ui import render_dashboard as render_movielens_recommendations
from nyc_collision_intelligence.ui import render_dashboard as render_nyc_collisions
from nyc_311_resolution_pipeline.ui import render_dashboard as render_nyc_311_resolution
from open_source_health_intelligence.ui import render_dashboard as render_open_source_health
from orbital_launch_intelligence.ui import render_dashboard as render_orbital_launches
from predictive_maintenance_pipeline.ui import render_dashboard as render_predictive_maintenance
from portfolio_pages.design import inject_design_system
from portfolio_pages.job_market import render_job_market
from research_evidence_pipeline.ui import render_dashboard as render_research_evidence
from wikipedia_attention_pipeline.ui import render_dashboard as render_wikipedia_attention


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
          <div class="section-kicker">Selected work / 01–20</div>
          <h2>Twenty live products.<br>One growing system.</h2>
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
          <article class="project-card project-card--food">
            <div class="project-index">08 / CONSUMER PRODUCT INTELLIGENCE</div>
            <div>
              <h3>Food Label & Product Choice</h3>
              <p>Compare packaged-food labels, audit missing data and discover similar nutrition profiles with visible priorities.</p>
              <div class="project-meta">
                <span>Open Food Facts</span><span>Percentiles</span><span>Similarity</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--cyber">
            <div class="project-index">09 / CYBERSECURITY OPERATIONS</div>
            <div>
              <h3>Cyber Vulnerability Prioritization</h3>
              <p>Order confirmed exploited vulnerabilities with current EPSS, deadlines, ransomware signals and delivery capacity.</p>
              <div class="project-meta">
                <span>CISA KEV</span><span>FIRST EPSS</span><span>TF-IDF</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--fx">
            <div class="project-index">10 / FINANCIAL MARKET INTELLIGENCE</div>
            <div>
              <h3>FX Market Regime Intelligence</h3>
              <p>Monitor euro reference rates, detect unusual sessions and translate volatility into transparent exposure scenarios.</p>
              <div class="project-meta">
                <span>ECB</span><span>Isolation Forest</span><span>Risk analytics</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--biodiversity">
            <div class="project-index">11 / BIODIVERSITY DATA INTELLIGENCE</div>
            <div>
              <h3>Biodiversity Observation Intelligence</h3>
              <p>Compare European species records, reveal observation concentration and audit the quality behind every spatial pattern.</p>
              <div class="project-meta">
                <span>GBIF</span><span>DBSCAN</span><span>Data quality</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--health">
            <div class="project-index">12 / SOFTWARE DELIVERY INTELLIGENCE</div>
            <div>
              <h3>Open Source Repository Health</h3>
              <p>Audit delivery flow, censored resolution times, backlog age, contributor concentration and release rhythm.</p>
              <div class="project-meta">
                <span>GitHub REST</span><span>Kaplan–Meier</span><span>HHI</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--cycle">
            <div class="project-index">13 / URBAN MOBILITY OPERATIONS</div>
            <div>
              <h3>London Cycle Rebalancing</h3>
              <p>Map live station pressure, detect spatial shortages and turn surplus bikes into a transparent move plan.</p>
              <div class="project-meta">
                <span>TfL BikePoint</span><span>DBSCAN</span><span>Operations</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--evidence">
            <div class="project-index">14 / DATA + AI ENGINEERING</div>
            <div>
              <h3>Research Evidence Pipeline</h3>
              <p>Turn live scientific metadata into a contracted data product and an evaluated, citation-bound evidence engine.</p>
              <div class="project-meta">
                <span>Europe PMC</span><span>Data contracts</span><span>Semantic retrieval</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--trial">
            <div class="project-index">15 / DATA + AI ENGINEERING</div>
            <div>
              <h3>Clinical Trial Operations ML</h3>
              <p>Contract public trial snapshots, validate an explainable discontinuation model and monitor calibration and drift.</p>
              <div class="project-meta">
                <span>ClinicalTrials.gov</span><span>Data contracts</span><span>Model monitoring</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--fda">
            <div class="project-index">16 / DATA + AI ENGINEERING</div>
            <div>
              <h3>FDA Recall Triage Pipeline</h3>
              <p>Contract three enforcement streams, audit quarantine and evaluate confidence-aware NLP with explicit deferral.</p>
              <div class="project-meta">
                <span>openFDA</span><span>Data contracts</span><span>Selective NLP</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--wiki">
            <div class="project-index">17 / DATA + AI ENGINEERING</div>
            <div>
              <h3>Wikipedia Attention Forecast</h3>
              <p>Replay delayed daily signals through watermarks, then backtest forecasts, uncertainty and anomaly review.</p>
              <div class="project-meta">
                <span>Wikimedia</span><span>Event time</span><span>Conformal forecast</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--311">
            <div class="project-index">18 / DATA + AI ENGINEERING</div>
            <div>
              <h3>NYC 311 Resolution Operations</h3>
              <p>Contract mature service requests, backtest resolution quantiles and expose calibrated planning uncertainty.</p>
              <div class="project-meta">
                <span>NYC Open Data</span><span>Data contracts</span><span>Quantile ML</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--recs">
            <div class="project-index">19 / DATA + AI ENGINEERING</div>
            <div>
              <h3>MovieLens Recommendation Serving</h3>
              <p>Contract interactions, hide future preference signals and evaluate personalized retrieval against popularity.</p>
              <div class="project-meta">
                <span>GroupLens</span><span>Temporal holdout</span><span>Latent factors</span>
              </div>
            </div>
          </article>
          <article class="project-card project-card--maintenance">
            <div class="project-index">20 / DATA + AI ENGINEERING</div>
            <div>
              <h3>Predictive Maintenance Decision</h3>
              <p>Contract machine cycles, calibrate rare-failure probability and convert intervention costs into an auditable alert policy.</p>
              <div class="project-meta">
                <span>UCI AI4I</span><span>Calibration</span><span>Cost-sensitive ML</span>
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
            "Food Label & Product Choice Intelligence",
            "Cyber Vulnerability Prioritization Intelligence",
            "FX Market Regime Intelligence",
            "Biodiversity Observation Intelligence",
            "Open Source Repository Health Intelligence",
            "London Cycle Rebalancing Intelligence",
            "Research Evidence Pipeline",
            "Clinical Trial Operations ML Pipeline",
            "FDA Recall Triage Pipeline",
            "Wikipedia Attention Forecast Pipeline",
            "NYC 311 Resolution Operations Pipeline",
            "MovieLens Recommendation Serving Pipeline",
            "Predictive Maintenance Decision Pipeline",
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
elif page == "Biodiversity Observation Intelligence":
    render_biodiversity()
elif page == "Open Source Repository Health Intelligence":
    render_open_source_health()
elif page == "London Cycle Rebalancing Intelligence":
    render_london_cycles()
elif page == "Research Evidence Pipeline":
    render_research_evidence()
elif page == "Clinical Trial Operations ML Pipeline":
    render_clinical_trials()
elif page == "FDA Recall Triage Pipeline":
    render_fda_recalls()
elif page == "Wikipedia Attention Forecast Pipeline":
    render_wikipedia_attention()
elif page == "NYC 311 Resolution Operations Pipeline":
    render_nyc_311_resolution()
elif page == "MovieLens Recommendation Serving Pipeline":
    render_movielens_recommendations()
elif page == "Predictive Maintenance Decision Pipeline":
    render_predictive_maintenance()
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
elif page == "Food Label & Product Choice Intelligence":
    render_food_labels()
elif page == "Cyber Vulnerability Prioritization Intelligence":
    render_cyber_vulnerabilities()
elif page == "FX Market Regime Intelligence":
    render_fx_regimes()
else:
    render_home()
