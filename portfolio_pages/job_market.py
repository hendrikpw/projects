"""Hosted wrapper for the existing Job Market Analytics project."""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_URL = "https://www.arbeitnow.com/api/job-board-api"
SKILLS = ("Python", "SQL", "Power BI", "AWS", "Azure", "Tableau", "Pandas", "Spark")


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_jobs() -> pd.DataFrame:
    response = requests.get(API_URL, timeout=15)
    response.raise_for_status()
    frame = pd.DataFrame(response.json().get("data", []))
    if frame.empty:
        return frame
    for column in ("title", "company_name", "location", "description"):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    if "remote" not in frame:
        frame["remote"] = False
    frame["remote"] = frame["remote"].astype(str).str.lower().isin({"true", "1", "yes"})
    return frame


def _skill_counts(descriptions: pd.Series) -> pd.DataFrame:
    text = " ".join(descriptions.astype(str)).lower()
    rows = [
        {"Skill": skill, "Mentions": len(re.findall(rf"\b{re.escape(skill.lower())}\b", text))}
        for skill in SKILLS
    ]
    return pd.DataFrame(rows).query("Mentions > 0").sort_values("Mentions", ascending=False)


def render_job_market() -> None:
    st.markdown('<div class="eyebrow">Live labor-market analytics</div>', unsafe_allow_html=True)
    st.title("Job Market Analytics")
    st.caption("Current listings from the public Arbeitnow Job Board API.")
    with st.spinner("Loading current job listings…"):
        try:
            jobs = _fetch_jobs()
        except (requests.RequestException, ValueError) as exc:
            st.error("The live job feed is temporarily unavailable.")
            st.caption(f"Technical detail: {exc}")
            return
    if jobs.empty:
        st.info("The API returned no listings. Try again later.")
        return
    search = st.text_input("Search jobs", placeholder="e.g. data, Python, analyst")
    filtered = jobs.copy()
    if search:
        mask = filtered[["title", "company_name", "location", "description"]].apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        filtered = filtered[mask]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Listings", f"{len(filtered):,}")
    k2.metric("Companies", f"{filtered['company_name'].nunique():,}")
    k3.metric("Locations", f"{filtered['location'].nunique():,}")
    k4.metric("Remote share", f"{filtered['remote'].mean():.0%}" if len(filtered) else "—")
    if filtered.empty:
        st.warning("No jobs match the current search.")
        return
    left, right = st.columns(2)
    with left:
        locations = filtered["location"].replace("", "Not specified").value_counts().head(10).sort_values()
        fig = px.bar(x=locations.values, y=locations.index, orientation="h", title="Top locations",
                     labels={"x": "Listings", "y": ""}, color_discrete_sequence=["#5eead4"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        skills = _skill_counts(filtered["description"])
        if skills.empty:
            st.info("No tracked skills were found in this selection.")
        else:
            fig = px.bar(skills.sort_values("Mentions"), x="Mentions", y="Skill", orientation="h",
                         title="Tracked skill mentions", color="Mentions",
                         color_continuous_scale=["#60a5fa", "#5eead4"])
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    display = [c for c in ("title", "company_name", "location", "remote") if c in filtered]
    st.subheader("Listing explorer")
    st.dataframe(filtered[display], hide_index=True, use_container_width=True)
    st.caption("Source: Arbeitnow Job Board API. Availability and completeness depend on the provider.")
