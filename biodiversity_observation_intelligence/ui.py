"""Streamlit interface for Biodiversity Observation Intelligence."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from biodiversity_observation_intelligence.src.analytics import (
    add_quality_features,
    cluster_summary,
    facet_table,
    grid_overlap,
    quality_report,
    spatial_clusters,
    summary_metrics,
)
from biodiversity_observation_intelligence.src.data import (
    CITATION_URL,
    OCCURRENCE_DOCS_URL,
    SPECIES,
    TERMS_URL,
    load_data,
)


PALETTE = ["#e5484d", "#fcfcfd", "#a7abb2", "#747982", "#4a4f57", "#292d33"]
MONTHS = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def _style_figure(figure: go.Figure, height: int = 470) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=64, b=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color="rgba(252,252,253,.72)"),
        title_font=dict(size=17, color="#fcfcfd"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor="#171a20", font_color="#fcfcfd", bordercolor="#5c6068"),
        colorway=PALETTE,
    )
    figure.update_xaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(252,252,253,.08)", zeroline=False)
    return figure


@st.cache_data(ttl=43_200, show_spinner=False)
def _cached_data(names: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return load_data(list(names))


def render_dashboard() -> None:
    """Render the complete GBIF observation and data-quality mini-product."""
    st.markdown(
        """
        <section class="page-hero">
          <div class="eyebrow">11 / Biodiversity data intelligence</div>
          <h1>Biodiversity<br>Observation Intelligence.</h1>
          <p>
            Explore when and where European species are recorded, identify dense
            observation zones and audit the quality behind every visible pattern.
          </p>
          <div class="source-line">GBIF · Taxonomy · Darwin Core occurrences</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    selected = st.multiselect(
        "Species comparison",
        list(SPECIES),
        default=["Erinaceus europaeus", "Lutra lutra"],
        max_selections=3,
        format_func=lambda name: f"{SPECIES[name]} · {name}",
    )
    if not selected:
        st.info("Choose one to three species to load the observation intelligence workspace.")
        return

    with st.spinner("Resolving taxonomy and loading GBIF observations…"):
        raw, facets, metadata = _cached_data(tuple(selected))
    data = add_quality_features(raw)
    if metadata["mode"] == "demo":
        st.warning(
            "GBIF is temporarily unavailable. The interface is using deterministic "
            "synthetic records and counts; they are not real biodiversity evidence."
        )
    else:
        st.success(
            f"Live GBIF data · {metadata['indexed_records']:,} indexed coordinate records · "
            f"{metadata['sample_records']:,} records in the spatial audit sample"
        )

    taxonomy = pd.DataFrame(metadata["species"])
    metrics = summary_metrics(data, metadata)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Indexed records", f"{metrics['indexed_records']:,}")
    k2.metric("Audit sample", f"{metrics['sample_records']:,}")
    k3.metric("Countries in sample", metrics["countries"])
    k4.metric("Datasets in sample", metrics["datasets"])
    k5.metric("Issue-free sample", f"{metrics['issue_free_share']:.1f}%")
    st.caption(
        "Indexed records are GBIF search counts for 2018–2026 with coordinates in Europe. "
        "The map and record audit use a bounded API sample. Neither measure is population "
        "size, abundance, occupancy or a representative survey estimate."
    )

    with st.expander("Taxonomic resolution and query scope"):
        view = taxonomy[
            ["query_name", "accepted_name", "taxon_key", "rank", "status", "confidence", "match_type", "family", "class", "indexed_records", "sample_records"]
        ].copy()
        view.columns = ["Query", "Accepted name", "GBIF taxon key", "Rank", "Status", "Confidence", "Match type", "Family", "Class", "Indexed records", "Sample records"]
        st.dataframe(view, width="stretch", hide_index=True)

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Observation geography</div>
          <h2>Map records.<br>Expose sampling concentration.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns([0.7, 0.7])
    with m1:
        radius = st.slider("Dense-zone radius", 25, 250, 90, 5, format="%d km")
    with m2:
        minimum = st.slider("Minimum nearby sample records", 3, 25, 8)
    clustered = spatial_clusters(data, radius, minimum)
    clusters = cluster_summary(clustered)
    dense_share = float(clustered["cluster"].ge(0).mean() * 100) if not clustered.empty else 0
    cluster_count = len(clusters)
    c1, c2 = st.columns(2)
    c1.metric("Dense sample zones", cluster_count)
    c2.metric("Sample inside zones", f"{dense_share:.1f}%")

    map_figure = px.scatter_mapbox(
        clustered,
        lat="latitude",
        lon="longitude",
        color="query_name",
        size="quality_score",
        hover_name="scientific_name",
        hover_data={
            "country": True,
            "event_date": True,
            "basis_of_record": True,
            "coordinate_uncertainty_m": ":,.0f",
            "cluster_label": True,
            "latitude": False,
            "longitude": False,
            "quality_score": True,
        },
        zoom=2.35,
        center={"lat": 51, "lon": 11},
        opacity=0.68,
        title="Bounded georeferenced occurrence sample",
        labels={"query_name": "Species"},
        color_discrete_sequence=PALETTE,
    )
    map_figure.update_layout(mapbox_style="carto-darkmatter")
    st.plotly_chart(_style_figure(map_figure, 670), width="stretch")
    st.caption(
        "DBSCAN uses great-circle distance on the API sample. Dense zones describe where "
        "sample records concentrate—not habitat, species range or ecological hotspots."
    )

    overlap = grid_overlap(data, 1.0)
    if not overlap.empty:
        overlap_view = overlap.copy()
        overlap_view.columns = ["Species A", "Species B", "Shared 1° cells", "Union cells", "Jaccard overlap"]
        st.markdown("#### Sample grid overlap")
        st.dataframe(overlap_view.style.format({"Jaccard overlap": "{:.1f}%"}), width="stretch", hide_index=True)

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Reporting patterns</div>
          <h2>Use complete counts.<br>Interpret effort, not abundance.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    years = facet_table(facets, "YEAR")
    years["year"] = pd.to_numeric(years["value"], errors="coerce")
    months = facet_table(facets, "MONTH")
    months["month_number"] = pd.to_numeric(months["value"], errors="coerce")
    months["month"] = months["month_number"].map(MONTHS)
    left, right = st.columns(2)
    with left:
        year_chart = px.line(
            years.sort_values("year"),
            x="year",
            y="count",
            color="query_name",
            markers=True,
            title="All indexed coordinate records by year",
            labels={"year": "", "count": "Records", "query_name": "Species"},
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(_style_figure(year_chart, 500), width="stretch")
    with right:
        month_chart = px.line(
            months.sort_values("month_number"),
            x="month_number",
            y="share",
            color="query_name",
            markers=True,
            title="Within-species monthly reporting profile",
            labels={"month_number": "Month", "share": "Share of dated records · %", "query_name": "Species"},
            color_discrete_sequence=PALETTE,
        )
        month_chart.update_xaxes(tickmode="array", tickvals=list(MONTHS), ticktext=list(MONTHS.values()))
        st.plotly_chart(_style_figure(month_chart, 500), width="stretch")

    countries = facet_table(facets, "COUNTRY")
    top_country_names = countries.groupby("value")["count"].sum().nlargest(12).index
    countries = countries[countries["value"].isin(top_country_names)]
    bases = facet_table(facets, "BASIS_OF_RECORD")
    country_col, basis_col = st.columns([1.05, 0.95])
    with country_col:
        country_chart = px.bar(
            countries,
            x="count",
            y="value",
            color="query_name",
            orientation="h",
            barmode="group",
            title="Largest country reporting volumes",
            labels={"count": "Indexed records", "value": "", "query_name": "Species"},
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(_style_figure(country_chart, 540), width="stretch")
    with basis_col:
        basis_chart = px.bar(
            bases,
            x="share",
            y="value",
            color="query_name",
            orientation="h",
            barmode="group",
            title="Basis-of-record composition",
            labels={"share": "Within-species share · %", "value": "", "query_name": "Species"},
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(_style_figure(basis_chart, 540), width="stretch")
    st.warning(
        "Changes across years, months or countries may reflect observer activity, platform "
        "coverage, digitisation, reporting delay and dataset publication—not biological change."
    )

    st.markdown(
        """
        <section class="section-intro">
          <div class="section-kicker">Evidence quality</div>
          <h2>Audit the record.<br>Preserve its provenance.</h2>
        </section>
        """,
        unsafe_allow_html=True,
    )
    report = quality_report(data)
    quality_long = report.melt(
        id_vars="query_name",
        value_vars=["event_date_coverage", "uncertainty_coverage", "within_10km", "issue_free"],
        var_name="measure",
        value_name="coverage",
    )
    quality_chart = px.bar(
        quality_long,
        x="coverage",
        y="measure",
        color="query_name",
        orientation="h",
        barmode="group",
        title="Audit-sample quality indicators",
        labels={"coverage": "Records meeting indicator · %", "measure": "", "query_name": "Species"},
        color_discrete_sequence=PALETTE,
    )
    st.plotly_chart(_style_figure(quality_chart, 500), width="stretch")

    license_data = data.groupby(["query_name", "license"], as_index=False).agg(records=("record_id", "nunique"))
    license_data["share"] = license_data["records"] / license_data.groupby("query_name")["records"].transform("sum") * 100
    dataset_data = (
        data.groupby(["dataset_key", "dataset_title", "license"], as_index=False)
        .agg(records=("record_id", "nunique"), species=("query_name", "nunique"), median_quality=("quality_score", "median"))
        .sort_values("records", ascending=False)
        .head(15)
    )
    license_col, dataset_col = st.columns([0.42, 0.58])
    with license_col:
        license_chart = px.bar(
            license_data,
            x="share",
            y="license",
            color="query_name",
            orientation="h",
            barmode="group",
            title="Occurrence license mix",
            labels={"share": "Sample share · %", "license": "", "query_name": "Species"},
            color_discrete_sequence=PALETTE,
        )
        st.plotly_chart(_style_figure(license_chart, 500), width="stretch")
    with dataset_col:
        st.markdown("#### Largest source datasets in the sample")
        st.dataframe(dataset_data, width="stretch", hide_index=True)

    st.markdown("#### Record explorer")
    q1, q2 = st.columns(2)
    with q1:
        minimum_quality = st.slider("Minimum quality score", 0, 100, 50, 5)
    with q2:
        basis_options = sorted(data["basis_of_record"].unique())
        selected_bases = st.multiselect("Basis of record", basis_options, default=basis_options)
    records = data[data["quality_score"].ge(minimum_quality) & data["basis_of_record"].isin(selected_bases)].copy()
    if records.empty:
        st.info("No sample records match the selected quality and evidence filters.")
    else:
        audit = records[
            ["query_name", "scientific_name", "event_date", "country", "locality", "basis_of_record", "coordinate_uncertainty_m", "quality_score", "issues", "license", "dataset_title", "record_url"]
        ].copy()
        audit.columns = ["Query species", "Recorded scientific name", "Event date", "Country", "Locality", "Basis", "Coordinate uncertainty · m", "Quality score", "GBIF issues", "License", "Dataset", "GBIF record"]
        st.dataframe(
            audit.style.format({"Coordinate uncertainty · m": "{:,.0f}", "Quality score": "{:.0f}"}),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download provenance-aware occurrence audit",
            audit.to_csv(index=False).encode("utf-8"),
            "gbif_biodiversity_observation_audit.csv",
            "text/csv",
            width="stretch",
        )

    l1, l2, l3 = st.columns(3)
    l1.link_button("GBIF Occurrence API", OCCURRENCE_DOCS_URL, width="stretch")
    l2.link_button("GBIF data terms", TERMS_URL, width="stretch")
    l3.link_button("GBIF citation guidance", CITATION_URL, width="stretch")
    st.caption(
        f"GBIF-mediated data · retrieved {metadata['retrieved_at']} · mode: {metadata['mode']} · "
        f"query: Europe, present occurrences, coordinates required, {metadata['start_year']}–{metadata['end_year']}. "
        "Record-level publisher, dataset and license provenance is retained in the export."
    )
