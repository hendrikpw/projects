import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_loader import fetch_jobs
from src.skill_extractor import extract_skills

st.set_page_config(page_title='Job Market Analytics Dashboard', layout='wide')

st.title('Job Market Analytics Dashboard')
st.markdown('Real-world labor market analytics using live public API data.')

try:
    df = fetch_jobs()
except Exception as e:
    st.error(f'Failed to fetch data: {e}')
    st.stop()

if df.empty:
    st.warning('No job data available.')
    st.stop()

# Sidebar filters
st.sidebar.header('Filters')

search_term = st.sidebar.text_input('Search keyword')

if 'location' in df.columns:
    selected_locations = st.sidebar.multiselect(
        'Locations',
        options=sorted(df['location'].dropna().unique())
    )
else:
    selected_locations = []

filtered_df = df.copy()

if search_term:
    filtered_df = filtered_df[
        filtered_df.astype(str)
        .apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)
    ]

if selected_locations:
    filtered_df = filtered_df[
        filtered_df['location'].isin(selected_locations)
    ]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('Total Jobs', len(filtered_df))

with col2:
    if 'company_name' in filtered_df.columns:
        st.metric('Companies', filtered_df['company_name'].nunique())

with col3:
    if 'location' in filtered_df.columns:
        st.metric('Locations', filtered_df['location'].nunique())

with col4:
    if 'remote' in filtered_df.columns:
        remote_jobs = (filtered_df['remote'] == 'True').sum()
        st.metric('Remote Jobs', int(remote_jobs))

st.divider()

if 'location' in filtered_df.columns:
    top_locations = (
        filtered_df['location']
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_locations.columns = ['Location', 'Jobs']

    fig_locations = px.bar(
        top_locations,
        x='Location',
        y='Jobs',
        title='Top Job Locations'
    )

    st.plotly_chart(fig_locations, use_container_width=True)

if 'company_name' in filtered_df.columns:
    top_companies = (
        filtered_df['company_name']
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_companies.columns = ['Company', 'Jobs']

    fig_companies = px.bar(
        top_companies,
        x='Company',
        y='Jobs',
        title='Top Hiring Companies'
    )

    st.plotly_chart(fig_companies, use_container_width=True)

# Skill analytics
if 'description' in filtered_df.columns:
    skill_counter = extract_skills(filtered_df['description'])

    if skill_counter:
        skills_df = pd.DataFrame(
            skill_counter.items(),
            columns=['Skill', 'Mentions']
        ).sort_values(by='Mentions', ascending=False)

        fig_skills = px.bar(
            skills_df.head(10),
            x='Skill',
            y='Mentions',
            title='Most Requested Skills'
        )

        st.plotly_chart(fig_skills, use_container_width=True)

st.subheader('Filtered Dataset')
st.dataframe(filtered_df)
