import requests
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title='Job Market Analytics Dashboard', layout='wide')

API_URL = 'https://www.arbeitnow.com/api/job-board-api'

@st.cache_data(ttl=3600)
def fetch_jobs():
    response = requests.get(API_URL, timeout=30)
    response.raise_for_status()
    data = response.json().get('data', [])
    return pd.DataFrame(data)

st.title('Job Market Analytics Dashboard')
st.markdown('Analyze real-world job postings using public API data.')

try:
    df = fetch_jobs()
except Exception as e:
    st.error(f'Failed to fetch data: {e}')
    st.stop()

if df.empty:
    st.warning('No job data available.')
    st.stop()

# Basic cleaning
if 'location' in df.columns:
    df['location'] = df['location'].fillna('Unknown')

if 'remote' in df.columns:
    df['remote'] = df['remote'].astype(str)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric('Total Jobs', len(df))

with col2:
    if 'company_name' in df.columns:
        st.metric('Companies', df['company_name'].nunique())

with col3:
    if 'location' in df.columns:
        st.metric('Locations', df['location'].nunique())

st.divider()

if 'location' in df.columns:
    top_locations = (
        df['location']
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

if 'company_name' in df.columns:
    top_companies = (
        df['company_name']
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

st.subheader('Raw Dataset')
st.dataframe(df)
