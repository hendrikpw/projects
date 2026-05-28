import requests
import pandas as pd

API_URL = 'https://www.arbeitnow.com/api/job-board-api'


def fetch_jobs():
    response = requests.get(API_URL, timeout=30)
    response.raise_for_status()

    data = response.json().get('data', [])
    df = pd.DataFrame(data)

    if 'location' in df.columns:
        df['location'] = df['location'].fillna('Unknown')

    if 'remote' in df.columns:
        df['remote'] = df['remote'].astype(str)

    return df
