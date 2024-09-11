import streamlit as st
from pytrends.request import TrendReq
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import time
import requests

# Set pandas option to handle FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Streamlit interface
st.set_page_config(layout="wide")
st.title('Google Trends Analysis')

# Set default values
DEFAULT_TIMEFRAME = 'all'
DEFAULT_TERMS = "Ricky Martin, Lady Gaga, Jennifer Lopez"

def fetch_trends(keywords, timeframe):
    """Fetch Google Trends data with error handling and rate limiting."""
    pytrends = TrendReq(hl='en-US', tz=360)
    retries = 3
    delay = 10  # Initial delay before retrying
    for attempt in range(retries):
        try:
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
            return pytrends.interest_over_time()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                st.warning("Rate limit exceeded. Please wait and try again later.")
            elif e.response.status_code == 400:
                st.error("Invalid request. Please check your search terms and try again.")
            else:
                st.error(f"HTTP error occurred: {e}")
            if attempt < retries - 1:
                st.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                st.error("Failed to fetch data after several attempts.")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            st.error(f"Request error occurred: {e}")
            if attempt < retries - 1:
                st.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                st.error("Failed to fetch data after several attempts.")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return pd.DataFrame()

def analyze_trends(series):
    """Analyze multiple trends in the time series."""
    results = []
    for term in series.columns:
        data = series[term].dropna()
        if len(data) > 12:
            decomposition = seasonal_decompose(data, period=12)
            trend = decomposition.trend.dropna()
            residual = decomposition.resid.dropna()

            # Identify significant changes and trend phases
            changes = np.diff(data)
            change_points = np.where(changes > 0, 'rising', 'falling')
            points = pd.DataFrame({'Date': data.index[1:], 'Change': change_points})

            # Extract main trends
            trend_points = points.groupby('Change').size()
            main_trends = trend_points.nlargest(3).index.tolist()

            results.append({
                'Keyword': term,
                'Main Trends': ', '.join(main_trends),
                'Peak Value': data.max(),
                'Lowest Value': data.min(),
            })
        else:
            results.append({
                'Keyword': term,
                'Main Trends': 'Insufficient data',
                'Peak Value': 'N/A',
                'Lowest Value': 'N/A',
            })

    return pd.DataFrame(results)

def plot_trends(data, terms_list):
    """Create a Plotly graph for the trends."""
    fig = go.Figure()
    for term in terms_list:
        fig.add_trace(go.Scatter(x=data.index, y=data[term], mode='lines', name=term))
    fig.update_layout(title='Google Trends Interest Over Time',
                      xaxis_title='Date',
                      yaxis_title='Interest Over Time',
                      legend_title='Search Terms',
                      xaxis=dict(rangeselector=dict(buttons=[dict(count=1, label='1m', step='month', stepmode='backward'),
                                                             dict(count=6, label='6m', step='month', stepmode='backward'),
                                                             dict(count=1, label='YTD', step='year', stepmode='todate'),
                                                             dict(label='All', step='all')], visible=True)),
                      yaxis=dict(title='Interest Over Time'))
    return fig

def main():
    """Main function to run the Streamlit app."""
    # Input for search terms and timeframe
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    # Initialize data to avoid UnboundLocalError
    data = pd.DataFrame()

    with col1:
        search_terms = st.text_input("Enter search terms separated by commas", DEFAULT_TERMS)
        terms_list = [term.strip() for term in search_terms.split(',')]
        timeframe = st.selectbox("Select Timeframe", ["today 1-m", "today 3-m", "today 6-m", "today 12-m", "past 5-y", "all"], index=5)
        if st.button('Get Trends'):
            if not terms_list:
                st.error("Please enter at least one search term.")
            else:
                data = fetch_trends(terms_list, timeframe)

    # Display Plotly graph outside columns
    if not data.empty:
        st.plotly_chart(plot_trends(data, terms_list), use_container_width=True)

        # Analysis
        st.subheader('Trend Analysis')
        analysis_results = analyze_trends(data)
                    
        # Display analysis in columns
        num_cols = min(5, len(analysis_results))
        cols = st.columns(num_cols)
                    
        for i, (index, row) in enumerate(analysis_results.iterrows()):
            with cols[i]:
                st.markdown(f"### {row['Keyword']}")
                st.markdown(f"**Main Trends:** {row['Main Trends']}")
                st.markdown(f"**Peak Value:** {row['Peak Value']:.2f}")
                st.markdown(f"**Lowest Value:** {row['Lowest Value']:.2f}")

if __name__ == "__main__":
    main()
