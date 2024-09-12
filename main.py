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

# Full list of country names
COUNTRIES = [
    "Global", "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", 
    "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", 
    "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", 
    "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", 
    "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo (Congo-Brazzaville)", 
    "Congo (Democratic Republic of the)", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czechia", "Denmark", 
    "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", 
    "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", 
    "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", 
    "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", 
    "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", 
    "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", 
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", 
    "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", 
    "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea", 
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", 
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis", 
    "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", 
    "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", 
    "Solomon Islands", "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", 
    "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", 
    "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", 
    "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", 
    "Vanuatu", "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
]

# Map of country names to country codes (ISO 3166-1 alpha-2)
COUNTRY_CODES = {
    "Global": "", "Afghanistan": "AF", "Albania": "AL", "Algeria": "DZ", "Andorra": "AD", "Angola": "AO", 
    "Antigua and Barbuda": "AG", "Argentina": "AR", "Armenia": "AM", "Australia": "AU", "Austria": "AT", 
    "Azerbaijan": "AZ", "Bahamas": "BS", "Bahrain": "BH", "Bangladesh": "BD", "Barbados": "BB", "Belarus": "BY", 
    "Belgium": "BE", "Belize": "BZ", "Benin": "BJ", "Bhutan": "BT", "Bolivia": "BO", 
    "Bosnia and Herzegovina": "BA", "Botswana": "BW", "Brazil": "BR", "Brunei": "BN", "Bulgaria": "BG", 
    "Burkina Faso": "BF", "Burundi": "BI", "Cabo Verde": "CV", "Cambodia": "KH", "Cameroon": "CM", 
    "Canada": "CA", "Central African Republic": "CF", "Chad": "TD", "Chile": "CL", "China": "CN", 
    "Colombia": "CO", "Comoros": "KM", "Congo (Congo-Brazzaville)": "CG", 
    "Congo (Democratic Republic of the)": "CD", "Costa Rica": "CR", "Croatia": "HR", "Cuba": "CU", 
    "Cyprus": "CY", "Czechia": "CZ", "Denmark": "DK", "Djibouti": "DJ", "Dominica": "DM", 
    "Dominican Republic": "DO", "Ecuador": "EC", "Egypt": "EG", "El Salvador": "SV", 
    "Equatorial Guinea": "GQ", "Eritrea": "ER", "Estonia": "EE", "Eswatini": "SZ", "Ethiopia": "ET", 
    "Fiji": "FJ", "Finland": "FI", "France": "FR", "Gabon": "GA", "Gambia": "GM", "Georgia": "GE", 
    "Germany": "DE", "Ghana": "GH", "Greece": "GR", "Grenada": "GD", "Guatemala": "GT", "Guinea": "GN", 
    "Guinea-Bissau": "GW", "Guyana": "GY", "Haiti": "HT", "Honduras": "HN", "Hungary": "HU", 
    "Iceland": "IS", "India": "IN", "Indonesia": "ID", "Iran": "IR", "Iraq": "IQ", "Ireland": "IE", 
    "Israel": "IL", "Italy": "IT", "Jamaica": "JM", "Japan": "JP", "Jordan": "JO", "Kazakhstan": "KZ", 
    "Kenya": "KE", "Kiribati": "KI", "Kuwait": "KW", "Kyrgyzstan": "KG", "Laos": "LA", "Latvia": "LV", 
    "Lebanon": "LB", "Lesotho": "LS", "Liberia": "LR", "Libya": "LY", "Liechtenstein": "LI", "Lithuania": "LT", 
    "Luxembourg": "LU", "Madagascar": "MG", "Malawi": "MW", "Malaysia": "MY", "Maldives": "MV", "Mali": "ML", 
    "Malta": "MT", "Marshall Islands": "MH", "Mauritania": "MR", "Mauritius": "MU", "Mexico": "MX", 
    "Micronesia": "FM", "Moldova": "MD", "Monaco": "MC", "Mongolia": "MN", "Montenegro": "ME", 
    "Morocco": "MA", "Mozambique": "MZ", "Myanmar": "MM", "Namibia": "NA", "Nauru": "NR", "Nepal": "NP", 
    "Netherlands": "NL", "New Zealand": "NZ", "Nicaragua": "NI", "Niger": "NE", "Nigeria": "NG", 
    "North Korea": "KP", "North Macedonia": "MK", "Norway": "NO", "Oman": "OM", "Pakistan": "PK", 
    "Palau": "PW", "Panama": "PA", "Papua New Guinea": "PG", "Paraguay": "PY", "Peru": "PE", 
    "Philippines": "PH", "Poland": "PL", "Portugal": "PT", "Qatar": "QA", "Romania": "RO", "Russia": "RU", 
    "Rwanda": "RW", "Saint Kitts and Nevis": "KN", "Saint Lucia": "LC", 
    "Saint Vincent and the Grenadines": "VC", "Samoa": "WS", "San Marino": "SM", 
    "Sao Tome and Principe": "ST", "Saudi Arabia": "SA", "Senegal": "SN", 
    "Serbia": "RS", "Seychelles": "SC", "Sierra Leone": "SL", "Singapore": "SG", 
    "Slovakia": "SK", "Slovenia": "SI", "Solomon Islands": "SB", "Somalia": "SO", 
    "South Africa": "ZA", "South Korea": "KR", "South Sudan": "SS", "Spain": "ES", 
    "Sri Lanka": "LK", "Sudan": "SD", "Suriname": "SR", "Sweden": "SE", 
    "Switzerland": "CH", "Syria": "SY", "Taiwan": "TW", "Tajikistan": "TJ", 
    "Tanzania": "TZ", "Thailand": "TH", "Timor-Leste": "TL", "Togo": "TG", 
    "Tonga": "TO", "Trinidad and Tobago": "TT", "Tunisia": "TN", "Turkey": "TR", 
    "Turkmenistan": "TM", "Tuvalu": "TV", "Uganda": "UG", "Ukraine": "UA", 
    "United Arab Emirates": "AE", "United Kingdom": "GB", "United States": "US", 
    "Uruguay": "UY", "Uzbekistan": "UZ", "Vanuatu": "VU", "Vatican City": "VA", 
    "Venezuela": "VE", "Vietnam": "VN", "Yemen": "YE", "Zambia": "ZM", "Zimbabwe": "ZW"
}

# Streamlit interface
st.set_page_config(layout="wide")
st.title('Google Trends Analysis')

# Set default values
DEFAULT_TIMEFRAME = 'all'
DEFAULT_TERMS = "solar panels, heat pump, energy storage, wind turbine, electric vehicle"

def fetch_trends(keywords, timeframe, geo):
    """Fetch Google Trends data and ignore related queries if they fail."""
    try:
        # Validate that keywords is not empty
        if not keywords:
            st.error("No valid keywords provided.")
            return pd.DataFrame(), {}

        # Resetting session for each request
        pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=1)

        # Build payload for the trends query
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')

        # Fetch interest over time (main trends data)
        df = pytrends.interest_over_time()

        if df.empty:
            st.warning("No data available for the selected parameters.")
            return pd.DataFrame(), {}

        if 'isPartial' in df.columns:
            df = df.drop(columns='isPartial')  # Drop the 'isPartial' column if it exists

        # Fetch related queries safely and ignore errors
        related_queries = {}
        try:
            related_queries_raw = pytrends.related_queries()
            if related_queries_raw:
                for term, queries in related_queries_raw.items():
                    if queries and 'top' in queries and queries['top']:
                        related_queries[term] = queries['top']
        except IndexError:
            # If related queries fail, log the error but continue with trend data
            st.warning("Related queries could not be fetched for one or more terms.")
        
        return df, related_queries  # Return trends data even if related queries fail

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.warning("Rate limit exceeded. Please wait and try again later.")
        elif e.response.status_code == 400:
            st.error("Bad request: Invalid parameters. Please check the timeframe and country codes.")
        else:
            st.error(f"HTTP error occurred: {e}")
        
    except requests.exceptions.RequestException as e:
        st.error(f"Request error occurred: {e}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

    return pd.DataFrame(), {}



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

    data = pd.DataFrame()

    with col1:
        search_terms = st.text_input("Enter search terms separated by commas", DEFAULT_TERMS)
        terms_list = [term.strip() for term in search_terms.split(',') if term.strip()] 
        timeframe = st.selectbox("Select Timeframe", ["today 1-m", "today 3-m", "today 12-m", "past 5-y", "all"], index=4)
        geo_name = st.selectbox("Select Country (leave empty for global)", COUNTRIES)
        geo_code = COUNTRY_CODES[geo_name]

        if st.button('Get Trends'):
            if not terms_list:
                st.error("Please enter at least one search term.")
            else:
                data, related_queries = fetch_trends(terms_list, timeframe, geo_code)

    # Display Plotly graph
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
                st.markdown(f"**Rising Periods:** {row['Rising Periods']}")
                st.markdown(f"**Falling Periods:** {row['Falling Periods']}")
                st.markdown(f"**Peak Value:** {row['Peak Value']}")
                st.markdown(f"**Lowest Value:** {row['Lowest Value']}")
        # Show related queries
        st.subheader("Related Queries")
        for term, queries in related_queries.items():
            if 'top' in queries:
                st.markdown(f"**Related Queries for {term}:**")
                st.write(queries['top'])

if __name__ == "__main__":
    main()
