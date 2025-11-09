# Ensure these libraries are installed:
# pip install streamlit pandas plotly numpy statsmodels

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose 
from plotly.subplots import make_subplots 

# --- 1. Configuration and Data Loading ---
USD_TO_INR_RATE = 88.02145 
WINDOW = 30
ANNUALIZATION_FACTOR = np.sqrt(252) # Trading days per year
CHART_HEIGHT = 450 

# COIN_MAPPING fixed to match the actual file names you uploaded.
COIN_MAPPING = {
    'Bitcoin (BTC)': 'BTC-USD From 2014 To Dec-2024.csv',
    'Ethereum (ETH)': 'ETH-USD From 2017 To Dec-2024.csv',
    'Binance Coin (BNB)': 'BNB-USD From 2017 To Dec-2024.csv',
    'Cardano (ADA)': 'ADA-USD From 2017 To Dec-2024.csv',
    'Ripple (XRP)': 'XRP-USD From 2017 To Dec-2024.csv',
    'Dogecoin (DOGE)': 'DOGE-USD From 2017 To Dec-2024.csv',
    'Solana (SOL)': 'SOL-USD From 2020 To Dec-2024.csv',
    'Staked Ether (STETH)': 'STETH-USD From 2020 To Dec-2024.csv',
    'Tether (USDT)': 'USDT-USD From 2017 To Dec-2024.csv',
    'USDC': 'USDC-USD From 2018 To Dec-2024.csv',
}

# Streamlit Page Settings
st.set_page_config(
    layout="wide", 
    page_title="Crypto Price Analysis Dashboard (INR)",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Global Light Theme Settings and CSS Fixes
# ----------------------------------------------------
LIGHT_BG = '#FFFFFF'     # WHITE background
DARK_TEXT = '#1C1C1C'    # BLACK text
GRID_COLOR = '#CCCCCC'   # Light grey grid lines
PLOTLY_TEMPLATE = 'plotly_white' # Plotly Light Theme

# CSS Injection: Set background to white and all text to black
st.markdown(
    f"""
    <style>
    /* Global App Background */
    .stApp {{
        background-color: {LIGHT_BG}; 
    }}
    /* Sidebar Background */
    .stSidebar > div:first-child {{ 
        background-color: {LIGHT_BG}; 
    }}
    /* FIX: Set all Streamlit text to black */
    h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stSubheader, .stTitle, div, label {{
        color: {DARK_TEXT} !important;
    }}
    /* Sidebar Selectbox background/text fix */
    .stSelectbox label, .stSelectbox div[data-baseweb="select"] div[role="button"] {{
        color: {DARK_TEXT} !important; 
        background-color: #E0E0E0 !important; /* Light grey box for visibility */
    }}
    /* Dropdown options should have dark text on white popover */
    div[data-baseweb="popover"] .stText, 
    div[data-baseweb="popover"] div[role="option"] {{
        color: #000000 !important; 
        background-color: #FFFFFF !important; 
    }}
    /* Filter header alignment */
    .stSubheader {{
        padding-top: 10px;
        margin-bottom: 0px;
    }}
    </style>
    """, 
    unsafe_allow_html=True
) 


# --- 2. Data Loading and Processing (with Caching) ---

@st.cache_data
def load_and_prepare_data(coin_key, rate):
    """Loads, cleans, and performs all calculations for the selected coin."""
    file_name = COIN_MAPPING[coin_key] 
    
    try:
        df = pd.read_csv(file_name, encoding='latin-1')
    except FileNotFoundError:
        st.error(f"FATAL ERROR: CSV file '{file_name}' not found. Please ensure the data file is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"FATAL ERROR: Data loading failed due to: {e}. Check your CSV file integrity.")
        return pd.DataFrame() 

    df['Date'] = pd.to_datetime(df['Date'])
    
    # NaN/Inf cleaning and numerical conversion
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # INR Conversion
    df['Close Price (INR)'] = df['Close'] * rate
    df['Open Price (INR)'] = df['Open'] * rate
    df['High Price (INR)'] = df['High'] * rate
    df['Low Price (INR)'] = df['Low'] * rate
    df['Trading Range (INR)'] = df['High Price (INR)'] - df['Low Price (INR)']
    df['Volume Price (INR)'] = df['Volume'] * df['Close Price (INR)'] 

    # Volatility Calculations
    df['Log Return'] = np.log(df['Close Price (INR)'].div(df['Close Price (INR)'].shift(1)))
    df['Daily Return (%)'] = df['Log Return'] * 100
    df['Annualized Volatility (%)'] = (
        df['Log Return'].rolling(window=WINDOW).std() * ANNUALIZATION_FACTOR * 100
    )
    
    # Calculate Cumulative Returns (for Growth Analysis)
    df['Cumulative Returns'] = (1 + df['Log Return']).cumprod() - 1

    df.dropna(subset=['Log Return'], inplace=True)
    df.set_index('Date', inplace=True) 

    return df

@st.cache_data
def load_all_close_prices(coin_mapping, rate):
    """Loads Close Price for all coins for the Correlation Heatmap."""
    all_data = {}
    for coin_name, file_name in coin_mapping.items():
        try:
            df = pd.read_csv(file_name, encoding='latin-1')
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df['Close Price (INR)'] = df['Close'] * rate
            all_data[coin_name] = df['Close Price (INR)']
        except Exception:
            # If a file fails to load, skip it
            pass
            
    # Combine into a single DataFrame, aligning by Date
    df_combined = pd.DataFrame(all_data).dropna()
    return df_combined


# --- 3. Sidebar (Coin Selection) ---
st.sidebar.header("Select Coin")

coin_options = sorted(list(COIN_MAPPING.keys()))
selected_coin = st.sidebar.selectbox(
    "Choose Cryptocurrency",
    options=coin_options, 
    index=coin_options.index('Bitcoin (BTC)') if 'Bitcoin (BTC)' in coin_options else 0
)

# Load data for the selected coin
df = load_and_prepare_data(selected_coin, USD_TO_INR_RATE)

if df.empty:
    st.stop()


# --- 3.1. Global Filter UI ---
st.sidebar.markdown("---")
st.sidebar.header("Global Time Filter")

# Month/Year Logic setup
month_to_num = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
month_names = ['All'] + list(month_to_num.keys())

# Extract unique years for the filter options
df_years = sorted(list(df.index.year.unique()), reverse=True)
all_years = ['All'] + df_years

# Global Year Filter
selected_year_global = st.sidebar.selectbox(
    'Filter by Year (Global)',
    options=all_years,
    index=0
)

# Global Month Filter
selected_month_name_global = st.sidebar.selectbox(
    'Filter by Month (Global)',
    options=month_names,
    index=0
)

# --- Apply Global Filtering Logic ---
filtered_df = df.copy() 

if selected_year_global != 'All':
    filtered_df = filtered_df[filtered_df.index.year == selected_year_global]

if selected_month_name_global != 'All':
    selected_month_num = month_to_num[selected_month_name_global]
    filtered_df = filtered_df[filtered_df.index.month == selected_month_num]

# If filtering results in an empty DataFrame, use the original data and display a message.
if filtered_df.empty:
    st.warning(f"Global filter combination (Year: {selected_year_global}, Month: {selected_month_name_global}) returned no data. Showing full data.")
    filtered_df = df.copy()


# --- 4. Streamlit Dashboard Layout ---
st.title(f"{selected_coin} Price Analysis Dashboard (INR) 📊")
st.caption(f"All prices are calculated using an estimated USD to INR rate of ₹{USD_TO_INR_RATE}")
st.markdown("---")


# --- 5. Functions to Create Charts ---

def update_light_theme_layout(fig):
    """Applies Light Theme settings to Plotly charts."""
    fig.update_layout(
        plot_bgcolor=LIGHT_BG,
        paper_bgcolor=LIGHT_BG,
        font=dict(color=DARK_TEXT), # Text color is black
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
    return fig


def create_line_chart(df, y_col, title, y_title, coin_name):
    """Standard Line Chart (Price/Volatility/Returns)"""
    fig = px.line(df.reset_index(), x='Date', y=y_col, title=f'{coin_name} - {title}', template=PLOTLY_TEMPLATE)
    # Rangeslider removed but panning/zooming enabled by default
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(xaxis_title="Date", yaxis_title=y_title, height=CHART_HEIGHT) 
    return update_light_theme_layout(fig)

def create_dual_axis_chart(df, coin_name):
    """Close Price and Volume on dual axes (Volume Color Fixed to Dark RED)."""
    df_reset = df.reset_index()
    fig = go.Figure()
    
    # Close Price Line Color: Dark Blue (#00008B)
    CLOSE_PRICE_COLOR = '#00008B' 
    fig.add_trace(go.Scatter(x=df_reset['Date'], y=df_reset['Close Price (INR)'], name='Close Price (INR)', yaxis='y1', line=dict(color=CLOSE_PRICE_COLOR))) 
    
    # Volume Bar Color: Crimson (#DC143C) - Dark Red
    VOLUME_COLOR = '#DC143C'
    fig.add_trace(go.Bar(x=df_reset['Date'], y=df_reset['Volume Price (INR)'], name='Volume (INR)', yaxis='y2', marker=dict(color=VOLUME_COLOR), opacity=0.7))

    fig.update_layout(
        title=f'{coin_name} - Close Price (INR) and Volume (INR) Over Time',
        # Update Y-axis title colors
        yaxis=dict(title=dict(text='Close Price (INR)', font=dict(color=CLOSE_PRICE_COLOR)), side='left', showgrid=False),
        yaxis2=dict(title=dict(text='Volume (INR)', font=dict(color=VOLUME_COLOR)), overlaying='y', side='right', showgrid=True),
        # Rangeslider removed but panning/zooming enabled by default
        xaxis=dict(rangeslider_visible=False),
        hovermode="x unified", height=CHART_HEIGHT, template=PLOTLY_TEMPLATE
    )
    return update_light_theme_layout(fig)

def create_candlestick_chart(df, coin_name):
    """OHLC Candlestick Chart."""
    df_reset = df.reset_index()
    fig = go.Figure(data=[go.Candlestick(
        x=df_reset['Date'],
        open=df_reset['Open Price (INR)'],
        high=df_reset['High Price (INR)'],
        low=df_reset['Low Price (INR)'],
        close=df_reset['Close Price (INR)'],
        name=f'{coin_name}/INR',
        increasing_line_color='green', 
        decreasing_line_color='red'    
    )])
    
    fig.update_layout(title=f'{coin_name} - Candlestick Chart (Open, High, Low, Close) in Indian Rupees', yaxis_title='Price (INR)', height=CHART_HEIGHT, template=PLOTLY_TEMPLATE) 
    
    # Rangeslider removed but panning/zooming enabled by default
    fig.update_xaxes(
        rangeslider_visible=False, 
    )
    
    return update_light_theme_layout(fig)

def create_histogram(df, coin_name):
    """Distribution of Daily Returns (Histogram)."""
    fig = px.histogram(df.reset_index().dropna(subset=['Daily Return (%)']), x='Daily Return (%)', nbins=80, marginal="box", histnorm='probability density', title=f'{coin_name} - Distribution of Daily Log Returns (%)', template=PLOTLY_TEMPLATE)
    fig.update_layout(xaxis_title="Daily Log Return (%)", yaxis_title="Density", height=CHART_HEIGHT) 
    fig.update_traces(marker_color='#5B96EB', selector=dict(type='histogram'))
    return update_light_theme_layout(fig)

def create_monday_correlation(df, coin_name):
    """Correlation between Open and Close on Mondays."""
    df_reset = df.reset_index()
    df_monday = df_reset[df_reset['Date'].dt.dayofweek == 0].copy()
    
    fig = px.scatter(
        df_monday, 
        x='Open Price (INR)', 
        y='Close Price (INR)', 
        color='Daily Return (%)',  
        color_continuous_scale=[(0.0, 'red'), (0.5, 'white'), (1.0, 'blue')], # Adjusted scale for light BG
        color_continuous_midpoint=0,      
        trendline='ols', 
        title=f'{coin_name} - Correlation: Open vs Close Price on MONDAYS',
        template=PLOTLY_TEMPLATE
    )

    fig.update_layout(
        xaxis=dict(title='Opening Price (INR)', title_font=dict(color='darkgreen'), tickfont=dict(color='darkgreen')),
        yaxis=dict(title='Closing Price (INR)', title_font=dict(color='darkred'), tickfont=dict(color='darkred')),
        height=CHART_HEIGHT 
    )
    
    fig.update_traces(marker=dict(size=7, line=dict(width=1, color=DARK_TEXT)), selector=dict(mode='markers'))
    
    if len(fig.data) > 1:
        fig.data[1].line.color = 'black'
        
    return update_light_theme_layout(fig)

def create_moving_average_plot(df, coin_name):
    """Moving Averages Plot for Trend Analysis."""
    df['SMA_50'] = df['Close Price (INR)'].rolling(window=50).mean()
    df['SMA_200'] = df['Close Price (INR)'].rolling(window=200).mean()
    df_reset = df.reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_reset['Date'], y=df_reset['Close Price (INR)'], mode='lines', name='Close Price (INR)', line=dict(color='darkblue', width=1)))
    fig.add_trace(go.Scatter(x=df_reset['Date'], y=df_reset['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=df_reset['Date'], y=df_reset['SMA_200'], mode='lines', name='SMA 200', line=dict(color='red', width=2)))
    
    fig.update_layout(
        title=f'{coin_name} - Price and Simple Moving Averages (50-Day & 200-Day)',
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        height=CHART_HEIGHT,
        # Rangeslider removed but panning/zooming enabled by default
        xaxis_rangeslider_visible=False, 
        template=PLOTLY_TEMPLATE
    )
    return update_light_theme_layout(fig)

def create_cumulative_returns_plot(df, coin_name):
    """Cumulative Returns Plot for Growth Analysis."""
    fig = px.line(df.reset_index(), 
                  x='Date', 
                  y='Cumulative Returns', 
                  title=f'{coin_name} - Cumulative Returns (Growth of $1 Investment)', 
                  template=PLOTLY_TEMPLATE)
    
    # Rangeslider removed but panning/zooming enabled by default
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        xaxis_title="Date", 
        yaxis_title="Cumulative Returns (Factor)", 
        height=CHART_HEIGHT,
        yaxis_tickformat=".2f"
    ) 
    return update_light_theme_layout(fig)

def create_decomposition_plot(df, coin_name):
    """Seasonal/Time Series Decomposition Plot using make_subplots."""
    
    if len(df) < 30:
        st.warning(f"Decomposition Analysis requires at least 30 days of data for {coin_name}.")
        return None

    try:
        # Use period=7 for weekly seasonality
        decomposition = seasonal_decompose(df['Close Price (INR)'], model='additive', period=7, extrapolate_trend='freq')
    except Exception as e:
        st.error(f"Decomposition Analysis failed: Ensure your data has no gaps and is sufficiently long. Error: {e}")
        return None
    
    decomposed_df = pd.DataFrame({
        'Date': df.index,
        'Observed': df['Close Price (INR)'],
        'Trend': decomposition.trend,
        'Seasonal': decomposition.seasonal,
        'Residual': decomposition.resid
    }).set_index('Date')
    
    fig = make_subplots(rows=4, cols=1, 
                        shared_xaxes=True, 
                        subplot_titles=('Observed Price', 'Trend Component', 'Seasonal Component', 'Residual Component'))

    # Darker colors for better visibility on white background
    fig.add_trace(go.Scatter(x=decomposed_df.index, y=decomposed_df['Observed'], name='Observed Price', line=dict(color='darkblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposed_df.index, y=decomposed_df['Trend'], name='Trend Component', line=dict(color='darkgreen')), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposed_df.index, y=decomposed_df['Seasonal'], name='Seasonal Component', line=dict(color='darkorange')), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomposed_df.index, y=decomposed_df['Residual'], name='Residual Component', line=dict(color='darkred')), row=4, col=1)
    
    fig.update_layout(
        height=CHART_HEIGHT * 2, 
        title_text=f"{coin_name} - Time Series Decomposition (Period=7 Days)",
        showlegend=False,
        # Rangeslider removed but panning/zooming enabled by default
        xaxis_rangeslider_visible=False 
    )
    
    return update_light_theme_layout(fig)

def create_correlation_heatmap(coin_mapping):
    """Creates a correlation heatmap for all cryptocurrencies."""
    # NOTE: This uses ALL available data for accurate cross-correlation.
    df_combined = load_all_close_prices(coin_mapping, USD_TO_INR_RATE)
    
    if df_combined.empty or len(df_combined.columns) < 2:
        st.warning("Could not load enough data to create Correlation Heatmap.")
        return None
        
    corr_matrix = df_combined.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdYlBu', # Red-Yellow-Blue scale is good for correlation (-1 to 1)
        zmin=-1, zmax=1
    ))
    
    fig.update_layout(
        title='Correlation Heatmap Across All Cryptocurrencies (Close Price INR)',
        xaxis_title='Cryptocurrency',
        yaxis_title='Cryptocurrency',
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange='reversed'),
        height=CHART_HEIGHT * 1.5,
        template=PLOTLY_TEMPLATE
    )
    
    # Add annotations (correlation values) for clarity
    annotations = []
    for i, row in enumerate(corr_matrix.values):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=corr_matrix.columns[j],
                    y=corr_matrix.index[i],
                    text=f"{val:.2f}",
                    showarrow=False,
                    # Black text for low correlation (light color on map), White text otherwise
                    font=dict(color='black' if abs(val) < 0.5 else 'white', size=10) 
                )
            )
    fig.update_layout(annotations=annotations)
    return update_light_theme_layout(fig)


# --- 6. Creating the Layout (Plots) ---

# Helper function to render a chart section using the globally filtered DF
def render_chart_section_global(df_to_use, section_header, sub_header_text, chart_func, *args):
    
    # FIX: Only call st.header() if the string is not empty
    if section_header:
        st.header(section_header)
        
    # FIX: Only call st.subheader() if the string is not empty
    if sub_header_text:
        st.subheader(sub_header_text)
    
    # FIX: Pass the arguments correctly. The *args contain the y_col, title, y_title, 
    # and selected_coin should be the last argument (coin_name).
    chart_args = list(args) + [selected_coin]
    
    # Render the chart using the globally filtered DataFrame
    # chart_func will now receive: (df, y_col, title, y_title, coin_name) OR (df, coin_name)
    chart = chart_func(df_to_use, *chart_args)
    st.plotly_chart(chart, use_container_width=True)
    st.markdown("---")


# 1. Descriptive Analysis & Price Action
render_chart_section_global(
    filtered_df, "1. Descriptive Analysis & Price Action", "1.1. Candlestick Chart (OHLC)", 
    create_candlestick_chart
)

# 2. Volume Analysis 
render_chart_section_global(
    filtered_df, "2. Volume Analysis", "2.1. Price and Volume Over Time", 
    create_dual_axis_chart
)

# 3. Trend Analysis
render_chart_section_global(
    filtered_df, "3. Trend Analysis", "3.1. Moving Averages (50-Day & 200-Day)", 
    create_moving_average_plot
)

# 4. Time Series Decomposition
render_chart_section_global(
    filtered_df, "4. Time Series Decomposition", "4.1. Price Decomposition (Trend, Seasonal, Residual)", 
    create_decomposition_plot
)

# 5. Volatility & Risk Analysis (Rolling Volatility)
render_chart_section_global(
    filtered_df, "5. Volatility & Risk Analysis", "5.1. Annualized Rolling Volatility (30-Day)", 
    create_line_chart, 'Annualized Volatility (%)', 'Annualized Rolling Volatility (30-Day)', 'Volatility (%)'
)

# 5. Volatility & Risk Analysis (Histogram)
render_chart_section_global(
    filtered_df, "5. Volatility & Risk Analysis", "5.2. Daily Log Returns Distribution (Histogram)", 
    create_histogram
)

# 6. Return & Growth Analysis (Cumulative Returns)
render_chart_section_global(
    filtered_df, "6. Return & Growth Analysis", "6.1. Cumulative Returns (Growth)", 
    create_cumulative_returns_plot
)

# 6. Return & Growth Analysis (Daily Log Returns)
render_chart_section_global(
    filtered_df, "6. Return & Growth Analysis", "6.2. Daily Log Returns Over Time", 
    create_line_chart, 'Daily Return (%)', 'Daily Log Return (%)', 'Daily Log Return (%)'
)

# 7. Correlation Analysis
st.header("7. Correlation Analysis")
st.subheader("7.1. Correlation Heatmap Across All Coins")
# Heatmap uses the full, unfiltered data for all coins
heatmap_chart = create_correlation_heatmap(COIN_MAPPING)
if heatmap_chart:
    st.plotly_chart(heatmap_chart, use_container_width=True)
st.markdown("---")

# 7.2. Open vs Close Price on Mondays (Single Coin)
render_chart_section_global(
    filtered_df, "", "7.2. Open vs Close Price on Mondays (Single Coin)", 
    create_monday_correlation
)