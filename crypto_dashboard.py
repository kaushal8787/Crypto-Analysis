# सुनिश्चित करें कि आपके पास ये लाइब्रेरीज़ स्थापित हैं:
# pip install streamlit pandas plotly numpy statsmodels

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose 
from plotly.subplots import make_subplots 

# --- 1. कॉन्फ़िगरेशन और डेटा लोडिंग ---
USD_TO_INR_RATE = 88.02145 
WINDOW = 30
ANNUALIZATION_FACTOR = np.sqrt(252) 
CHART_HEIGHT = 450 

# COIN_MAPPING को आपके द्वारा अपलोड किए गए वास्तविक फ़ाइल नामों के अनुरूप ठीक किया गया है।
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

# Streamlit पेज सेटिंग्स
st.set_page_config(
    layout="wide", 
    page_title="Crypto Price Analysis Dashboard (INR)",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# ग्लोबल लाइट थीम सेटिंग्स और CSS फिक्स
# ----------------------------------------------------
LIGHT_BG = '#FFFFFF'     # WHITE background
DARK_TEXT = '#1C1C1C'    # BLACK text
GRID_COLOR = '#CCCCCC'   # Light grey grid lines
PLOTLY_TEMPLATE = 'plotly_white' # Plotly Light Theme

# CSS इंजेक्शन: बैकग्राउंड को सफेद और सभी टेक्स्ट को काला करें
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
    /* FIX: सभी Streamlit टेक्स्ट को काला करें */
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
    </style>
    """, 
    unsafe_allow_html=True
) 


# --- 2. डेटा लोडिंग और प्रोसेसिंग (Caching के साथ) ---

@st.cache_data
def load_and_prepare_data(coin_key, rate):
    """चयनित कॉइन के लिए डेटा लोड करता है, उसे साफ करता है, और सभी गणनाएँ करता है।"""
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
    
    # NaN/Inf सफाई और संख्यात्मक रूपांतरण
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # INR कन्वर्ज़न
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
    
    # Cumulative Returns की गणना (Growth Analysis के लिए)
    df['Cumulative Returns'] = (1 + df['Log Return']).cumprod() - 1

    df.dropna(subset=['Log Return'], inplace=True)
    df.set_index('Date', inplace=True) 

    return df

@st.cache_data
def load_all_close_prices(coin_mapping, rate):
    """Correlation Heatmap के लिए सभी कॉइन की Close Price लोड करता है।"""
    all_data = {}
    for coin_name, file_name in coin_mapping.items():
        try:
            df = pd.read_csv(file_name, encoding='latin-1')
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df['Close Price (INR)'] = df['Close'] * rate
            all_data[coin_name] = df['Close Price (INR)']
        except Exception:
            pass
            
    df_combined = pd.DataFrame(all_data).dropna()
    return df_combined


# --- 3. Sidebar (कॉइन चयन) ---
st.sidebar.header("Select Coin")

coin_options = sorted(list(COIN_MAPPING.keys()))
selected_coin = st.sidebar.selectbox(
    "Choose Cryptocurrency",
    options=coin_options, 
    index=coin_options.index('Bitcoin (BTC)') if 'Bitcoin (BTC)' in coin_options else 0
)

# चयनित कॉइन के लिए डेटा लोड करें
df = load_and_prepare_data(selected_coin, USD_TO_INR_RATE)

if df.empty:
    st.stop()


# --- 3.1. Filter UI (NEW) ---
st.sidebar.markdown("---")
st.sidebar.header("Time Filter")

# Extract unique years for the filter options
df_years = sorted(list(df.index.year.unique()), reverse=True) # Sort in descending order

# Year Filter
selected_year = st.sidebar.selectbox(
    'Filter by Year',
    options=['All'] + df_years,
    index=0
)

# Month Filter
month_to_num = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
month_names = ['All'] + list(month_to_num.keys())

selected_month_name = st.sidebar.selectbox(
    'Filter by Month',
    options=month_names,
    index=0
)

# --- 3.2. Apply Filtering Logic ---
filtered_df = df.copy() 

if selected_year != 'All':
    filtered_df = filtered_df[filtered_df.index.year == selected_year]

if selected_month_name != 'All':
    selected_month_num = month_to_num[selected_month_name]
    filtered_df = filtered_df[filtered_df.index.month == selected_month_num]

# If filtering results in an empty DataFrame, use the original data and display a message.
if filtered_df.empty:
    st.warning(f"Filter combination (Year: {selected_year}, Month: {selected_month_name}) returned no data. Showing full data.")
    # Resetting to full data only if the initial filter produced nothing
    filtered_df = df.copy()


# --- 4. Streamlit डैशबोर्ड लेआउट ---
st.title(f"{selected_coin} Price Analysis Dashboard (INR) 📊")
st.caption(f"All prices are calculated using an estimated USD to INR rate of ₹{USD_TO_INR_RATE}")
st.markdown("---")


# --- 5. चार्ट बनाने के लिए फ़ंक्शंस ---

def update_light_theme_layout(fig):
    """Plotly चार्ट्स पर लाइट थीम सेटिंग्स लागू करता है।"""
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
        st.warning(f"{coin_name} के लिए Decomposition Analysis को दिखाने के लिए कम से कम 30 दिनों का डेटा चाहिए।")
        return None

    try:
        decomposition = seasonal_decompose(df['Close Price (INR)'], model='additive', period=7, extrapolate_trend='freq')
    except Exception as e:
        st.error(f"Decomposition Analysis फेल हो गया: सुनिश्चित करें कि आपके डेटा में कोई गैप नहीं है और यह पर्याप्त रूप से लंबा है। त्रुटि: {e}")
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
    df_combined = load_all_close_prices(coin_mapping, USD_TO_INR_RATE)
    
    if df_combined.empty or len(df_combined.columns) < 2:
        st.warning("Correlation Heatmap बनाने के लिए पर्याप्त डेटा लोड नहीं किया जा सका।")
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
                    # White text for high correlation (dark color on map), Black text otherwise
                    font=dict(color='black' if abs(val) < 0.5 else 'white', size=10) 
                )
            )
    fig.update_layout(annotations=annotations)
    return update_light_theme_layout(fig)


# --- 6. Streamlit में लेआउट बनाना (Plots) ---

st.header("1. Descriptive Analysis & Price Action")
# 1. Candlestick Chart (OHLC)
st.subheader("1.1. Candlestick Chart (OHLC)")
st.plotly_chart(create_candlestick_chart(filtered_df, selected_coin), use_container_width=True)
st.markdown("---")

# 2. Volume Analysis (Volume Dark Red, Price Dark Blue)
st.header("2. Volume Analysis")
st.subheader("2.1. Price and Volume Over Time")
st.plotly_chart(create_dual_axis_chart(filtered_df, selected_coin), use_container_width=True)
st.markdown("---")


# 3. Trend Analysis
st.header("3. Trend Analysis")
st.subheader("3.1. Moving Averages (50-Day & 200-Day)")
st.plotly_chart(create_moving_average_plot(filtered_df, selected_coin), use_container_width=True)
st.markdown("---")

# 4. Time Series Decomposition
st.header("4. Time Series Decomposition")
st.subheader("4.1. Price Decomposition (Trend, Seasonal, Residual)")
decomposition_chart = create_decomposition_plot(filtered_df, selected_coin)
if decomposition_chart:
    st.plotly_chart(decomposition_chart, use_container_width=True)
st.markdown("---")


# 5. Volatility & Risk Analysis
st.header("5. Volatility & Risk Analysis")
st.subheader("5.1. Annualized Rolling Volatility (30-Day)")
st.plotly_chart(create_line_chart(filtered_df, 'Annualized Volatility (%)', 'Annualized Rolling Volatility (30-Day)', 'Volatility (%)', selected_coin), use_container_width=True)
st.markdown("---")

st.subheader("5.2. Daily Log Returns Distribution (Histogram)")
st.plotly_chart(create_histogram(filtered_df, selected_coin), use_container_width=True)
st.markdown("---")

# 6. Return & Growth Analysis
st.header("6. Return & Growth Analysis")
st.subheader("6.1. Cumulative Returns (Growth)")
st.plotly_chart(create_cumulative_returns_plot(filtered_df, selected_coin), use_container_width=True)
st.markdown("---")

st.subheader("6.2. Daily Log Returns Over Time")
st.plotly_chart(create_line_chart(filtered_df, 'Daily Return (%)', 'Daily Log Return (%)', 'Daily Log Return (%)', selected_coin), use_container_width=True)
st.markdown("---")

# 7. Correlation Analysis
st.header("7. Correlation Analysis")
st.subheader("7.1. Correlation Heatmap Across All Coins")
# Note: Heatmap uses the full, unfiltered data for all coins
heatmap_chart = create_correlation_heatmap(COIN_MAPPING)
if heatmap_chart:
    st.plotly_chart(heatmap_chart, use_container_width=True)
st.markdown("---")

st.subheader("7.2. Open vs Close Price on Mondays (Single Coin)")
st.plotly_chart(create_monday_correlation(filtered_df, selected_coin), use_container_width=True)
st.markdown("---")