# 📈 Cryptocurrency Analysis Dashboard (INR)

This project features a dynamic and comprehensive cryptocurrency analysis dashboard built using **Streamlit** and **Plotly**. It allows users to select from 10 major cryptocurrencies and visualize their price trends, volatility, growth, and market behavior, with all pricing data calculated and displayed in **Indian Rupees (INR)**.

All core price and statistical calculations (Log Returns, Volatility, Moving Averages) are performed in Python using Pandas and NumPy.

---

## ✨ Features and Analysis Included

The dashboard provides a deep dive into the selected cryptocurrency across seven major analysis categories, with all prices converted from USD using an estimated rate of ₹88.02.

### Analysis Sections:

1. **Descriptive Analysis & Price Action:** Core visualization of open, high, low, and close prices.
2. **Volume Analysis:** Price trend overlaid with trading volume (in INR) to gauge liquidity and market activity.
3. **Trend Analysis:** Price comparison against 50-day and 200-day Simple Moving Averages (SMAs) to identify long and short-term trends.
4. **Time Series Decomposition:** Separation of the price series into its core **Trend**, **Seasonal**, and **Residual** components (using `statsmodels`).
5. **Volatility & Risk Analysis:** Measures market risk through Annualized Rolling Volatility and the distribution of daily log returns (Histogram).
6. **Return & Growth Analysis:** Visualization of daily log returns and **Cumulative Returns** (the growth factor of a hypothetical $1 investment).
7. **Correlation Analysis:**
   * **Global Heatmap:** Correlation matrix showing the relationship between **all 10 cryptocurrency close prices** (full historical data).
   * **Open vs Close:** Scatter plot analyzing the correlation between opening and closing prices specifically on Mondays.

### User Controls:

* **Coin Selector:** Select any of the 10 supported cryptocurrencies from the sidebar.
* **Global Time Filter:** Filter all time-series charts by a specific **Year** and **Month** using the sidebar controls.
* **Zoom & Pan:** All Plotly charts are interactive, allowing the user to zoom in and pan (scroll horizontally) across the data without the clutter of a range slider.

---

## 🛠️ Setup and Installation

### 1. Prerequisites

You need Python 3.8+ installed on your system.

### 2. Project Structure

Ensure your project directory contains the following files:

### 3. Install Dependencies

Install all required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### To run this project loaclly 

```bash
streamlit run crypto_dashboard.py
```


## Deployment on Streamlit Cloud

The project is deployed and live: [https://cryptocanalysis.streamlit.app/](https://cryptocanalysis.streamlit.app/)
