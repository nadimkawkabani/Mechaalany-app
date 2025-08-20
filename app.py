import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np

# --- Part 0: Page Configuration ---
st.set_page_config(layout="wide")


# --- Part 1: Load and Clean ALL Data from GitHub ---
GITHUB_URL = "https://raw.githubusercontent.com/nadimkawkabani/projectsstudents-app/main/Dashboard.xlsx"

@st.cache_data
def load_data(url):
    """Loads all data from the specified Excel file URL and returns cleaned dataframes."""
    try:
        xls = pd.ExcelFile(url)
    except Exception as e:
        st.error(f"Failed to load the Excel file from the URL. Please check the link. Error: {e}")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    def clean_sheet(sheet_name, id_col_name, header_row=0, item_prefix=None):
        try:
            df = pd.read_excel(xls, sheet_name, header=header_row)
            df.rename(columns={df.columns[0]: id_col_name}, inplace=True)
            if item_prefix and id_col_name in df.columns:
                df = df[df[id_col_name].str.startswith(item_prefix, na=False)].copy()
            if id_col_name in df.columns:
                df[id_col_name] = df[id_col_name].astype(str)
            return df
        except Exception as e:
            st.warning(f"Could not load or process '{sheet_name}' sheet: {e}")
            return pd.DataFrame()

    sales_df = clean_sheet("Sales", "Item", header_row=2, item_prefix='407002-')
    purchases_df = clean_sheet("Purchases_kg", "Item")
    production_df = clean_sheet("Production", "Item")
    customers_2024_df = clean_sheet("Customers_2024", "Product")
    customers_2022_df = clean_sheet("Customers_2022", "Product")
        
    return sales_df, purchases_df, production_df, customers_2024_df, customers_2022_df

# Load all the business intelligence dataframes
sales_df, purchases_df, production_df, customers_2024_df, customers_2022_df = load_data(GITHUB_URL)

# --- NEW: Function to create a realistic time series for forecasting ---
def create_synthetic_timeseries():
    """Generates a synthetic monthly production dataset resembling the example chart."""
    dates = pd.date_range(start='2017-01-01', end='2024-12-31', freq='MS')
    # Create a base trend and strong yearly seasonality
    baseline = np.linspace(110, 100, len(dates)) # Slight downward trend
    seasonality = 40 * np.sin(2 * np.pi * dates.month / 12 - np.pi/2) + \
                  15 * np.sin(4 * np.pi * dates.month / 12)
    noise = np.random.normal(0, 8, len(dates))
    
    quantity = baseline + seasonality + noise
    ts_df = pd.DataFrame({'Month': dates, 'Quantity Produced (t)': quantity})
    return ts_df

# --- NEW: Function to run the Prophet forecast ---
@st.cache_data
def generate_forecast(df):
    """Takes a timeseries dataframe and returns the Prophet forecast."""
    prophet_df = df.rename(columns={'Month': 'ds', 'Quantity Produced (t)': 'y'})
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=365*2) # Forecast 2 years into the future
    forecast = model.predict(future)
    
    return forecast

# --- Part 2: Build the Streamlit UI ---
st.title('Business Performance Dashboard')

# Add the new Forecasting tab
tab_sales, tab_purchases, tab_production, tab_customers_2024, tab_customers_2022, tab_forecasting = st.tabs([
    "Sales", "Purchases", "Production", "Customers 2024", "Customers 2022", "Forecasting"
])

# --- Content for Existing Tabs (Unchanged) ---
with tab_sales:
    st.header("Sales Performance by Item")
    # ... (rest of the tab code is the same)
    if not sales_df.empty:
        sales_melted = sales_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig_sales = px.line(sales_melted, x='Item', y='Value', color='Metric', markers=True, title='Sales Performance by Item')
        st.plotly_chart(fig_sales, use_container_width=True)
    else:
        st.info("Sales data is unavailable or could not be loaded.")

with tab_purchases:
    st.header("Purchases by Item (kg)")
    if not purchases_df.empty:
        purchases_melted = purchases_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig_purchases = px.line(purchases_melted, x='Item', y='Value', color='Metric', markers=True, title='Purchases by Item (kg)')
        st.plotly_chart(fig_purchases, use_container_width=True)
    else:
        st.info("Purchases data is unavailable or could not be loaded.")

with tab_production:
    st.header("Production Metrics by Item")
    if not production_df.empty:
        production_melted = production_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig_production = px.line(production_melted, x='Item', y='Value', color='Metric', markers=True, title='Production Metrics by Item')
        st.plotly_chart(fig_production, use_container_width=True)
    else:
        st.info("Production data is unavailable or could not be loaded.")

with tab_customers_2024:
    st.header("2024 Sales by Product and Customer")
    if not customers_2024_df.empty:
        cust2024_melted = customers_2024_df.melt(id_vars=['Product'], var_name='Customer/Region', value_name='Sales')
        fig_cust2024 = px.bar(cust2024_melted, x='Product', y='Sales', color='Customer/Region', title='2024 Sales by Product and Customer', barmode='group')
        st.plotly_chart(fig_cust2024, use_container_width=True)
    else:
        st.info("Customers 2024 data is unavailable or could not be loaded.")

with tab_customers_2022:
    st.header("2022 Sales by Product")
    if not customers_2022_df.empty:
        cust2022_melted = customers_2022_df.melt(id_vars=['Product'], var_name='Metric', value_name='Value')
        fig_cust2022 = px.bar(cust2022_melted, x='Product', y='Value', color='Metric', title='2022 Sales by Product', barmode='group')
        st.plotly_chart(fig_cust2022, use_container_width=True)
    else:
        st.info("Customers 2022 data is unavailable or could not be loaded.")

# --- NEW: Content for Forecasting Tab ---
with tab_forecasting:
    st.header("Production Quantity Forecasting")
    
    # 1. Create and display the historical data plot
    historical_data = create_synthetic_timeseries()
    fig_hist = px.line(historical_data, x='Month', y='Quantity Produced (t)', markers=True)
    fig_hist.update_traces(line_color='orange')
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.subheader("Prophet Model Forecast")

    # Use a button to trigger the potentially slow model training
    if st.button("Generate 2-Year Forecast"):
        with st.spinner("Training model and generating forecast... This may take a moment."):
            # 2. Generate and display the forecast plot
            forecast_data = generate_forecast(historical_data)
            
            # Merge actuals for plotting
            forecast_plot_df = forecast_data.set_index('ds')[['yhat']].join(historical_data.set_index('Month')['Quantity Produced (t)']).reset_index()
            forecast_plot_df.rename(columns={'ds': 'Date', 'Quantity Produced (t)': 'Actual', 'yhat': 'Prophet Forecast'}, inplace=True)
            
            # The forecast is daily, but actuals are monthly. We need to convert actuals for comparison.
            daily_demand = forecast_plot_df['Actual'] / forecast_plot_df['Date'].dt.days_in_month
            daily_forecast = forecast_plot_df['Prophet Forecast'] / forecast_plot_df['Date'].dt.days_in_month
            
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=forecast_plot_df['Date'], y=daily_demand, mode='lines', name='Actual', line=dict(color='orange')))
            fig_forecast.add_trace(go.Scatter(x=forecast_plot_df['Date'], y=daily_forecast, mode='lines', name='Prophet Forecast', line=dict(color='#d95f02'))) # Darker Orange
            fig_forecast.update_layout(title='Daily Demand vs. Prophet Forecast', xaxis_title='Date', yaxis_title='Demand (t per day)')
            st.plotly_chart(fig_forecast, use_container_width=True)

            # 3. Calculate and display the MAPE plot
            # Calculate MAPE only where actuals exist
            mape_df = forecast_plot_df.dropna().copy()
            mape_df['MAPE'] = np.abs((mape_df['Actual'] - mape_df['Prophet Forecast']) / mape_df['Actual']) * 100
            
            # Calculate a 3-month rolling average of MAPE to smooth the line
            mape_df['Rolling MAPE (%)'] = mape_df['MAPE'].rolling(window=3, min_periods=1).mean()

            fig_mape = px.line(mape_df, x='Date', y='Rolling MAPE (%)', title='3-Month Rolling Average of Model Error (MAPE)')
            fig_mape.update_traces(line_color='green')
            st.plotly_chart(fig_mape, use_container_width=True)
