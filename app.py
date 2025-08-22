import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Part 0: Page Configuration ---
st.set_page_config(layout="wide")


# --- Part 1: Data Loading & Generation ---
# All data loading and generation functions are placed here for clarity.
@st.cache_data
def load_bi_data(url):
    # ... (This function is correct and does not need changes) ...
    try:
        xls = pd.ExcelFile(url)
    except Exception as e:
        st.error(f"Failed to load the Excel file from the URL. Please check the link. Error: {e}")
        return tuple(pd.DataFrame() for _ in range(5))
    def clean_sheet(sheet_name, id_col_name, header_row=0, item_prefix=None):
        try:
            df = pd.read_excel(xls, sheet_name, header=header_row)
            df.rename(columns={df.columns[0]: id_col_name}, inplace=True)
            if item_prefix and id_col_name in df.columns:
                df = df[df[id_col_name].str.startswith(item_prefix, na=False)].copy()
            if id_col_name in df.columns:
                df[id_col_name] = df[id_col_name].astype(str)
            return df
        except Exception: return pd.DataFrame()
    sales_df = clean_sheet("Sales", "Item", header_row=2, item_prefix='407002-')
    purchases_df = clean_sheet("Purchases_kg", "Item")
    production_df = clean_sheet("Production", "Item")
    customers_2024_df = clean_sheet("Customers_2024", "Product")
    customers_2022_df = clean_sheet("Customers_2022", "Product")
    return sales_df, purchases_df, production_df, customers_2024_df, customers_2022_df

@st.cache_data
def create_synthetic_timeseries():
    # ... (This function is correct and does not need changes) ...
    dates = pd.date_range(start='2017-01-01', end='2023-12-31', freq='D')
    baseline = np.linspace(4.5, 4.2, len(dates))
    yearly_seasonality = 0.5 * np.sin(2 * np.pi * dates.dayofyear / 365.25 - np.pi/2)
    weekly_seasonality = 0.2 * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise = np.random.normal(0, 0.2, len(dates))
    demand = baseline + yearly_seasonality + weekly_seasonality + noise
    ts_df = pd.DataFrame({'Date': dates, 'Demand (t per day)': demand})
    monthly_df = ts_df.resample('MS', on='Date').sum().reset_index()
    monthly_df.rename(columns={'Date': 'Month', 'Demand (t per day)': 'Quantity Produced (t)'}, inplace=True)
    return monthly_df, ts_df

@st.cache_data
def generate_forecast(df):
    # ... (This function is correct and does not need changes) ...
    prophet_df = df.rename(columns={'Date': 'ds', 'Demand (t per day)': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=365*2)
    forecast = model.predict(future)
    return forecast

# --- NEW: Enhanced Data Generation for Interactive Optimization Tab ---
@st.cache_data
def get_scenario_data():
    """Creates a more detailed DataFrame for interactive scenario analysis."""
    data = {
        'Scenario': ['Base', 'Labor Shift', 'Season Extension'],
        'Tonnage (t)': [1500, 2000, 2600],
        'Revenue (USD)': [480000, 640000, 832000],
        'Incremental Cost (USD)': [0, 70000, 105000],
        'Labor Cost': [150000, 170000, 190000],
        'Salt Cost': [50000, 65000, 85000],
        'Energy Cost': [80000, 100000, 130000],
        'OEE (%)': [92.5, 91.8, 93.2],
        'Labor Constraint Value': [15, 18, 20],
        'Brine Constraint Value': [8, 6, 5],
        'Calendar Constraint Value': [10, 8, 7]
    }
    return pd.DataFrame(data)

# --- Load all data ---
GITHUB_URL = "https://raw.githubusercontent.com/nadimkawkabani/projectsstudents-app/main/Dashboard.xlsx"
sales_df, purchases_df, production_df, customers_2024_df, customers_2022_df = load_bi_data(GITHUB_URL)
monthly_production_hist, daily_demand_hist = create_synthetic_timeseries()
scenario_df = get_scenario_data()

# --- Part 2: Main App Layout ---
st.title('Business Performance Dashboard')

tab_list = ["Sales", "Purchases", "Production", "Customers 2024", "Customers 2022", "Forecasting", "Optimization & Scenarios"]
tab_sales, tab_purchases, tab_production, tab_cust24, tab_cust22, tab_forecast, tab_opt = st.tabs(tab_list)

# --- BI and Forecasting Tabs (No changes) ---
with tab_sales: #...
    st.header("Sales Performance by Item")
    if not sales_df.empty:
        sales_melted = sales_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig = px.line(sales_melted, x='Item', y='Value', color='Metric', markers=True)
        st.plotly_chart(fig, use_container_width=True, key="sales_chart")
    else: st.info("Sales data is unavailable.")
with tab_purchases: #...
    st.header("Purchases by Item (kg)")
    if not purchases_df.empty:
        purchases_melted = purchases_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig = px.line(purchases_melted, x='Item', y='Value', color='Metric', markers=True)
        st.plotly_chart(fig, use_container_width=True, key="purchases_chart")
    else: st.info("Purchases data is unavailable.")
with tab_production: #...
    st.header("Production Metrics by Item")
    if not production_df.empty:
        production_melted = production_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig = px.line(production_melted, x='Item', y='Value', color='Metric', markers=True)
        st.plotly_chart(fig, use_container_width=True, key="production_chart")
    else: st.info("Production data is unavailable.")
with tab_cust24: #...
    st.header("2024 Sales by Product and Customer")
    if not customers_2024_df.empty:
        melted = customers_2024_df.melt(id_vars=['Product'], var_name='Customer/Region', value_name='Sales')
        fig = px.bar(melted, x='Product', y='Sales', color='Customer/Region', barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="cust2024_chart")
    else: st.info("Customers 2024 data is unavailable.")
with tab_cust22: #...
    st.header("2022 Sales by Product")
    if not customers_2022_df.empty:
        melted = customers_2022_df.melt(id_vars=['Product'], var_name='Metric', value_name='Value')
        fig = px.bar(melted, x='Product', y='Value', color='Metric', barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="cust2022_chart")
    else: st.info("Customers 2022 data is unavailable.")
with tab_forecast: #...
    st.header("Demand Forecasting")
    st.subheader("Historical Monthly Production")
    fig_hist = px.line(monthly_production_hist, x='Month', y='Quantity Produced (t)', markers=True)
    fig_hist.update_traces(line_color='orange')
    st.plotly_chart(fig_hist, use_container_width=True, key="forecast_hist_chart")
    st.markdown("---")
    st.header("Prophet Model Forecast")
    if st.button("Generate 2-Year Forecast"):
        with st.spinner("Training model..."):
            forecast_data = generate_forecast(daily_demand_hist)
            st.subheader("Daily Demand vs. Prophet Forecast")
            col1, col2 = st.columns(2)
            show_actual = col1.checkbox("Show Actual Demand", value=True)
            show_forecast = col2.checkbox("Show Prophet Forecast", value=True)
            fig_forecast = go.Figure()
            if show_actual: fig_forecast.add_trace(go.Scatter(x=daily_demand_hist['Date'], y=daily_demand_hist['Demand (t per day)'], mode='lines', name='Actual', line=dict(color='orange', width=1.5)))
            if show_forecast: fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], mode='lines', name='Prophet Forecast', line=dict(color='#d95f02', width=1.5)))
            fig_forecast.update_layout(xaxis_title='Date', yaxis_title='Demand (t per day)', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            if fig_forecast.data: st.plotly_chart(fig_forecast, use_container_width=True, key="forecast_vs_actual_chart")
            else: st.info("Please select a line to display.")
            if show_actual and show_forecast:
                st.subheader("Model Error (Rolling MAPE)")
                results_df = pd.merge(daily_demand_hist, forecast_data[['ds', 'yhat']], left_on='Date', right_on='ds')
                results_df['MAPE'] = np.abs((results_df['Demand (t per day)'] - results_df['yhat']) / results_df['Demand (t per day)']) * 100
                results_df['Rolling MAPE (%)'] = results_df['MAPE'].rolling(window=30, min_periods=1).mean()
                fig_mape = px.line(results_df, x='Date', y='Rolling MAPE (%)')
                fig_mape.update_traces(line_color='green')
                st.plotly_chart(fig_mape, use_container_width=True, key="mape_chart")

# --- NEW: Fully Interactive Optimization & Scenarios Tab ---
with tab_opt:
    st.header("Optimization & Scenario Analysis")

    # The main controller for the tab
    selected_scenario = st.selectbox(
        "Select a Scenario to Analyze:",
        options=scenario_df['Scenario'],
        key="scenario_select"
    )
    
    # Filter all data to the selected scenario
    scenario_data = scenario_df[scenario_df['Scenario'] == selected_scenario].iloc[0]

    st.markdown(f"### Key Metrics for: **{selected_scenario}**")
    
    # KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Tonnage (t)", f"{scenario_data['Tonnage (t)']:,}")
    kpi2.metric("Revenue (USD)", f"${scenario_data['Revenue (USD)']:,}")
    kpi3.metric("Incremental Cost (USD)", f"${scenario_data['Incremental Cost (USD)']:,}")
    kpi4.metric("OEE (%)", f"{scenario_data['OEE (%)']:.1f}%")
    
    st.markdown("---")

    # Create two columns for the main visuals
    viz1, viz2 = st.columns(2)

    with viz1:
        st.subheader("Financial Breakdown")
        # Create a dynamic bar chart for costs vs. revenue
        financial_breakdown = pd.DataFrame({
            'Category': ['Revenue', 'Labor Cost', 'Salt Cost', 'Energy Cost'],
            'Amount (USD)': [
                scenario_data['Revenue (USD)'],
                -scenario_data['Labor Cost'], # Costs are negative
                -scenario_data['Salt Cost'],
                -scenario_data['Energy Cost']
            ]
        })
        
        fig_financial = px.bar(
            financial_breakdown, 
            x='Category', 
            y='Amount (USD)',
            color='Category',
            color_discrete_map={'Revenue': 'green', 'Labor Cost': 'red', 'Salt Cost': 'indianred', 'Energy Cost': 'darkred'},
            title=f"Financials for {selected_scenario} Scenario"
        )
        st.plotly_chart(fig_financial, use_container_width=True)

        st.subheader("Constraint Analysis (Shadow Prices)")
        # Bar chart for constraints
        shadow_prices = pd.DataFrame({
            'Constraint': ['Labour', 'Brine Capacity', 'Calendar'],
            'Marginal Value (USD per unit)': [
                scenario_data['Labor Constraint Value'],
                scenario_data['Brine Constraint Value'],
                scenario_data['Calendar Constraint Value']
            ]
        })
        fig_shadow = px.bar(shadow_prices, x='Constraint', y='Marginal Value (USD per unit)', 
                                color_discrete_sequence=['purple'], title=f"Constraint Values for {selected_scenario}")
        st.plotly_chart(fig_shadow, use_container_width=True)

    with viz2:
        st.subheader("OEE Performance Gauge")
        # Create a dynamic gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = scenario_data['OEE (%)'],
            title = {'text': f"OEE for {selected_scenario}"},
            gauge = {
                'axis': {'range': [85, 100]},
                'steps': [
                    {'range': [85, 90], 'color': "lightgray"},
                    {'range': [90, 95], 'color': "gray"}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 98}
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("Overall Scenario Comparison")
        # Bar chart comparing tonnage across all scenarios, highlighting the selected one
        tonnage_df = scenario_df[['Scenario', 'Tonnage (t)']].copy()
        tonnage_df['Color'] = np.where(tonnage_df['Scenario'] == selected_scenario, 'orange', 'grey')
        
        fig_tonnage_comp = px.bar(
            tonnage_df, 
            x='Scenario', 
            y='Tonnage (t)', 
            title="Tonnage Comparison Across Scenarios"
        )
        fig_tonnage_comp.update_traces(marker_color=tonnage_df['Color'])
        st.plotly_chart(fig_tonnage_comp, use_container_width=True)
