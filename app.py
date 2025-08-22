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
    """Loads all BI data from the specified Excel file URL."""
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
        except Exception:
            return pd.DataFrame()

    sales_df = clean_sheet("Sales", "Item", header_row=2, item_prefix='407002-')
    purchases_df = clean_sheet("Purchases_kg", "Item")
    production_df = clean_sheet("Production", "Item")
    customers_2024_df = clean_sheet("Customers_2024", "Product")
    customers_2022_df = clean_sheet("Customers_2022", "Product")
        
    return sales_df, purchases_df, production_df, customers_2024_df, customers_2022_df

@st.cache_data
def create_synthetic_timeseries():
    """Generates synthetic daily and monthly data for forecasting."""
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
    """Generates Prophet forecast."""
    # ... (This function is correct and does not need changes) ...
    prophet_df = df.rename(columns={'Date': 'ds', 'Demand (t per day)': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=365*2)
    forecast = model.predict(future)
    return forecast

# --- NEW: Data Generation for Optimization Tab ---
@st.cache_data
def get_oee_data():
    """Generates synthetic OEE data."""
    np.random.seed(42)
    oee_data = pd.DataFrame({
        'Quantity Produced t': np.random.normal(121.8, 25.7, 100),
        'Production Loss %': np.random.normal(7, 1.2, 100).clip(5, 8.9),
        'Approx OEE %': np.random.normal(93, 1.2, 100).clip(91.1, 95)
    })
    return oee_data

@st.cache_data
def get_scenario_data():
    """Creates DataFrames for scenario analysis visuals."""
    # Data for Tonnage bar chart and main table
    scenario_df = pd.DataFrame({
        'Scenario': ['Base', 'Labor Shift', 'Season Extension'],
        'Tonnage (t)': [1500, 2000, 2600],
        'Labor Hours': [14000, 16500, 19000],
        'Diversion Ratio': [0.20, 0.15, 0.10],
        'Overtime Hrs': [0, 1200, 1800],
        'Incremental Cost (USD)': [0, 70000, 105000],
        'Revenue (USD)': [480000, 640000, 832000]
    })

    # Data for Box Plot
    np.random.seed(0)
    box_plot_data = pd.DataFrame({
        'Base': np.random.normal(1500, 50, 100),
        'Labor Shift': np.random.normal(2000, 100, 100),
        'Season Extension': np.random.normal(2600, 150, 100)
    }).melt(var_name='Scenario', value_name='Tonnage (t)')

    return scenario_df, box_plot_data

# --- Load all data ---
GITHUB_URL = "https://raw.githubusercontent.com/nadimkawkabani/projectsstudents-app/main/Dashboard.xlsx"
sales_df, purchases_df, production_df, customers_2024_df, customers_2022_df = load_bi_data(GITHUB_URL)
monthly_production_hist, daily_demand_hist = create_synthetic_timeseries()
oee_data = get_oee_data()
scenario_df, scenario_box_plot_data = get_scenario_data()


# --- Part 2: Main App Layout ---
st.title('Business Performance Dashboard')

# Add the new "Optimization & Scenarios" tab
tab_list = [
    "Sales", "Purchases", "Production", 
    "Customers 2024", "Customers 2022", 
    "Forecasting", "Optimization & Scenarios"
]
tab_sales, tab_purchases, tab_production, tab_cust24, tab_cust22, tab_forecast, tab_opt = st.tabs(tab_list)

# --- BI Tabs (No changes) ---
with tab_sales:
    # ...
    st.header("Sales Performance by Item")
    if not sales_df.empty:
        sales_melted = sales_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig = px.line(sales_melted, x='Item', y='Value', color='Metric', markers=True)
        st.plotly_chart(fig, use_container_width=True, key="sales_chart")
    else: st.info("Sales data is unavailable.")
with tab_purchases:
    # ...
    st.header("Purchases by Item (kg)")
    if not purchases_df.empty:
        purchases_melted = purchases_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig = px.line(purchases_melted, x='Item', y='Value', color='Metric', markers=True)
        st.plotly_chart(fig, use_container_width=True, key="purchases_chart")
    else: st.info("Purchases data is unavailable.")
with tab_production:
    # ...
    st.header("Production Metrics by Item")
    if not production_df.empty:
        production_melted = production_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig = px.line(production_melted, x='Item', y='Value', color='Metric', markers=True)
        st.plotly_chart(fig, use_container_width=True, key="production_chart")
    else: st.info("Production data is unavailable.")
with tab_cust24:
    # ...
    st.header("2024 Sales by Product and Customer")
    if not customers_2024_df.empty:
        melted = customers_2024_df.melt(id_vars=['Product'], var_name='Customer/Region', value_name='Sales')
        fig = px.bar(melted, x='Product', y='Sales', color='Customer/Region', barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="cust2024_chart")
    else: st.info("Customers 2024 data is unavailable.")
with tab_cust22:
    # ...
    st.header("2022 Sales by Product")
    if not customers_2022_df.empty:
        melted = customers_2022_df.melt(id_vars=['Product'], var_name='Metric', value_name='Value')
        fig = px.bar(melted, x='Product', y='Value', color='Metric', barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="cust2022_chart")
    else: st.info("Customers 2022 data is unavailable.")
# --- Forecasting Tab (No changes) ---
with tab_forecast:
    # ...
    st.header("Demand Forecasting")
    st.subheader("Historical Monthly Production")
    fig_hist = px.line(monthly_production_hist, x='Month', y='Quantity Produced (t)', markers=True)
    fig_hist.update_traces(line_color='orange')
    st.plotly_chart(fig_hist, use_container_width=True, key="forecast_hist_chart")
    st.markdown("---")
    st.header("Prophet Model Forecast")
    if st.button("Generate 2-Year Forecast"):
        with st.spinner("Training model and generating forecast..."):
            forecast_data = generate_forecast(daily_demand_hist)
            st.subheader("Daily Demand vs. Prophet Forecast")
            col1, col2 = st.columns(2)
            show_actual = col1.checkbox("Show Actual Demand", value=True)
            show_forecast = col2.checkbox("Show Prophet Forecast", value=True)
            fig_forecast = go.Figure()
            if show_actual:
                fig_forecast.add_trace(go.Scatter(x=daily_demand_hist['Date'], y=daily_demand_hist['Demand (t per day)'], mode='lines', name='Actual', line=dict(color='orange', width=1.5)))
            if show_forecast:
                fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], mode='lines', name='Prophet Forecast', line=dict(color='#d95f02', width=1.5)))
            fig_forecast.update_layout(xaxis_title='Date', yaxis_title='Demand (t per day)', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            if fig_forecast.data:
                st.plotly_chart(fig_forecast, use_container_width=True, key="forecast_vs_actual_chart")
            else:
                st.info("Please select at least one line to display using the checkboxes above.")
            if show_actual and show_forecast:
                st.subheader("Model Error (Rolling MAPE)")
                results_df = pd.merge(daily_demand_hist, forecast_data[['ds', 'yhat']], left_on='Date', right_on='ds')
                results_df['MAPE'] = np.abs((results_df['Demand (t per day)'] - results_df['yhat']) / results_df['Demand (t per day)']) * 100
                results_df['Rolling MAPE (%)'] = results_df['MAPE'].rolling(window=30, min_periods=1).mean()
                fig_mape = px.line(results_df, x='Date', y='Rolling MAPE (%)')
                fig_mape.update_traces(line_color='green')
                st.plotly_chart(fig_mape, use_container_width=True, key="mape_chart")
# --- NEW: Content for Optimization & Scenarios Tab ---
with tab_opt:
    st.header("Optimization & Scenario Analysis")
    
    # Section 1: OEE Analysis
    st.subheader("Overall Equipment Effectiveness (OEE) Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Descriptive Statistics")
        # Display the descriptive statistics table
        st.dataframe(oee_data.describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].rename(index={
            'mean': 'Mean', 'std': 'Standard deviation', 'min': 'Minimum', '25%': '25th percentile',
            '50%': 'Median', '75%': '75th percentile', 'max': 'Maximum'
        }))
        st.latex(r'''OEE = \frac{Run \ time}{Planned \ Production \ time} \times \frac{Ideal \ Cycle \ Time \times Total \ Count}{Run \ Time} \times \frac{Good \ Count}{Total \ Count}''')

    with col2:
        st.markdown("#### OEE Distribution")
        # Display the OEE histogram
        fig_oee = px.histogram(oee_data, x='Approx OEE %', nbins=10, title="Frequency of OEE values")
        fig_oee.update_traces(marker_color='orange', marker_line_color='black', marker_line_width=1.5)
        st.plotly_chart(fig_oee, use_container_width=True)

    st.markdown("---")

    # Section 2: Scenario Modeling
    st.subheader("Scenario Modeling and Financial Impact")
    st.markdown("#### Optimization Goal")
    st.latex(r'''\max \sum_{d=1}^{D} Shift\_Hours_{d} \times Filler_{Rate} - \lambda \times Overtime_{d}''')

    # Interactive Scenario Selection
    selected_scenario = st.selectbox(
        "Select a Scenario to Analyze:",
        options=scenario_df['Scenario']
    )
    
    # Filter data for the selected scenario
    scenario_data_selected = scenario_df[scenario_df['Scenario'] == selected_scenario].iloc[0]

    # Display metrics for the selected scenario
    st.markdown(f"#### Results for: **{selected_scenario}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tonnage (t)", f"{scenario_data_selected['Tonnage (t)']:,}")
    col2.metric("Incremental Cost (USD)", f"${scenario_data_selected['Incremental Cost (USD)']:,}")
    col3.metric("Revenue (USD)", f"${scenario_data_selected['Revenue (USD)']:,}")
    
    # Display charts for all scenarios
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Tonnage Comparison")
        fig_tonnage = px.bar(scenario_df, x='Scenario', y='Tonnage (t)', color='Scenario', 
                             color_discrete_map={'Base': 'steelblue', 'Labor Shift': 'orange', 'Season Extension': 'green'})
        st.plotly_chart(fig_tonnage, use_container_width=True)
    with col2:
        st.markdown("##### Tonnage Distribution (Box Plot)")
        fig_box = px.box(scenario_box_plot_data, x='Scenario', y='Tonnage (t)', color='Scenario',
                         color_discrete_map={'Base': 'orange', 'Labor Shift': 'orange', 'Season Extension': 'orange'})
        st.plotly_chart(fig_box, use_container_width=True)

    # Display the financial comparison table
    st.markdown("#### Financial Comparison of Scenarios")
    financial_df = scenario_df[['Scenario', 'Incremental Cost (USD)', 'Revenue (USD)']].copy()
    financial_df['Incremental Revenue (USD)'] = financial_df['Revenue (USD)'] - financial_df.loc[0, 'Revenue (USD)']
    financial_df['Incremental Profit (USD)'] = financial_df['Incremental Revenue (USD)'] - financial_df['Incremental Cost (USD)']
    financial_df['ROI (%)'] = (financial_df['Incremental Profit (USD)'] / financial_df['Incremental Cost (USD)']).replace([np.inf, -np.inf], 0) * 100
    st.dataframe(financial_df.style.format({
        'Incremental Cost (USD)': '${:,.0f}', 'Revenue (USD)': '${:,.0f}',
        'Incremental Revenue (USD)': '${:,.0f}', 'Incremental Profit (USD)': '${:,.0f}',
        'ROI (%)': '{:.0f}%'
    }))

    st.markdown("---")

    # Section 3: Sensitivity & Constraint Analysis
    st.subheader("Sensitivity and Constraint Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Impact of Cost Drivers on NPV")
        sensitivity_data = pd.DataFrame({
            'Factor': ['Energy Price', 'Salt Cost', 'Labour Cost', 'Forecast Variance'],
            'Impact on NPV (USD)': [30000, 18000, 15000, 8000]
        })
        fig_sens = px.bar(sensitivity_data, y='Factor', x='Impact on NPV (USD)', orientation='h', color_discrete_sequence=['orange'])
        st.plotly_chart(fig_sens, use_container_width=True)
    with col2:
        st.markdown("##### Shadow Prices of Binding Constraints")
        shadow_prices = pd.DataFrame({
            'Constraint': ['Labour', 'Brine Capacity', 'Calendar'],
            'Marginal Value (USD per unit)': [20, 5, 7]
        })
        fig_shadow = px.bar(shadow_prices, x='Constraint', y='Marginal Value (USD per unit)', color_discrete_sequence=['purple'])
        st.plotly_chart(fig_shadow, use_container_width=True)
