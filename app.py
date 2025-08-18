import streamlit as st
import pandas as pd
import plotly.express as px

# --- Part 0: Page Configuration ---
st.set_page_config(layout="wide")

# --- Part 1: Load and Clean ALL Data from GitHub ---
# **CRITICAL CHANGE**: Replace this URL with the RAW URL of your Excel file from GitHub
GITHUB_URL = "https://raw.githubusercontent.com/nadimkawkabani/projectsstudents-app/main/Dashboard.xlsx"
@st.cache_data
def load_data(url):
    """Loads all data from the specified Excel file URL and returns cleaned dataframes."""
    try:
        # We read the Excel file directly from the URL
        xls = pd.ExcelFile(url)
    except Exception as e:
        st.error(f"Failed to load the Excel file from the URL. Please check the link. Error: {e}")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # 1. Sales Data
    try:
        sales_df = pd.read_excel(xls, "Sales", header=2)
        sales_df.rename(columns={sales_df.columns[0]: "Item"}, inplace=True)
        sales_df = sales_df[sales_df["Item"].str.startswith('407002-', na=False)].copy()
    except Exception as e:
        st.warning(f"Could not load or process 'Sales' sheet: {e}")
        sales_df = pd.DataFrame() 

    # 2. Purchases Data
    try:
        purchases_df = pd.read_excel(xls, "Purchases_kg", header=0)
        purchases_df.rename(columns={purchases_df.columns[0]: 'Item'}, inplace=True)
    except Exception as e:
        st.warning(f"Could not load or process 'Purchases_kg' sheet: {e}")
        purchases_df = pd.DataFrame()

    # 3. Production Data
    try:
        production_df = pd.read_excel(xls, "Production", header=0)
        production_df.rename(columns={production_df.columns[0]: 'Item'}, inplace=True)
    except Exception as e:
        st.warning(f"Could not load or process 'Production' sheet: {e}")
        production_df = pd.DataFrame()

    # 4. Customers 2024 Data
    try:
        customers_2024_df = pd.read_excel(xls, "Customers_2024", header=0)
        customers_2024_df.rename(columns={customers_2024_df.columns[0]: 'Product'}, inplace=True)
    except Exception as e:
        st.warning(f"Could not load or process 'Customers_2024' sheet: {e}")
        customers_2024_df = pd.DataFrame()

    # 5. Customers 2022 Data
    try:
        customers_2022_df = pd.read_excel(xls, "Customers_2022", header=0)
        customers_2022_df.rename(columns={customers_2022_df.columns[0]: 'Product'}, inplace=True)
    except Exception as e:
        st.warning(f"Could not load or process 'Customers_2022' sheet: {e}")
        customers_2022_df = pd.DataFrame()
        
    return sales_df, purchases_df, production_df, customers_2024_df, customers_2022_df

# Load all the dataframes using the GitHub URL
sales_df, purchases_df, production_df, customers_2024_df, customers_2022_df = load_data(GITHUB_URL)


# --- Part 2: Build the Streamlit UI ---
st.title('Business Performance Dashboard')

tab_sales, tab_purchases, tab_production, tab_customers_2024, tab_customers_2022 = st.tabs([
    "Sales", "Purchases", "Production", "Customers 2024", "Customers 2022"
])

# ... (The rest of the UI code is exactly the same as before) ...

# --- Content for Sales Tab ---
with tab_sales:
    st.header("Sales Performance by Item")
    if not sales_df.empty:
        sales_melted = sales_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig_sales = px.line(sales_melted, x='Item', y='Value', color='Metric', markers=True, title='Sales Performance by Item')
        st.plotly_chart(fig_sales, use_container_width=True)
    else:
        st.info("Sales data is unavailable or could not be loaded.")

# --- Content for Purchases Tab ---
with tab_purchases:
    st.header("Purchases by Item (kg)")
    if not purchases_df.empty:
        purchases_melted = purchases_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig_purchases = px.line(purchases_melted, x='Item', y='Value', color='Metric', markers=True, title='Purchases by Item (kg)')
        st.plotly_chart(fig_purchases, use_container_width=True)
    else:
        st.info("Purchases data is unavailable or could not be loaded.")

# --- Content for Production Tab ---
with tab_production:
    st.header("Production Metrics by Item")
    if not production_df.empty:
        production_melted = production_df.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig_production = px.line(production_melted, x='Item', y='Value', color='Metric', markers=True, title='Production Metrics by Item')
        st.plotly_chart(fig_production, use_container_width=True)
    else:
        st.info("Production data is unavailable or could not be loaded.")

# --- Content for Customers 2024 Tab ---
with tab_customers_2024:
    st.header("2024 Sales by Product and Customer")
    if not customers_2024_df.empty:
        cust2024_melted = customers_2024_df.melt(id_vars=['Product'], var_name='Customer/Region', value_name='Sales')
        fig_cust2024 = px.bar(cust2024_melted, x='Product', y='Sales', color='Customer/Region', title='2024 Sales by Product and Customer', barmode='group')
        st.plotly_chart(fig_cust2024, use_container_width=True)
    else:
        st.info("Customers 2024 data is unavailable or could not be loaded.")

# --- Content for Customers 2022 Tab ---
with tab_customers_2022:
    st.header("2022 Sales by Product")
    if not customers_2022_df.empty:
        cust2022_melted = customers_2022_df.melt(id_vars=['Product'], var_name='Metric', value_name='Value')
        fig_cust2022 = px.bar(cust2022_melted, x='Product', y='Value', color='Metric', title='2022 Sales by Product', barmode='group')
        st.plotly_chart(fig_cust2022, use_container_width=True)
    else:
        st.info("Customers 2022 data is unavailable or could not be loaded.")
