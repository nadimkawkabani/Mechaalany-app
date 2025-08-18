import streamlit as st
import pandas as pd
import plotly.express as px

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
            # **CRITICAL FIX**: Ensure the ID column is of string type to prevent merge/filter issues
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

# Load all the dataframes
sales_df, purchases_df, production_df, customers_2024_df, customers_2022_df = load_data(GITHUB_URL)


# --- Part 2: Sidebar Filters (More Robust Version) ---
st.sidebar.header('Dashboard Filters')

# --- FIX for Item Filter ---
# Create a comprehensive list of all unique 'Item' values from all relevant sheets
all_items = []
if 'Item' in sales_df.columns:
    all_items.extend(sales_df['Item'].dropna().unique())
if 'Item' in purchases_df.columns:
    all_items.extend(purchases_df['Item'].dropna().unique())
if 'Item' in production_df.columns:
    all_items.extend(production_df['Item'].dropna().unique())
# Get the final unique list and sort it
all_items = sorted(list(set(all_items)))

selected_items = st.sidebar.multiselect(
    'Select Items:',
    options=all_items,
    default=all_items[:5] if all_items else [] # Default to first 5, or empty if no items
)

# --- FIX for Product Filter ---
all_products = []
if 'Product' in customers_2024_df.columns:
    all_products.extend(customers_2024_df['Product'].dropna().unique())
if 'Product' in customers_2022_df.columns:
    all_products.extend(customers_2022_df['Product'].dropna().unique())
all_products = sorted(list(set(all_products)))

selected_products = st.sidebar.multiselect(
    'Select Products:',
    options=all_products,
    default=all_products[:5] if all_products else []
)

# --- Part 3: Main Dashboard Layout ---
st.title('Business Performance Dashboard')

# The filtering logic itself was correct, the issue was in the list creation.
# This section remains largely the same.
sales_df_filtered = sales_df[sales_df['Item'].isin(selected_items)] if selected_items else sales_df
purchases_df_filtered = purchases_df[purchases_df['Item'].isin(selected_items)] if selected_items else purchases_df
production_df_filtered = production_df[production_df['Item'].isin(selected_items)] if selected_items else production_df
customers_2024_df_filtered = customers_2024_df[customers_2024_df['Product'].isin(selected_products)] if selected_products else customers_2024_df
customers_2022_df_filtered = customers_2022_df[customers_2022_df['Product'].isin(selected_products)] if selected_products else customers_2022_df

# Create tabs
tab_sales, tab_purchases, tab_production, tab_customers_2024, tab_customers_2022 = st.tabs([
    "Sales", "Purchases", "Production", "Customers 2024", "Customers 2022"
])

# --- Content for Each Tab ---
with tab_sales:
    st.header("Sales Performance")
    # Using a list comprehension to find available metrics is safer
    available_metrics = [col for col in sales_df.columns if col != 'Item']
    sales_metrics = st.multiselect(
        'Select Sales Metrics to Display:',
        options=available_metrics,
        default=[m for m in ['Sum of NET', 'Sum of QTY'] if m in available_metrics]
    )
    if not sales_df_filtered.empty and sales_metrics:
        sales_melted = sales_df_filtered.melt(id_vars=['Item'], value_vars=sales_metrics, var_name='Metric', value_name='Value')
        fig_sales = px.line(sales_melted, x='Item', y='Value', color='Metric', markers=True, title='Sales Performance by Selected Items')
        st.plotly_chart(fig_sales, use_container_width=True)
    else:
        st.info("No data to display. Please select items from the sidebar and metrics above.")

with tab_purchases:
    st.header("Purchases by Item (kg)")
    if not purchases_df_filtered.empty:
        purchases_melted = purchases_df_filtered.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig_purchases = px.line(purchases_melted, x='Item', y='Value', color='Metric', markers=True, title='Purchases by Selected Items (kg)')
        st.plotly_chart(fig_purchases, use_container_width=True)
    else:
        st.info("No data to display for the selected items. Please select items from the sidebar.")

with tab_production:
    st.header("Production Metrics by Item")
    if not production_df_filtered.empty:
        production_melted = production_df_filtered.melt(id_vars=['Item'], var_name='Metric', value_name='Value')
        fig_production = px.line(production_melted, x='Item', y='Value', color='Metric', markers=True, title='Production Metrics by Selected Items')
        st.plotly_chart(fig_production, use_container_width=True)
    else:
        st.info("No data to display for the selected items. Please select items from the sidebar.")

with tab_customers_2024:
    st.header("2024 Sales by Product and Customer")
    if not customers_2024_df_filtered.empty:
        cust2024_melted = customers_2024_df_filtered.melt(id_vars=['Product'], var_name='Customer/Region', value_name='Sales')
        fig_cust2024 = px.bar(cust2024_melted, x='Product', y='Sales', color='Customer/Region', title='2024 Sales by Selected Products and Customer', barmode='group')
        st.plotly_chart(fig_cust2024, use_container_width=True)
    else:
        st.info("No data to display. Please select products from the sidebar.")

with tab_customers_2022:
    st.header("2022 Sales by Product")
    if not customers_2022_df_filtered.empty:
        cust2022_melted = customers_2022_df_filtered.melt(id_vars=['Product'], var_name='Metric', value_name='Value')
        fig_cust2022 = px.bar(cust2022_melted, x='Product', y='Value', color='Metric', title='2022 Sales by Selected Products', barmode='group')
        st.plotly_chart(fig_cust2022, use_container_width=True)
    else:
        st.info("No data to display. Please select products from the sidebar.")
