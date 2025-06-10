import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Streamlit page setup
st.set_page_config(page_title="Suicide Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/nadimkawkabani/Desktop/Healthcare Analytics/who_suicide_statistics.csv")
    df.dropna(inplace=True)
    return df

df = load_data()

# Sidebar filters
st.sidebar.title("🔎 Filters")
selected_year = st.sidebar.selectbox("Select Year", sorted(df["year"].unique()))
selected_gender = st.sidebar.selectbox("Select Gender", df["sex"].unique())
selected_age = st.sidebar.selectbox("Select Age Group", df["age"].unique())

# Apply filters
filtered_df = df[
    (df["year"] == selected_year) &
    (df["sex"] == selected_gender) &
    (df["age"] == selected_age)
]

# Display summary
st.title("📊 Suicide Rates Dashboard")
st.markdown(f"""
Showing data for:
- **Year**: {selected_year}  
- **Gender**: {selected_gender}  
- **Age Group**: {selected_age}
""")

st.subheader("📌 Summary Statistics")
col1, col2 = st.columns(2)
col1.metric("Total Suicides", int(filtered_df["suicides_no"].sum()))
col2.metric("Countries in Selection", filtered_df["country"].nunique())

# Top 10 countries by suicides
st.subheader("🌍 Top 10 Countries by Suicide Count (Filtered)")
top_countries = filtered_df.groupby("country")["suicides_no"].sum().sort_values(ascending=False).head(10)

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette="Blues_r", ax=ax1)
ax1.set_title(f"Top 10 Countries ({selected_year}, {selected_gender}, {selected_age})")
ax1.set_xlabel("Number of Suicides")
st.pyplot(fig1)

# Global trend over time (unfiltered)
st.subheader("📈 Global Suicide Trend (1985–2016)")
trend_df = df.groupby("year")["suicides_no"].sum().reset_index()
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=trend_df, x="year", y="suicides_no", marker="o", ax=ax2)
ax2.set_title("Total Suicides Globally Over Time")
ax2.set_ylabel("Total Suicides")
st.pyplot(fig2)

# Optional: download filtered data
st.subheader("🗃 Export Filtered Data")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered Data as CSV", csv, "filtered_data.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("📌 Data source: WHO Suicide Statistics via Kaggle")
