import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset (replace with your actual data path)
df = pd.read_csv("sample_sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Revenue'] = df['Quantity'] * df['UnitPrice']

st.set_page_config(page_title="📊 Sales Dashboard", layout="wide")
st.title("📈 Interactive Sales Dashboard")

# Sidebar Controls
st.sidebar.title("🎛️ Dashboard Controls")

# Date Range Filter
start_date, end_date = st.sidebar.date_input(
    "📅 Select Date Range:",
    [df['Date'].min(), df['Date'].max()]
)
df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Chart Selector
chart_type = st.sidebar.selectbox(
    "📉 Select Chart Type:",
    ["Revenue Over Time", "Units Sold by Product", "Revenue by Category"]
)

# Display Chart
st.subheader(f"Chart: {chart_type}")
if chart_type == "Revenue Over Time":
    chart_data = df_filtered.groupby('Date')["Revenue"].sum()
    st.line_chart(chart_data)

elif chart_type == "Units Sold by Product":
    chart_data = df_filtered.groupby("Product")["Units_Sold"].sum().sort_values(ascending=False)
    st.bar_chart(chart_data)

elif chart_type == "Revenue by Category":
    chart_data = df_filtered.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    st.bar_chart(chart_data)

# Download Button
st.markdown("---")
st.subheader("💾 Download Filtered Data")
st.download_button(
    label="Download CSV",
    data=df_filtered.to_csv(index=False).encode('utf-8'),
    file_name="filtered_sales_data.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
