# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

st.title("📊 Sales Data Dashboard")

# Upload or load default CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("sample_sales_data.csv")  # Replace with your local file path
st.write("📋 Column names:", df.columns.tolist())
# Preprocess
df['Date'] = pd.to_datetime(df['Date'])
df['Revenue'] = df['Quantity'] * df['UnitPrice']

df['Month'] = df['Date'].dt.to_period('M').astype(str)

# Filters
region = st.selectbox("Select Region", ["All"] + sorted(df['Region'].unique().tolist()))
product = st.selectbox("Select Product", ["All"] + sorted(df['Product'].unique().tolist()))

filtered_df = df.copy()
if region != "All":
    filtered_df = filtered_df[filtered_df['Region'] == region]
if product != "All":
    filtered_df = filtered_df[filtered_df['Product'] == product]

st.write("Filtered Data", filtered_df.head())

# Line plot of revenue over time
monthly_revenue = filtered_df.groupby('Month')['Revenue'].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=monthly_revenue, x='Month', y='Revenue', marker='o', ax=ax)
plt.xticks(rotation=45)
plt.title("Monthly Revenue")
st.pyplot(fig)

# Forecasting with Linear Regression
filtered_df['Day'] = filtered_df['Date'].map(pd.Timestamp.toordinal)
X = filtered_df[['Day']]
y = filtered_df['Revenue']
if len(X) > 5:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"📉 RMSE: {rmse:.2f}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(X_test, y_test, color='blue', label='Actual')
    ax2.plot(X_test, y_pred, color='red', label='Predicted')
    ax2.set_title("Forecasted Revenue (Test Set)")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.warning("Not enough data to train a model. Try selecting a different filter.")
