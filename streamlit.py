import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import kpss
import seaborn as sns

# Title
st.title("Sales Forecasting with Linear Regression")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Function to process the uploaded file
def process_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Check for necessary columns
        if all(col in df.columns for col in ['UnitPrice', 'Quantity', 'InvoiceDate']):
            # Calculate Sales
            df['Sales'] = df['UnitPrice'] * df['Quantity']
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])   # Convert to datetime
            df.set_index('InvoiceDate', inplace=True)

            # Ensure the index is unique
            if not df.index.is_unique:
                st.warning('Duplicate indices detected. Resetting index to ensure uniqueness.')
                df.reset_index(inplace=True)
                df.set_index('InvoiceDate', inplace=True)

            # Feature Engineering
            df['Month'] = df.index.month
            df['DayOfWeek'] = df.index.dayofweek
            df['IsWeekend'] = df['DayOfWeek'] >= 5

            # Drop rows with missing essential data
            df = df.dropna(subset=['CustomerID', 'Description'])
            # Filter out invalid data
            df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

            return df
        else:
            st.error("The uploaded file must contain 'UnitPrice', 'Quantity', and 'InvoiceDate' columns.")
    return None

# Preprocess data
df = process_data(uploaded_file)

if df is not None:
    st.write("Data Preview:", df.head())

    # Log transformation of Sales
    df['Sales_log'] = np.log(df['Sales'] + 1)  # Adding 1 to avoid log(0) errors

    # KPSS test for stationarity
    kpss_result = kpss(df['Sales_log'].dropna(), regression='c')
    st.write(f"KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}")

    # Check for stationarity
    if kpss_result[1] < 0.05:
        st.write("The log-transformed data is trend stationary. Proceeding with differencing.")
    else:
        st.write("The log-transformed data is not trend stationary.")

    # Differencing
    df['Sales_diff'] = df['Sales_log'].diff().dropna()

    # Plotting the differenced data
    st.subheader("First-Order Differenced Log-Transformed Sales Data")
    plt.figure(figsize=(10, 6))
    plt.plot(df['Sales_diff'], label='Differenced Sales Log')
    plt.title('First-Order Differenced Log-Transformed Sales Data')
    plt.xlabel('Date')
    plt.ylabel('Differenced Log(Sales)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # KPSS test on differenced data
    kpss_result_diff = kpss(df['Sales_diff'].dropna(), regression='c')
    st.write(f"KPSS Statistic (differenced): {kpss_result_diff[0]}, p-value: {kpss_result_diff[1]}")

    # Resample the data to monthly
    df_monthly = df.resample('M').agg({
        'Sales_diff': 'sum',
        'UnitPrice': 'mean'
    }).reset_index()

    df_monthly['Month'] = df_monthly['InvoiceDate'].dt.month
    df_monthly['DayOfWeek'] = df_monthly['InvoiceDate'].dt.dayofweek
    df_monthly['IsWeekend'] = df_monthly['DayOfWeek'] >= 5

    # Prepare data for modeling
    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend']]
    y = df_monthly['Sales_diff']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Linear Regression model
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)

    # Predict using the trained model
    y_pred = lin_model.predict(X_test)

    # Future prediction
    future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = df_monthly['UnitPrice'].mean()  # Assuming constant unit price
    combined_data['IsWeekend'] = combined_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Generate predictions for future dates
    lin_sales_predictions = lin_model.predict(combined_data)

    # Plot the historical data and predictions
    st.subheader("Historical and Forecasted Monthly Sales (Linear Regression)")
    plt.figure(figsize=(14, 7))
    plt.plot(df_monthly['InvoiceDate'], y, label='Historical Sales', color='blue')
    plt.plot(future_dates, lin_sales_predictions, label='Linear Regression Predictions', linestyle='--', color='red')
    plt.title('Historical and Forecasted Monthly Sales (Linear Regression)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
