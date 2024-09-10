import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss

def process_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Calculate Sales by multiplying UnitPrice and Quantity
        df['Sales'] = df['UnitPrice'] * df['Quantity']
        
        # Create a mapping of StockCode to the most common Description
        stockcode_description_map = df.groupby('StockCode')['Description'].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        ).to_dict()

        # Fill missing Description values based on the StockCode
        df['Description'] = df.apply(
            lambda row: stockcode_description_map[row['StockCode']] if pd.isnull(row['Description']) else row['Description'],
            axis=1
        )

        # Drop rows with missing Description or CustomerID
        df = df.dropna(subset=['Description', 'CustomerID'])

        # Remove rows with non-positive values in specified columns
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['CustomerID'] > 0)]

        return df
    return None

def apply_differencing(df):
    # Log transformation and differencing
    df['Sales_diff'] = df['Sales_log'].diff().dropna()

    # Apply seasonal differencing (e.g., period=12 for monthly data)
    df['Sales_seasonal_diff'] = df['Sales_log'].diff(12).dropna()

    # Plot the differenced data
    plt.figure(figsize=(10, 6))
    plt.plot(df['Sales_seasonal_diff'], label='Seasonally Differenced Sales Log')
    plt.title('Seasonal Differencing of Log-Transformed Sales Data')
    plt.xlabel('Date')
    plt.ylabel('Seasonally Differenced Log(Sales)')
    plt.legend()
    plt.grid(True)
    st.pyplot()

    # Re-run the KPSS test on the seasonally differenced data
    kpss_seasonal = kpss(df['Sales_seasonal_diff'].dropna(), regression='c')
    st.write(f'KPSS Statistic (seasonal differencing): {kpss_seasonal[0]}')
    st.write(f'KPSS p-value (seasonal differencing): {kpss_seasonal[1]}')

def check_stationarity(df):
    if 'Sales' in df.columns:
        df['Sales_log'] = np.log(df['Sales'] + 1)
        kpss_result = kpss(df['Sales_log'].dropna(), regression='c')
        st.write('KPSS Statistic:', kpss_result[0])
        st.write('p-value:', kpss_result[1])

        if kpss_result[1] < 0.05:
            st.write("The log-transformed data is trend stationary.")
            if st.checkbox('Apply differencing despite stationarity?'):
                apply_differencing(df)
        else:
            st.write("The log-transformed data is not trend stationary, applying differencing.")
            apply_differencing(df)
    else:
        st.error("The 'Sales' column is required for stationarity checks but is not present in the DataFrame.")

def main():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = process_data(uploaded_file)
        if df is not None:
            check_stationarity(df)
        else:
            st.error("Please ensure the uploaded file includes 'UnitPrice' and 'Quantity' columns to calculate 'Sales'.")

if __name__ == "__main__":
    main()
