import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data cleaning and preprocessing
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

    # Remove rows where 'Quantity', 'UnitPrice', or 'CustomerID' is negative or zero
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['CustomerID'] > 0)]

    # Adding a small value to avoid log(0) issues for log transformation
    df['Sales_log'] = np.log(df['Sales'] + 1)
    
    # KPSS test to check for stationarity on the log-transformed 'Sales' data
    kpss_result = kpss(df['Sales_log'].dropna(), regression='c')  # 'c' assumes constant trend
    st.write(f'KPSS Statistic: {kpss_result[0]}')
    st.write(f'p-value: {kpss_result[1]}')

    if kpss_result[1] < 0.05:
        st.write("The log-transformed data is trend stationary.")
    else:
        st.write("The log-transformed data is not trend stationary.")
        # Performing differencing if not stationary
        df['Sales_diff'] = df['Sales_log'].diff().dropna()

        # Plot the differenced data
        st.subheader("First-Order Differenced Log-Transformed Sales Data")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Sales_diff'], label='Differenced Sales Log', color='blue')
        ax.set_title('Differenced Log-Transformed Sales Data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Differenced Log(Sales)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Re-run the KPSS test on the differenced data
        kpss_result_diff = kpss(df['Sales_diff'].dropna(), regression='c')
        st.write(f'KPSS Statistic (differenced): {kpss_result_diff[0]}')
        st.write(f'KPSS p-value (differenced): {kpss_result_diff[1]}')

    # Display the count of remaining missing values across columns and the cleaned DataFrame
    st.write("Missing values in each column after cleaning:", df.isnull().sum())
    st.write("Cleaned Data", df)

# Instructions to the user
st.write("Please upload a CSV file with a 'Sales' column to analyze the time series data.")
