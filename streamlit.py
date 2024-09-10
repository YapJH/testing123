import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss

def load_data():
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

def clean_data(df):
    # Check for 'Sales' column
    if 'Sales' not in df.columns:
        st.error("Uploaded file does not contain a 'Sales' column. Please check your CSV file.")
        return None
    
    # Data cleaning process
    df = df.dropna(subset=['Description', 'CustomerID'])  # Dropping rows where 'Description' or 'CustomerID' is NaN
    df = df[df['Quantity'] > 0]  # Removing rows with non-positive quantities

    return df

def kpss_test(df, column='Sales_log'):
    # Run KPSS test
    kpss_result = kpss(df[column].dropna(), regression='c')
    st.write(f'KPSS Statistic for {column}: {kpss_result[0]}')
    st.write(f'p-value for {column}: {kpss_result[1]}')
    return kpss_result

def plot_data(df, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df, label='Value', color='blue')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

df = load_data()

if df is not None:
    df = clean_data(df)
    if df is not None:
        # Log transformation to stabilize variance
        df['Sales_log'] = np.log(df['Sales'] + 1)  # Adding 1 to avoid log(0)

        # Perform KPSS test on the log-transformed data
        result = kpss_test(df, 'Sales_log')
        
        # Check if differencing is needed
        if result[1] >= 0.05:  # If the p-value is high, indicating non-stationarity
            st.write("The data is not trend stationary and will be differenced.")
            df['Sales_diff'] = df['Sales_log'].diff().dropna()  # Differencing the data

            # Re-test KPSS on differenced data
            kpss_test(df, 'Sales_diff')

            # Plot the differenced data
            plot_data(df['Sales_diff'].dropna(), 'First-Order Differenced Log-Transformed Sales Data', 'Differenced Log(Sales)')
        else:
            st.write("The data is trend stationary. No differencing needed.")
            plot_data(df['Sales_log'], 'Log-Transformed Sales Data', 'Log(Sales)')
