import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss

def process_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Sales' not in df.columns:
            st.error("The uploaded file does not contain a 'Sales' column. Please check your CSV file.")
            return None
        # Continue with further processing if the 'Sales' column is present
        # Your existing data processing code here
        return df
    return None


def check_stationarity(df):
    if 'Sales' in df.columns:
        df['Sales_log'] = np.log(df['Sales'] + 1)
        kpss_result = kpss(df['Sales_log'].dropna(), regression='c')
        st.write('KPSS Statistic:', kpss_result[0])
        st.write('p-value:', kpss_result[1])
        
        if kpss_result[1] < 0.05:
            st.write("The log-transformed data is trend stationary")
        else:
            st.write("The log-transformed data is not trend stationary, applying differencing.")
            df['Sales_diff'] = df['Sales_log'].diff().dropna()
            plt.figure(figsize=(10, 6))
            plt.plot(df['Sales_diff'], label='Differenced Sales Log')
            plt.title('First-Order Differenced Log-Transformed Sales Data')
            plt.xlabel('Date')
            plt.ylabel('Differenced Log(Sales)')
            plt.legend()
            plt.grid(True)
            st.pyplot()
            
            kpss_result_diff = kpss(df['Sales_diff'].dropna(), regression='c')
            st.write('KPSS Statistic (differenced):', kpss_result_diff[0])
            st.write('KPSS p-value (differenced):', kpss_result_diff[1])
    else:
        st.error("The 'Sales' column is required for stationarity checks but is not present in the DataFrame.")

def display_data(df):
    if df is not None:
        missing_values = df.isnull().sum()
        st.write("Missing values in each column:", missing_values)
        st.write("Cleaned Data", df)
    else:
        st.write("No data to display. Please upload a valid CSV file.")

def main():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    cleaned_data = process_data(uploaded_file)
    if cleaned_data is not None:
        check_stationarity(cleaned_data)
    display_data(cleaned_data)

if __name__ == "__main__":
    main()
