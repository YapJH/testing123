import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.stattools import kpss

# Process the data from the CSV file
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

def apply_transformations(df):
    # Convert 'InvoiceDate' to datetime if not already done
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    if df.duplicated(subset='InvoiceDate').any():
        st.warning('Duplicate dates detected. Aggregating sales data by date.')
        df = df.groupby('InvoiceDate').agg({
            'Sales': 'sum',  # Summing sales if there are duplicates
            'UnitPrice': 'mean'  # Averaging unit price
        }).reset_index()

    df.set_index('InvoiceDate', inplace=True)

    # Log transformation
    df['Sales_log'] = np.log(df['Sales'] + 1)
    # First order differencing
    df['Sales_diff'] = df['Sales_log'].diff().dropna()
    
    # Plotting the differenced data
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Sales_diff'], label='Differenced Sales Log')
    plt.title('First-Order Differenced Log-Transformed Sales Data')
    plt.xlabel('Date')
    plt.ylabel('Differenced Log(Sales)')
    plt.legend()
    plt.grid(True)
    st.pyplot()

    # KPSS test to check for stationarity after differencing
    from statsmodels.tsa.stattools import kpss
    kpss_result_diff = kpss(df['Sales_diff'].dropna(), regression='c')
    st.write(f'KPSS Statistic (differenced): {kpss_result_diff[0]}')
    st.write(f'KPSS p-value (differenced): {kpss_result_diff[1]}')

    return df

def main():
    st.title("Sales Forecasting App")
    uploaded_file = st.file_uploader("Choose a CSV file")

    if uploaded_file is not None:
        df = process_data(uploaded_file)
        if df is not None:
            st.write(df.head())
            transformed_df = apply_transformations(df)
            model_type = st.selectbox("Select Model Type", ['Linear Regression', 'KNN', 'Random Forest', 'Decision Tree', 'XGBoost', 'Neural Network'])
            model_and_predict(transformed_df, model_type)

if __name__ == "__main__":
    main()
