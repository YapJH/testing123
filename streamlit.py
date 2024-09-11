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
    # Log transformation and differencing
    df['Sales_log'] = np.log(df['Sales'] + 1)
    df['Sales_diff'] = df['Sales_log'].diff().dropna()

    # Check for stationarity with KPSS
    if 'Sales_diff' in df:
        kpss_result_diff = kpss(df['Sales_diff'], regression='c')
        st.write(f'KPSS Statistic (differenced): {kpss_result_diff[0]}')
        st.write(f'KPSS p-value (differenced): {kpss_result_diff[1]}')

    return df

def model_and_predict(df, model_type='Linear Regression'):
    if not all(col in df.columns for col in ['Sales']):
        st.error("The dataset is missing required columns: ['Sales']")
        return

    X = df[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend']]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select and train model
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'KNN':
        model = KNeighborsRegressor(n_neighbors=5)
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Decision Tree':
        model = Decision TreeRegressor(random_state=42)
    elif model_type == 'XGBoost':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    elif model_type == 'Neural Network':
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    else:
        st.error("Invalid model type selected.")
        return

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.write(f'Predictions: {predictions}')

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
