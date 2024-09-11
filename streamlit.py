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

def model_and_predict(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.set_index('InvoiceDate', inplace=True)
    df_monthly = df.resample('M').agg({
        'Sales_diff': 'sum',
        'UnitPrice': 'mean',
        'Country_Encoded': 'mean'
    }).reset_index()
    df_monthly['Month'] = df_monthly['InvoiceDate'].dt.month
    df_monthly['DayOfWeek'] = df_monthly['InvoiceDate'].dt.dayofweek
    df_monthly['IsWeekend'] = df_monthly['DayOfWeek'] >= 5

    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]
    y = df_monthly['Sales_diff']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)

    future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = df_monthly['UnitPrice'].mean()
    combined_data['IsWeekend'] = combined_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    combined_data['Country_Encoded'] = df_monthly['Country_Encoded'].mode()[0]

    lin_sales_predictions = lin_model.predict(combined_data)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_monthly['InvoiceDate'], y, label='Historical Sales', color='blue')
    ax.plot(future_dates, lin_sales_predictions, label='Linear Regression Predictions', linestyle='--', color='red')
    ax.set_title('Historical and Forecasted Monthly Sales (Linear Regression)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Sales Forecasting App")
    uploaded_file = st.file_uploader("Choose a CSV file")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        model_and_predict(df)

if __name__ == "__main__":
    main()
