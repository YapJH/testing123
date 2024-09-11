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

# Process the data from the CSV file
def process_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Check for necessary columns
        if all(col in df.columns for col in ['UnitPrice', 'Quantity', 'InvoiceDate']):
            # Calculate Sales
            df['Sales'] = df['UnitPrice'] * df['Quantity']
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])  # Convert to datetime
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

# Function to handle predictions and plotting for all models, with yearly option
def model_and_predict(df, model_type='Linear Regression', resample_type='M'):
    if not all(col in df.columns for col in ['Sales', 'UnitPrice']):
        st.error(f"The dataset is missing required columns: ['Sales', 'UnitPrice']")
        return

    # Choose to resample by month or by year
    if resample_type == 'Y':  # Resample yearly
        df_resampled = df.resample('Y').agg({'Sales': 'sum', 'UnitPrice': 'mean'}).reset_index()
    else:  # Resample monthly by default
        df_resampled = df.resample('M').agg({'Sales': 'sum', 'UnitPrice': 'mean'}).reset_index()

    # Re-create the missing columns after resampling
    df_resampled['Month'] = df_resampled['InvoiceDate'].dt.month
    df_resampled['DayOfWeek'] = df_resampled['InvoiceDate'].dt.dayofweek
    df_resampled['IsWeekend'] = df_resampled['DayOfWeek'] >= 5

    # Features and target
    X = df_resampled[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend']]
    y = df_resampled['Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'KNN':
        model = KNeighborsRegressor(n_neighbors=5)
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Decision Tree':
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == 'XGBoost':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    elif model_type == 'Neural Network':
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    else:
        st.error("Invalid model type selected.")
        return

    # Train the model
    model.fit(X_train, y_train)

    # Prepare future data for prediction (next 12 months or years)
    if resample_type == 'Y':  # Forecast for the next 12 years
        future_dates = pd.date_range(start=df_resampled['InvoiceDate'].max() + pd.DateOffset(years=1), periods=12, freq='Y')
    else:  # Forecast for the next 12 months
        future_dates = pd.date_range(start=df_resampled['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    
    future_data = pd.DataFrame(index=future_dates)
    future_data['Month'] = future_data.index.month
    future_data['DayOfWeek'] = future_data.index.dayofweek
    future_data['UnitPrice'] = df_resampled['UnitPrice'].mean()  # Assuming constant UnitPrice
    future_data['IsWeekend'] = future_data['DayOfWeek'] >= 5
    future_data = future_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend']]

    # Predict future sales
    future_sales_predictions = model.predict(future_data)

    # Plot historical and future predictions
    plt.figure(figsize=(14, 7))
    plt.plot(df_resampled['InvoiceDate'], y, label='Historical Sales', color='blue')
    plt.plot(future_dates, future_sales_predictions, label=f'{model_type} Predictions', linestyle='--', color='red')
    if resample_type == 'Y':
        plt.title(f'Historical and Forecasted Yearly Sales ({model_type})')
    else:
        plt.title(f'Historical and Forecasted Monthly Sales ({model_type})')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    st.pyplot()

# Streamlit app
def main():
    st.title("Sales Forecasting with Machine Learning Models")
    st.write("Upload a sales data CSV file and select a machine learning model for forecasting future sales.")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # If a file is uploaded
    if uploaded_file is not None:
        # Process the data
        df = process_data(uploaded_file)

        if df is not None:
            st.write("Data Preview:")
            st.write(df.head())

            # Model selection
            model_type = st.selectbox('Select a Machine Learning Model', 
                                      ['Linear Regression', 'KNN', 'Random Forest', 'Decision Tree', 'XGBoost', 'Neural Network'])

            # Resample type selection (monthly or yearly)
            resample_type = st.selectbox('Choose resampling period (Monthly or Yearly)', ['Monthly', 'Yearly'])

            # Run model and plot predictions
            if resample_type == 'Monthly':
                model_and_predict(df, model_type=model_type, resample_type='M')
            else:
                model_and_predict(df, model_type=model_type, resample_type='Y')
        else:
            st.error("Error processing the file. Please check the format.")
    else:
        st.write("Please upload a CSV file.")

if __name__ == "__main__":
    main()
