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


def process_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Calculate Sales by multiplying UnitPrice and Quantity
        df['Sales'] = df['UnitPrice'] * df['Quantity']
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])  # Ensure 'InvoiceDate' is in datetime
        df.set_index('InvoiceDate', inplace=True)

        df['Month'] = df.index.month
        df['DayOfWeek'] = df.index.dayofweek
        df['IsWeekend'] = df['DayOfWeek'] >= 5
        
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

# Function to train different models and plot results
def model_data_and_plot(df, model_type='Linear Regression'):
    # Ensure necessary columns exist before aggregation
    required_columns = ['Sales', 'UnitPrice', 'Country_Encoded']
    if not all(col in df.columns for col in required_columns):
        st.error(f"The dataset is missing required columns: {required_columns}")
        return

    # Aggregate existing data to monthly
    df_monthly = df.resample('M').agg({
        'Sales': 'sum',
        'UnitPrice': 'mean',
        'Country_Encoded': 'mean'
    }).reset_index()

    # Prepare data for modeling
    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]
    y = df_monthly['Sales']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select the model
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

    # Train the selected model
    model.fit(X_train, y_train)

    # Prepare future dates for prediction
    future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    future_data = pd.DataFrame(index=future_dates)
    future_data['Month'] = future_data.index.month
    future_data['DayOfWeek'] = future_data.index.dayofweek
    future_data['UnitPrice'] = df_monthly['UnitPrice'].mean()  # Assuming constant unit price
    future_data['IsWeekend'] = future_data['DayOfWeek'] >= 5
    future_data['Country_Encoded'] = df_monthly['Country_Encoded'].mode()[0]

    # Ensure the order of features matches the model's expectations
    future_data = future_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]

    # Generate predictions for the future period
    future_predictions = model.predict(future_data)

    # Plot the results
    plt.figure(figsize=(14, 7))
    plt.plot(df_monthly['InvoiceDate'], y, label='Historical Sales', color='blue')
    plt.plot(future_dates, future_predictions, label=f'{model_type} Predictions', linestyle='--', color='red')
    plt.title(f'Historical and Forecasted Monthly Sales ({model_type})')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    st.pyplot()

# Streamlit application
def main():
    st.title('Sales Forecasting App')
    st.write('Upload your sales data (CSV format) and choose a machine learning model to forecast future sales.')

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        # Process the uploaded data
        df = process_data(uploaded_file)

        if df is not None:
            st.write("Data preview:")
            st.write(df.head())

            # Model selection
            model_type = st.selectbox('Choose a model for forecasting', 
                                      ['Linear Regression', 'KNN', 'Random Forest', 'Decision Tree', 'XGBoost', 'Neural Network'])

            # Run the model and display forecast
            model_data_and_plot(df, model_type=model_type)
        else:
            st.error("Error processing the file. Please check the format.")
    else:
        st.write("Please upload a CSV file.")

if __name__ == "__main__":
    main()
