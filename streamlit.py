import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import kpss

def process_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df.set_index('InvoiceDate', inplace=True)
        
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

def model_data(df):
    # Aggregate to monthly data
    df_monthly = df.resample('M').agg({
        'Sales_diff': 'sum',
        'UnitPrice': 'mean',
        'Country_Encoded': 'mean'
    })
    df_monthly['Month'] = df_monthly.index.month
    df_monthly['DayOfWeek'] = df_monthly.index.dayofweek
    df_monthly['IsWeekend'] = df_monthly['DayOfWeek'] >= 5

    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]
    y = df_monthly['Sales_diff']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    
    # Predictions
    future_dates = pd.date_range(start=df_monthly.index.max() + pd.DateOffset(months=1), periods=12, freq='M')
    future_data = pd.DataFrame(index=future_dates)
    future_data['Month'] = future_data.index.month
    future_data['DayOfWeek'] = future_data.index.dayofweek
    future_data['UnitPrice'] = df_monthly['UnitPrice'].mean()  # Assuming constant unit price
    future_data['IsWeekend'] = future_data['DayOfWeek'] >= 5
    future_data['Country_Encoded'] = df_monthly['Country_Encoded'].mode()[0]
    
    future_data = future_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]
    y_future = lin_model.predict(future_data)
    
    return df_monthly, future_dates, y_future

def plot_results(df_monthly, future_dates, y_future):
    plt.figure(figsize=(14, 7))
    plt.plot(df_monthly.index, df_monthly['Sales_diff'], label='Historical Sales', color='blue')
    plt.plot(future_dates, y_future, label='Linear Regression Predictions', linestyle='--', color='red')
    plt.title('Historical andForecasted Monthly Sales (Linear Regression)")
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

def main():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df_prepared = process_data(uploaded_file)
        if df_prepared is not None:
            df_monthly, future_dates, y_future = model_data(df_prepared)
            plot_results(df_monthly, future_dates, y_future)

if __name__ == "__main__":
    main()
