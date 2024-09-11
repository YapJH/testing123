import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import kpss
from sklearn.preprocessing import LabelEncoder  # Make sure this is at the top of your script


# Function to process the uploaded data
def process_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Ensure 'InvoiceDate' is in datetime format
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Extract year, month, day, day of the week, hour from 'InvoiceDate'
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Day'] = df['InvoiceDate'].dt.day
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Hour'] = df['InvoiceDate'].dt.hour

        # Create a boolean column 'IsWeekend' indicating whether the day is a weekend
        df['IsWeekend'] = df['DayOfWeek'] >= 5  # True for Saturday and Sunday

        # Mapping descriptions to stock codes and filling missing descriptions
        stockcode_description_map = df.groupby('StockCode')['Description'].apply(lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
        df['Description'] = df.apply(
            lambda row: stockcode_description_map.get(row['StockCode'], row['Description']) if pd.isnull(row['Description']) else row['Description'],
            axis=1
        )

        # Drop rows with any missing 'Description' or 'CustomerID'
        df = df.dropna(subset=['Description', 'CustomerID'])

        # Filter out rows with non-positive 'Quantity', 'UnitPrice', or missing 'CustomerID'
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['CustomerID'] > 0)]

        df['Sales'] = df['UnitPrice'] * df['Quantity']
        
        df = df[df['Quantity'] > 0]
        df = df[df['UnitPrice'] > 0]
        
        return df

    return None


# Function for EDA
def perform_eda(df):
    # Monthly sales visualization
    df['Month'] = df['InvoiceDate'].dt.to_period('M')
    monthly_sales = df.groupby('Month').agg(Total_Sales=('Quantity', 'sum')).reset_index()
    monthly_sales['Month'] = monthly_sales['Month'].dt.strftime('%Y-%m')

    st.subheader("Monthly Sales")
    fig, ax = plt.subplots()
    ax.plot(monthly_sales['Month'], monthly_sales['Total_Sales'], marker='o')
    ax.set_xticklabels(monthly_sales['Month'], rotation=45)
    ax.set_title('Monthly Sales Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Sales')
    ax.grid(True)
    st.pyplot(fig)

    # Weekly sales visualization
    df['Week'] = df['InvoiceDate'].dt.strftime('%Y-%U')
    weekly_sales = df.groupby('Week').agg(Total_Sales=('Quantity', 'sum')).reset_index()

    st.subheader("Weekly Sales")
    fig, ax = plt.subplots()
    ax.plot(weekly_sales['Week'], weekly_sales['Total_Sales'], marker='o')
    ax.set_xticklabels(weekly_sales['Week'], rotation=45)
    ax.set_title('Weekly Sales Over Time')
    ax.set_xlabel('Week')
    ax.set_ylabel('Total Sales')
    ax.grid(True)
    st.pyplot(fig)

    # Quarterly sales visualization
    df['Quarter'] = df['InvoiceDate'].dt.to_period('Q')
    quarterly_sales = df.groupby('Quarter').agg(Total_Sales=('Quantity', 'sum')).reset_index()
    quarterly_sales['Quarter'] = quarterly_sales['Quarter'].astype(str)

    st.subheader("Quarterly Sales")
    fig, ax = plt.subplots()
    ax.plot(quarterly_sales['Quarter'], quarterly_sales['Total_Sales'], marker='o')
    ax.set_xticklabels(quarterly_sales['Quarter'], rotation=45)
    ax.set_title('Quarterly Sales Over Time')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Total Sales')
    ax.grid(True)
    st.pyplot(fig)

    # Scatter plot of sales
    st.subheader("Sales Data Visualization Over Month")
    fig, ax = plt.subplots()
    ax.scatter(df['InvoiceDate'], df['Sales'], alpha=0.5)
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.set_title('Sales Scatter Plot Over Time')
    ax.set_xlabel('Invoice Date')
    ax.set_ylabel('Sales')
    ax.grid(True)
    plt.gcf().autofmt_xdate()
    st.pyplot(fig)


def check_stationarity(df):
    df['Sales'] = df['UnitPrice'] * df['Quantity']
    df['Sales_log'] = np.log(df['Sales'] + 1)  # Log transformation
    
    stationary = False
    data_to_test = df['Sales_log']
    
    while not stationary:
        # KPSS test for stationarity
        kpss_result = kpss(data_to_test.dropna(), regression='c')
        st.write('KPSS Statistic:', kpss_result[0])
        st.write('p-value:', kpss_result[1])
        
        if kpss_result[1] < 0.05:
            st.write("The data is trend stationary.")
            df['Sales_diff'] = df['Sales_log']  # No differencing needed, use Sales_log
            stationary = True  # Break the loop if data is stationary
        else:
            st.write("The data is not trend stationary, differencing the data.")
            # Perform differencing and set the differenced data as the next data to test
            df['Sales_diff'] = data_to_test.diff().dropna()
            data_to_test = df['Sales_diff']

    # Return the entire DataFrame, not just the differenced series
    return df





# Function for modeling
def perform_modeling(df_stationary):
    # Ensure the 'InvoiceDate' is set as the index for resampling
    if 'InvoiceDate' not in df_stationary.columns:
        st.error("'InvoiceDate' column is missing or not in the correct format.")
        return

    # Set 'InvoiceDate' as the index if not already
    df_stationary['InvoiceDate'] = pd.to_datetime(df_stationary['InvoiceDate'], errors='coerce')
    df_stationary.set_index('InvoiceDate', inplace=True)

    # Label encoding for categorical features
    df_stationary['Country_Encoded'] = pd.factorize(df_stationary['Country'])[0]

    # Feature engineering
    df_stationary['Month'] = df_stationary.index.month
    df_stationary['DayOfWeek'] = df_stationary.index.dayofweek
    df_stationary['IsWeekend'] = df_stationary['DayOfWeek'] >= 5

    # Resample data to monthly
    try:
        df_monthly = df_stationary.resample('M').agg({
            'Sales_diff': 'sum',
            'UnitPrice': 'mean',
            'Country_Encoded': 'mean'
        }).reset_index()
    except Exception as e:
        st.error(f"Error during resampling: {e}")
        return

    # Preparing the features and labels
    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]
    y = df_monthly['Sales_diff']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression Model
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)

    # Prediction
    y_pred = lin_model.predict(X_test)

    st.write("Model Coefficients:", lin_model.coef_)
    st.write("Intercept:", lin_model.intercept_)
    st.write("Score:", lin_model.score(X_test, y_test))

    # Future predictions
    future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    future_data = pd.DataFrame(index=future_dates)
    future_data['Month'] = future_data.index.month
    future_data['DayOfWeek'] = future_data.index.dayofweek
    future_data['UnitPrice'] = df_monthly['UnitPrice'].mean()
    future_data['IsWeekend'] = future_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    future_data['Country_Encoded'] = df_monthly['Country_Encoded'].mode()[0]

    # Predict future sales
    future_sales_predictions = lin_model.predict(future_data)

    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(df_monthly['InvoiceDate'], df_monthly['Sales_diff'], label='Historical Sales')
    plt.plot(future_dates, future_sales_predictions, 'r--', label='Predicted Sales')
    plt.title('Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    st.pyplot()


def main():
    st.title("Sales Data Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_cleaned = process_data(uploaded_file)
        if df_cleaned is not None:
            perform_eda(df_cleaned)
            df_stationary = check_stationarity(df_cleaned)  # Get the stationary DataFrame
            perform_modeling(df_stationary)  # Use the stationary DataFrame for modeling
        else:
            st.error("Data could not be processed. Check the file format.")


if __name__ == "__main__":
    main()
