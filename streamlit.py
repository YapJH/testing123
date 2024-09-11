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

# Title for the app
st.title("Data Cleaning and Preparation")

# File upload section
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Function to process the uploaded data
def process_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Ensure 'InvoiceDate' is in datetime format
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

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


def perform_eda_and_checks(df):
    # Convert 'InvoiceDate' to datetime if it's not already
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Ensure 'Sales' column exists
    if 'Sales' in df.columns:
        # Check basic statistics and distribution of 'Sales'
        st.write("Basic Statistical Overview of Sales:")
        st.write(df['Sales'].describe())

        # Sales value counts
        st.write("Sales Value Counts:")
        st.write(df['Sales'].value_counts().head(10))  # Show top 10 most frequent sales values

        # Time series plot of sales
        st.subheader("Time Series Plot of Sales")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['InvoiceDate'], df['Sales'], label='Sales Data')
        ax.set_title('Sales Data Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Log transformation of sales to stabilize variance
        df['Sales_log'] = np.log(df['Sales'] + 1)

        # KPSS test for stationarity on log-transformed data
        kpss_result = kpss(df['Sales_log'].dropna(), regression='c')
        st.write(f"KPSS Statistic on log-transformed data: {kpss_result[0]}")
        st.write(f"KPSS p-value on log-transformed data: {kpss_result[1]}")

        # Decision based on p-value
        if kpss_result[1] < 0.05:
            st.write("The log-transformed data is trend stationary.")
        else:
            st.write("The log-transformed data is not trend stationary, suggesting differencing might be needed.")

        # Applying first order differencing
        df['Sales_diff'] = df['Sales_log'].diff().dropna()

        # Plotting the differenced data
        st.subheader("Differenced Sales Data")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['InvoiceDate'], df['Sales_diff'], label='Differenced Sales Log')
        ax.set_title('First-Order Differenced Log-Transformed Sales Data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Differenced Log(Sales)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Re-running KPSS test on differenced data
        kpss_result_diff = kpss(df['Sales_diff'].dropna(), regression='c')
        st.write(f"KPSS Statistic after differencing: {kpss_result_diff[0]}")
        st.write(f"KPSS p-value after differencing: {kpss_result_diff[1]}")

    else:
        st.error("The required 'Sales' column is not present in the DataFrame.")
