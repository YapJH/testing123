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

    plt.figure(figsize=(14, 7))
    plt.scatter(
        df['InvoiceDate'],
        df['Sales'],
        color='blue',
        marker='o',
        s=10,
        alpha=0.5
    )

    # Formatting the date axis
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # One tick per year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    # Formatting the y-axis to show labels with commas for thousands
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Sales')
    plt.title('Sales Data Visualization Over Years')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(['Individual Sales'], loc='upper left')

    # Display the plot in Streamlit
    st.pyplot(plt)
    
# Process the data
df = process_data(uploaded_file)

# Display the cleaned data
if df is not None:
    st.write("Cleaned Data Preview:", df.head())
    
    # Call the EDA function after confirming the data is loaded and cleaned
    perform_eda(df)
else:
    st.write("No data to display. Please upload a file with the correct format.")



