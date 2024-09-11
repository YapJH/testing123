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
            stationary = True  # Break the loop if data is stationary
        else:
            st.write("The data is not trend stationary, differencing the data.")
            # Perform differencing and set the differenced data as the next data to test
            df['Sales_diff'] = data_to_test.diff().dropna()
            data_to_test = df['Sales_diff']

    # Return the entire DataFrame, not just the differenced series
    return df





def perform_modeling(df_stationary):
    # Check if the necessary columns exist
    # Label Encoding StockCode
    le_stockcode = LabelEncoder()
    df_stationary['StockCode_Encoded'] = le_stockcode.fit_transform(df_stationary['StockCode'])

    # Label Encoding Country
    le_country = LabelEncoder()
    df_stationary['Country_Encoded'] = le_country.fit_transform(df_stationary['Country'])

    # Convert IsWeekend to binary encoding (1 for True, 0 for False)
    df_stationary['IsWeekend_Encoded'] = df_stationary['IsWeekend'].astype(int)

    
    required_columns = ['Sales_diff', 'UnitPrice', 'Country_Encoded', 'InvoiceDate']
    missing_columns = [col for col in required_columns if col not in df_stationary.columns]
    
    if missing_columns:
        st.error(f"The following required columns are missing: {missing_columns}")
        return
    
    # Resample data to monthly aggregates
    try:
        df_stationary.set_index('InvoiceDate', inplace=True)
        df_monthly = df_stationary.resample('M').agg({
            'Sales_diff': 'sum',  
            'UnitPrice': 'mean',
            'Country_Encoded': 'mean'
        }).reset_index()

        # Feature engineering
        df_monthly['Month'] = df_monthly['InvoiceDate'].dt.month
        df_monthly['DayOfWeek'] = df_monthly['InvoiceDate'].dt.dayofweek
        df_monthly['IsWeekend'] = df_monthly['DayOfWeek'] >= 5

        # Prepare the features (X) and target (y)
        X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]
        y = df_monthly['Sales_diff']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Linear Regression model
        lin_model = LinearRegression()
        lin_model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = lin_model.predict(X_test)

        # Display model performance
        st.write("Model Coefficients:", lin_model.coef_)
        st.write("Intercept:", lin_model.intercept_)
        st.write("R² Score (Test Set):", lin_model.score(X_test, y_test))

        # Future predictions for the next 12 months
        future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
        future_data = pd.DataFrame(index=future_dates)
        future_data['Month'] = future_data.index.month
        future_data['DayOfWeek'] = future_data.index.dayofweek
        future_data['UnitPrice'] = df_monthly['UnitPrice'].mean()  # Assuming constant unit price
        future_data['IsWeekend'] = future_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        future_data['Country_Encoded'] = df_monthly['Country_Encoded'].mode()[0]

        # Predict future sales
        future_sales_predictions = lin_model.predict(future_data)

        # Plot the results
        plt.figure(figsize=(14, 7))
        plt.plot(df_monthly['InvoiceDate'], df_monthly['Sales_diff'], label='Historical Sales')
        plt.plot(future_dates, future_sales_predictions, 'r--', label='Predicted Sales')
        plt.title('Sales Forecast (Historical vs. Predicted)')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True)
        st.pyplot()

    except KeyError as e:
        st.error(f"KeyError: {e}")


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
