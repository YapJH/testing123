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
from sklearn.preprocessing import LabelEncoder  
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def process_data(uploaded_file):
    try:
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
            stockcode_description_map = df.groupby('StockCode')['Description'].apply(
                lambda x: x.mode()[0] if not x.mode().empty else None).to_dict()
            df['Description'] = df.apply(
                lambda row: stockcode_description_map.get(row['StockCode'], row['Description']) if pd.isnull(row['Description']) else row['Description'],
                axis=1
            )
            
            # Label Encoding for 'StockCode'
            le_stockcode = LabelEncoder()
            df['StockCode_Encoded'] = le_stockcode.fit_transform(df['StockCode'])

            # Label Encoding for 'Country'
            le_country = LabelEncoder()
            df['Country_Encoded'] = le_country.fit_transform(df['Country'])

            # Convert 'IsWeekend' to binary encoding
            df['IsWeekend_Encoded'] = df['IsWeekend'].astype(int)

            
            # Drop rows with any missing 'Description' or 'CustomerID'
            df = df.dropna(subset=['Description', 'CustomerID'])

            # Filter out rows with non-positive 'Quantity', 'UnitPrice', or missing 'CustomerID'
            df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['CustomerID'] > 0)]

            # Calculate 'Sales' as 'UnitPrice' * 'Quantity'
            df['Sales'] = df['UnitPrice'] * df['Quantity']

            # Filter out rows with non-positive 'Sales'
            df = df[df['Sales'] > 0]

            return df

    except Exception as e:
        # Use Streamlit's error message display to show what went wrong
        st.error(f"An error occurred: {e}")
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






def prepare_data(df):
    # Ensure 'InvoiceDate' is set as the DataFrame index and is of datetime type
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    if not df.index.name or df.index.name != 'InvoiceDate':
        df.set_index('InvoiceDate', inplace=True)

    # Create a new DataFrame starting from the earliest date in the existing data
    earliest_date = df.index.min()
    new_data = {
        'InvoiceDate': pd.date_range(start=earliest_date, periods=100, freq='D'),
        'Sales_diff': [100 + i * 5 for i in range(100)],
        'UnitPrice': [10] * 100,
        'Country_Encoded': [0, 1, 0, 1] * 25
    }
    new_df = pd.DataFrame(new_data)
    new_df['InvoiceDate'] = pd.to_datetime(new_df['InvoiceDate'])
    new_df.set_index('InvoiceDate', inplace=True)
    new_df['Month'] = new_df.index.month
    new_df['DayOfWeek'] = new_df.index.dayofweek
    new_df['IsWeekend'] = new_df['DayOfWeek'] >= 5

    # Aggregate existing data to monthly granularity
    df_monthly = df.resample('M').agg({
        'Sales_diff': 'sum',
        'UnitPrice': 'mean',
        'Country_Encoded': 'mean'
    }).reset_index()
    df_monthly['Month'] = df_monthly['InvoiceDate'].dt.month
    df_monthly['DayOfWeek'] = df_monthly['InvoiceDate'].dt.dayofweek
    df_monthly['IsWeekend'] = df_monthly['DayOfWeek'] >= 5

    return df_monthly, new_df













def monthly_sales_linear_regression(df_monthly):
    # Prepare features and target variables
    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]
    y = df_monthly['Sales_diff']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = lin_model.predict(X_test)

    # Prepare the combined future and historical index for future dates
    future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = df_monthly['UnitPrice'].mean()
    combined_data['IsWeekend'] = combined_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    combined_data['Country_Encoded'] = df_monthly['Country_Encoded'].mode()[0]

    # Generate predictions for the entire period
    lin_sales_predictions = lin_model.predict(combined_data)

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(df_monthly['InvoiceDate'], y, label='Historical Sales', color='blue')
    plt.plot(future_dates, lin_sales_predictions, label='Linear Regression Predictions', linestyle='--', color='red')
    plt.title('Historical and Forecasted Monthly Sales (Linear Regression)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()


def main():
    st.title("Sales Data Analysis")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_cleaned = process_data(uploaded_file)
        if df_cleaned is not None:
            perform_eda(df_cleaned)
            df_stationary = check_stationarity(df_cleaned)  # Get the stationary data
            df_monthly, new_df = prepare_data(df_stationary)  # Prepare the data for modeling
            
            # Call the Linear Regression model function
            monthly_sales_linear_regression(df_monthly)  # Use the monthly sales linear regression model
        else:
            st.error("Data could not be processed. Check the file format.")

if __name__ == "__main__":
    main()
