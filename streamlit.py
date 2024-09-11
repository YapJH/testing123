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





def perform_modeling(df):
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df.set_index('InvoiceDate', inplace=True)

    # Aggregate to monthly data
    df_monthly = df.resample('M').agg({
        'Sales_diff': 'sum',
        'UnitPrice': 'mean',
        'Country_Encoded': 'mean'
    }).reset_index()
    df_monthly['Month'] = df_monthly['InvoiceDate'].dt.month
    df_monthly['DayOfWeek'] = df_monthly['InvoiceDate'].dt.dayofweek
    df_monthly['IsWeekend'] = df_monthly['DayOfWeek'] >= 5

    # Prepare data for modeling
    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]
    y = df_monthly['Sales_diff']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    models = {
        'Linear Regression': LinearRegression(),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    future_data = pd.DataFrame({
        'Month': [X_train['Month'].mean()] * 12,  # Adjust appropriately if you want real future months
        'DayOfWeek': [0] * 12,  # Example placeholders
        'UnitPrice': [X_train['UnitPrice'].mean()] * 12,
        'IsWeekend': [0] * 12,
        'Country_Encoded': [X_train['Country_Encoded'].mode()[0]] * 12
    })
    
    future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    
    plt.figure(figsize=(15, 10))

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        future_predictions = model.predict(future_data)
        
        # Plotting
        plt.plot(df_monthly['InvoiceDate'], y, label='Historical Sales')
        plt.plot(future_dates, future_predictions, linestyle='--', label=f'{name} Predictions')

        # Display R^2 score and MSE
        st.write(f"{name} R^2 Score: {r2_score(y_test, predictions):.2f}")
        st.write(f"{name} MSE: {mean_squared_error(y_test, predictions):.2f}")
        
    plt.title('Sales Forecast Comparison')
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
            perform_modeling(df_stationary)  # Use the stationary data for modeling
        else:
            st.error("Data could not be processed. Check the file format.")

if __name__ == "__main__":
    main()
