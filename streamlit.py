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
from statsmodels.tsa.stattools import kpss
from sklearn.preprocessing import LabelEncoder  
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor  # Import for Neural Networks
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
    fig, ax = plt.subplots(figsize=(14, 7))  # Create a figure and axis for plotting
    ax.plot(df_monthly['InvoiceDate'], y, label='Historical Sales', color='blue')
    ax.plot(future_dates, lin_sales_predictions, label='Linear Regression Predictions', linestyle='--', color='red')
    ax.set_title('Historical and Forecasted Monthly Sales (Linear Regression)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()

    # Use st.pyplot to display the plot in Streamlit
    st.pyplot(fig)


def yearly_sales_linear_regression(new_df):
    # Prepare features and target variables
    X = new_df[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]
    y = new_df['Sales_diff']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)

    # Prepare future dates for forecasting
    future_dates = pd.date_range(start=new_df.index.min(), periods=len(new_df) + 12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = X['UnitPrice'].mean()  # Assuming constant unit price
    combined_data['IsWeekend'] = 0  # Assuming non-weekend for simplicity
    combined_data['Country_Encoded'] = X['Country_Encoded'].mode()[0]  # Most frequent or specific value

    # Ensure the order of features matches the model's expectations
    combined_data = combined_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]

    # Generate predictions for the entire period
    lin_sales_predictions = lin_model.predict(combined_data)

    # Plotting the results
    fig, ax = plt.subplots(figsize=(14, 7))  # Create a figure and axis for plotting
    ax.plot(new_df.index, y, label='Historical Sales', color='blue')
    ax.plot(future_dates, lin_sales_predictions, label='Linear Regression Predictions', linestyle='--', color='red')
    ax.set_title('Historical and Forecasted Yearly Sales (Linear Regression)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()

    # Use st.pyplot to display the plot in Streamlit
    st.pyplot(fig)



def monthly_sales_knn(df_monthly):
    # Prepare features and target variables
    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]
    y = df_monthly['Sales_diff']

    # Train the KNN model
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X, y)

    # Prepare future dates for forecasting
    future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = df_monthly['UnitPrice'].mean()  # Assuming constant unit price
    combined_data['IsWeekend'] = combined_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    combined_data['Country_Encoded'] = df_monthly['Country_Encoded'].mode()[0]  # Most frequent or specific value

    # Generate predictions for the entire period
    combined_sales_predictions = knn_model.predict(combined_data)

    # Plotting the results
    fig, ax = plt.subplots(figsize=(14, 7))  # Create a figure and axis for plotting
    ax.plot(df_monthly['InvoiceDate'], y, label='Historical Sales', color='blue')
    ax.plot(future_dates, combined_sales_predictions, label='KNN Predictions', linestyle='--', color='red')
    ax.set_title('Historical and Forecasted Monthly Sales (KNN)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()

    # Use st.pyplot to display the plot in Streamlit
    st.pyplot(fig)


def yearly_sales_knn(new_df):
    # Prepare features and target variables
    X = new_df[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]  # Example features
    y = new_df['Sales_diff']  # Target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the KNN model
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Prepare future dates for forecasting
    future_dates = pd.date_range(start=new_df.index.min(), periods=len(new_df) + 12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = X['UnitPrice'].mean()  # Assuming constant unit price
    combined_data['IsWeekend'] = 0  # Assuming no weekend for simplicity
    combined_data['Country_Encoded'] = X['Country_Encoded'].mode()[0]  # Most frequent or specific value

    # Ensure the order of features matches the model's expectations
    combined_data = combined_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]

    # Generate predictions for the entire period
    combined_sales_predictions = knn_model.predict(combined_data)

    # Plotting the results
    fig, ax = plt.subplots(figsize=(14, 7))  # Create a figure and axis for plotting
    ax.plot(new_df.index, y, label='Historical Sales', color='blue')
    ax.plot(future_dates, combined_sales_predictions, label='KNN Predictions', linestyle='--', color='red')
    ax.set_title('Historical and Forecasted Sales (KNN)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()

    # Use st.pyplot to display the plot in Streamlit
    st.pyplot(fig)





def monthly_sales_random_forest(df_monthly):
    # Prepare features and target variables
    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]  # Example features
    y = df_monthly['Sales_diff']  # Target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train, y_train)

    # Prepare future dates for forecasting
    future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = df_monthly['UnitPrice'].mean()  # Assuming constant unit price
    combined_data['IsWeekend'] = combined_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    combined_data['Country_Encoded'] = df_monthly['Country_Encoded'].mode()[0]  # Most frequent value

    # Ensure the order of features matches the model's expectations
    combined_data = combined_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]

    # Generate predictions for the entire period using the trained Random Forest model
    combined_sales_predictions = random_forest_model.predict(combined_data)

    # Plotting the results
    fig, ax = plt.subplots(figsize=(14, 7))  # Create a figure and axis for plotting
    ax.plot(df_monthly['InvoiceDate'], y, label='Historical Sales', color='blue')
    ax.plot(future_dates, combined_sales_predictions, label='Model Predictions (Random Forest)', linestyle='--', color='red')
    ax.set_title('Historical and Forecasted Monthly Sales (Random Forest)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()

    # Use st.pyplot to display the plot in Streamlit
    st.pyplot(fig)



def yearly_sales_random_forest(new_df):
    # Prepare features and target variables
    X = new_df[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]  # Example features
    y = new_df['Sales_diff']  # Target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train, y_train)

    # Prepare future dates for forecasting
    future_dates = pd.date_range(start=new_df.index.min(), periods=len(new_df) + 12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = X['UnitPrice'].mean()  # Assuming constant unit price
    combined_data['IsWeekend'] = 0  # Assuming non-weekend for simplicity
    combined_data['Country_Encoded'] = X['Country_Encoded'].mode()[0]  # Most frequent or specific value

    # Ensure the order of features matches the model's expectations
    combined_data = combined_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]

    # Generate predictions for the entire period using the trained Random Forest model
    combined_sales_predictions = random_forest_model.predict(combined_data)

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(new_df.index, y, label='Historical Sales', color='blue')
    plt.plot(future_dates, combined_sales_predictions, label='Model Predictions (Random Forest)', linestyle='--', color='red')
    plt.title('Historical and Forecasted Sales (Random Forest)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    # Display the plot in Streamlit
    st.pyplot(plt)


def monthly_sales_XGBoost(df_monthly):
    # Prepare features and target variables
    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]  
    y = df_monthly['Sales_diff']  

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBRegressor model
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)

    # Prepare future dates for prediction
    future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = df_monthly['UnitPrice'].mean()  
    combined_data['IsWeekend'] = combined_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    combined_data['Country_Encoded'] = df_monthly['Country_Encoded'].mode()[0]  

    # Generate predictions from XGBoost model
    xgb_predictions = xgb_model.predict(combined_data)

    # Plotting the results
    plt.figure(figsize=(10, 5))  # Adjust figure size to your preference
    plt.plot(df_monthly['InvoiceDate'], y, label='Historical Sales', color='blue')
    plt.plot(future_dates, xgb_predictions, label='XGBoost Predictions', linestyle='--', color='red')
    plt.title('XGBoost Predictions')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)  # Add grid for better readability
    plt.tight_layout()
    plt.show()

    # Display the plot in Streamlit
    st.pyplot(plt)



def yearly_sales_XGBoost(new_df):
    # Prepare features and target variables
    X = new_df[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]  
    y = new_df['Sales_diff']  

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBRegressor model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Prepare future dates for forecasting
    future_dates = pd.date_range(start=new_df.index.min(), periods=len(new_df) + 12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = X['UnitPrice'].mean()  
    combined_data['IsWeekend'] = 0  # Assuming non-weekend for simplicity
    combined_data['Country_Encoded'] = X['Country_Encoded'].mode()[0]  

    # Ensure the order of features matches the model's expectations
    combined_data = combined_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]

    # Generate predictions for the entire period
    combined_sales_predictions = model.predict(combined_data)

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(new_df.index, y, label='Historical Sales')
    plt.plot(future_dates, combined_sales_predictions, label='Model Predictions', linestyle='--', color='red')
    plt.title('Historical and Forecasted Sales (XGBoost)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    # Display the plot in Streamlit
    st.pyplot(plt)


def monthly_sales_decision_tree(df_monthly):
    # Prepare features and target variables
    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]  
    y = df_monthly['Sales_diff']  

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree model
    decision_tree_model = DecisionTreeRegressor(random_state=42)
    decision_tree_model.fit(X_train, y_train)

    # Prepare future dates for forecasting
    future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = X['UnitPrice'].mean()  
    combined_data['IsWeekend'] = 0  # Assuming non-weekend for simplicity
    combined_data['Country_Encoded'] = X['Country_Encoded'].mode()[0]  

    # Ensure the order of features matches the model's expectations
    combined_data = combined_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]

    # Generate predictions for the entire period
    combined_sales_predictions = decision_tree_model.predict(combined_data)

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(df_monthly['InvoiceDate'], y, label='Historical Sales', color='blue')
    plt.plot(future_dates, combined_sales_predictions, label='Model Predictions (Decision Tree)', linestyle='--', color='red')
    plt.title('Historical and Forecasted Monthly Sales (Decision Tree)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    # Display the plot in Streamlit
    st.pyplot(plt)


def yearly_sales_decision_tree(new_df):
    # Prepare features and target variables
    X = new_df[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]  
    y = new_df['Sales_diff']  

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Decision Tree model
    decision_tree_model = DecisionTreeRegressor(random_state=42)
    decision_tree_model.fit(X_train, y_train)

    # Prepare future dates for forecasting
    future_dates = pd.date_range(start=new_df.index.min(), periods=len(new_df) + 12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = X['UnitPrice'].mean()  
    combined_data['IsWeekend'] = 0  # Assuming non-weekend for simplicity
    combined_data['Country_Encoded'] = X['Country_Encoded'].mode()[0]  

    # Ensure the order of features matches the model's expectations
    combined_data = combined_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]

    # Generate predictions for the entire period
    combined_sales_predictions = decision_tree_model.predict(combined_data)

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(new_df.index, y, label='Historical Sales', color='blue')
    plt.plot(future_dates, combined_sales_predictions, label='Model Predictions (Decision Tree)', linestyle='--', color='red')
    plt.title('Historical and Forecasted Sales (Decision Tree)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    # Display the plot in Streamlit (if using Streamlit)
    st.pyplot(plt)


def monthly_sales_neural_network(df_monthly):
    # Prepare features and target variables
    X = df_monthly[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]  
    y = df_monthly['Sales_diff']  

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Neural Network model
    neural_network_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    neural_network_model.fit(X_train, y_train)

    # Prepare future dates for forecasting
    future_dates = pd.date_range(start=df_monthly['InvoiceDate'].max() + pd.DateOffset(months=1), periods=12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = df_monthly['UnitPrice'].mean()  # constant unit price
    combined_data['IsWeekend'] = combined_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    combined_data['Country_Encoded'] = df_monthly['Country_Encoded'].mode()[0]  # Most frequent value

    # Ensure the order of features matches the model's expectations
    combined_data = combined_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]

    # Generate predictions for the entire period
    combined_sales_predictions = neural_network_model.predict(combined_data)

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(df_monthly['InvoiceDate'], y, label='Historical Sales', color='blue')
    plt.plot(future_dates, combined_sales_predictions, label='Model Predictions (Neural Network)', linestyle='--', color='red')
    plt.title('Historical and Forecasted Monthly Sales (Neural Network)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    # Display the plot in Streamlit (if using Streamlit)
    st.pyplot(plt)



def yearly_sales_neural_network(new_df):
    # Prepare features and target variables
    X = new_df[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]  # Example features
    y = new_df['Sales_diff']  # Target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Neural Network model
    neural_network_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    neural_network_model.fit(X_train, y_train)

    # Prepare future dates for forecasting
    future_dates = pd.date_range(start=new_df.index.min(), periods=len(new_df) + 12, freq='M')
    combined_data = pd.DataFrame(index=future_dates)
    combined_data['Month'] = combined_data.index.month
    combined_data['DayOfWeek'] = combined_data.index.dayofweek
    combined_data['UnitPrice'] = X['UnitPrice'].mean()  # constant unit price
    combined_data['IsWeekend'] = 0  # non-weekend for simplicity
    combined_data['Country_Encoded'] = X['Country_Encoded'].mode()[0]  # Most frequent value

    # Ensure the order of features matches the model's expectations
    combined_data = combined_data[['Month', 'DayOfWeek', 'UnitPrice', 'IsWeekend', 'Country_Encoded']]

    # Generate predictions for the entire period
    combined_sales_predictions = neural_network_model.predict(combined_data)

    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(new_df.index, y, label='Historical Sales')
    plt.plot(future_dates, combined_sales_predictions, label='Model Predictions (Neural Network)', linestyle='--', color='red')
    plt.title('Historical and Forecasted Sales (Neural Network)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    # Display the plot in Streamlit (if using Streamlit)
    st.pyplot(plt)







def main():
    st.title("Sales Data Analysis")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_cleaned = process_data(uploaded_file)
        if df_cleaned is not None:
            st.title("Exploratory Data Analysis (EDA)")
            perform_eda(df_cleaned)

            st.title("Check feature and stationary")
            df_stationary = check_stationarity(df_cleaned)  # Get the stationary data
            df_monthly, new_df = prepare_data(df_stationary)  # Prepare the data for modeling

            st.title("Model Forecast")

            # Let the user choose the algorithm
            algorithm = st.selectbox(
                "Select the algorithm for forecasting",
                ("Linear Regression", "KNN", "Random Forest", "XGBoost", "Decision Tree", "Neural Network")
            )

            # Let the user choose between monthly and yearly forecast
            forecast_type = st.selectbox(
                "Select forecast type",
                ("Monthly", "Yearly")
            )

            # Call the appropriate model function based on user selection
            if algorithm == "Linear Regression":
                if forecast_type == "Monthly":
                    monthly_sales_linear_regression(df_monthly)
                else:
                    yearly_sales_linear_regression(new_df)

            elif algorithm == "KNN":
                if forecast_type == "Monthly":
                    monthly_sales_knn(df_monthly)
                else:
                    yearly_sales_knn(new_df)

            elif algorithm == "Random Forest":
                if forecast_type == "Monthly":
                    monthly_sales_random_forest(df_monthly)
                else:
                    yearly_sales_random_forest(new_df)

            elif algorithm == "XGBoost":
                if forecast_type == "Monthly":
                    monthly_sales_XGBoost(df_monthly)
                else:
                    yearly_sales_XGBoost(new_df)

            elif algorithm == "Decision Tree":
                if forecast_type == "Monthly":
                    monthly_sales_decision_tree(df_monthly)
                else:
                    yearly_sales_decision_tree(new_df)

            elif algorithm == "Neural Network":
                if forecast_type == "Monthly":
                    monthly_sales_neural_network(df_monthly)
                else:
                    yearly_sales_neural_network(new_df)

        else:
            st.error("Data could not be processed. Check the file format.")

if __name__ == "__main__":
    main()

