import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import kpss
import seaborn as sns

import streamlit as st
import pandas as pd

# Title for the app
st.title("Data Cleaning and Preparation")

# File upload section
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Function to process the uploaded data
def process_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
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

        return df

    return None

# Process the data
df = process_data(uploaded_file)

# Display the cleaned data
if df is not None:
    st.write("Cleaned Data Preview:", df.head())
else:
    st.write("No data to display. Please upload a file with the correct format.")
