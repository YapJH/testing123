import streamlit as st
import pandas as pd

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

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

    # Display the count of remaining missing values across columns
    missing_values = df.isnull().sum()
    st.write("Missing values in each column:", missing_values)

    # Drop rows where 'Quantity', 'UnitPrice', or 'CustomerID' is negative or zero
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['CustomerID'] > 0)]

    # Show the cleaned DataFrame
    st.write("Cleaned Data", df)

# This code snippet integrates file upload, data cleaning, and display within a Streamlit app.
