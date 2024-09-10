import streamlit as st
import pandas as pd

def process_data(uploaded_file):
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

        # Remove rows with non-positive values in specified columns
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['CustomerID'] > 0)]

        return df
    return None

def display_data(df):
    if df is not None:
        # Display the count of remaining missing values across columns
        missing_values = df.isnull().sum()
        st.write("Missing values in each column:", missing_values)

        # Show the cleaned DataFrame
        st.write("Cleaned Data", df)
    else:
        st.write("No data to display. Please upload a valid CSV file.")

def main():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    cleaned_data = process_data(uploaded_file)
    display_data(cleaned_data)

if __name__ == "__main__":
    main()
