import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from pandas.plotting import scatter_matrix
import numpy as np
import plotly.express as px

# Set up page configuration and style
st.set_page_config(page_title='Sales Data Analysis', layout='wide')
sns.set(style="whitegrid")

def load_data():
    # Assuming the user uploads a file named 'no_outlier.csv'
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    return None

df = load_data()

if df is not None:
    st.write("Data Loaded Successfully!")
    st.write(df.head())  # Show the first few rows of the DataFrame

    # EDA Section
    st.header("Exploratory Data Analysis")

    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Display monthly sales quantity over time
    st.subheader("Monthly Sales Quantity Over Time")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.strftime('%Y-%m')
    monthly_sales = df.groupby('Month').agg({'Quantity': 'sum'}).reset_index()
    fig, ax = plt.subplots()
    ax.plot(monthly_sales['Month'], monthly_sales['Quantity'], marker='o')
    ax.set_title('Monthly Sales Quantity')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Quantity Sold')
    ax.grid(True)
    st.pyplot(fig)

    # Feature and Stationarity Checks
    st.header("Feature and Stationarity Checks")
    # Assuming some checks are performed here, provide a placeholder
    st.write("Feature and stationarity checks to be implemented.")

    # Model Outputs
    st.header("Model Outputs")
    # K-means clustering example
    features = df[['Quantity', 'UnitPrice']]  # Assume these features for simplicity
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    df['Cluster'] = kmeans.labels_
    
    # Visualize the clusters
    fig, ax = plt.subplots()
    sns.scatterplot(x='UnitPrice', y='Quantity', hue='Cluster', data=df, ax=ax, palette='viridis')
    ax.set_title('Customer Segmentation by Sales and Unit Price')
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

# Additional interactive elements can be added based on user needs
