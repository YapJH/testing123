import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from pandas.plotting import scatter_matrix

# Set the style of seaborn
sns.set(style="whitegrid")

# Streamlit title and description
st.title('Sales Data Analysis Web App')
st.write('This app provides various sales data visualizations and insights.')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load data and preprocess without displaying preprocessing steps
    df = pd.read_csv(uploaded_file)

    # Preprocess Data: Parsing dates and generating new features
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.strftime('%Y-%m')
    df['Week'] = df['InvoiceDate'].dt.strftime('%Y-%U')
    df['Quarter'] = df['InvoiceDate'].dt.to_period('Q')
    df['Year'] = df['InvoiceDate'].dt.year
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'] >= 5
    df['Sales'] = df['UnitPrice'] * df['Quantity']

    # Hide preprocessing, just show the results

    st.subheader('Exploratory Data Analysis (EDA)')

    # Show dataset overview
    if st.checkbox('Show Raw Data'):
        st.write(df.head())

    # Monthly Sales Quantity Over Time
    monthly_sales = df.groupby('Month').agg({'Quantity': 'sum'}).reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales['Month'], monthly_sales['Quantity'], marker='o')
    plt.xticks(rotation=45)
    plt.title('Monthly Sales Quantity Over Time')
    plt.xlabel('Month')
    plt.ylabel('Total Quantity Sold')
    plt.grid(True)
    st.pyplot(plt)

    # Weekly Sales Quantity Over Time
    weekly_sales = df.groupby('Week').agg({'Quantity': 'sum'}).reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_sales['Week'], weekly_sales['Quantity'], marker='o')
    plt.xticks(rotation=45)
    plt.title('Weekly Sales Quantity Over Time')
    plt.xlabel('Week')
    plt.ylabel('Total Quantity Sold')
    plt.grid(True)
    st.pyplot(plt)

    # Quarterly Sales Quantity Over Time
    quarterly_sales = df.groupby('Quarter').agg({'Quantity': 'sum'}).reset_index()
    quarterly_sales['Quarter'] = quarterly_sales['Quarter'].astype(str)
    plt.figure(figsize=(12, 6))
    plt.plot(quarterly_sales['Quarter'], quarterly_sales['Quantity'], marker='o')
    plt.xticks(rotation=45)
    plt.title('Quarterly Sales Quantity Over Time')
    plt.xlabel('Quarter')
    plt.ylabel('Total Quantity Sold')
    plt.grid(True)
    st.pyplot(plt)

    st.subheader('Top 20 Countries by Revenue')

    # Calculate sales by country
    country_revenue = df.groupby('Country').agg({'Sales': 'sum'}).reset_index()
    top_country = country_revenue.sort_values(by='Sales', ascending=False).head(20)
    plt.figure(figsize=(15, 5))
    bars = plt.bar(top_country['Country'], top_country['Sales'], color='#37C6AB', edgecolor='black', linewidth=1)
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.title('Top 20 Countries by Revenue (Log Scale)')
    plt.xlabel('Country')
    plt.ylabel('Sales')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'${int(yval):,}', ha='center', va='bottom', fontsize=10, rotation=90)
    st.pyplot(plt)

    st.subheader('Sales Data Over Time')

    # Scatter plot of sales over time
    plt.figure(figsize=(14, 7))
    plt.scatter(df['InvoiceDate'], df['Sales'], color='blue', marker='o', s=10, alpha=0.5)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Sales Data Over Years')
    plt.xlabel('Year')
    plt.ylabel('Sales')
    plt.legend(['Individual Sales'], loc='upper left')
    st.pyplot(plt)

    st.subheader('Customer Behavior Analysis')

    # Top 10 customers by sales
    customer_sales = df.groupby('CustomerID')['Sales'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 4))
    customer_sales.head(10).plot(kind='bar')
    plt.title('Top 10 Customers by Sales')
    plt.xlabel('CustomerID')
    plt.ylabel('Total Sales')
    st.pyplot(plt)

    # Clustering for customer segmentation
    st.subheader('Customer Segmentation Using K-means')

    features = df[['Sales', 'Quantity', 'UnitPrice']]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    df['Cluster'] = kmeans.labels_
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='UnitPrice', y='Sales', hue='Cluster', data=df, palette='viridis')
    plt.title('Customer Segmentation by Sales and Unit Price')
    st.pyplot(plt)

    # Text analysis of descriptions
    st.subheader('Top 20 Most Frequent Terms in Product Descriptions')

    vect = CountVectorizer(stop_words='english')
    X = vect.fit_transform(df['Description'].dropna())
    word_counts = np.asarray(X.sum(axis=0)).ravel().tolist()
    counts_df = pd.DataFrame({'term': vect.get_feature_names_out(), 'occurrences': word_counts})
    top_terms = counts_df.sort_values(by='occurrences', ascending=False).head(20)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='term', y='occurrences', data=top_terms)
    plt.xticks(rotation=45)
    plt.title('Top 20 Most Frequent Terms in Product Descriptions')
    st.pyplot(plt)

    # Scatter plot matrix
    st.subheader('Scatter Matrix of Selected Variables')

    attributes = ['Sales', 'UnitPrice', 'Quantity']
    scatter_matrix(df[attributes], figsize=(12, 8), alpha=0.2)
    st.pyplot(plt)

    # Correlation heatmap
    st.subheader('Correlation Heatmap')

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)
