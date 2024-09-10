import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
from pandas.plotting import scatter_matrix
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

# Set the style of seaborn
sns.set(style="whitegrid")

# Streamlit title and description
st.title('Sales Data Analysis Web App')
st.write('This app provides various sales data visualizations and insights.')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Preprocessing: Mapping StockCode to most common Description and cleaning
    stockcode_description_map = df.groupby('StockCode')['Description'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None).to_dict()

    # Fill missing Description values based on the StockCode
    df['Description'] = df.apply(
        lambda row: stockcode_description_map[row['StockCode']] if pd.isnull(row['Description']) else row['Description'],
        axis=1
    )

    # Drop rows where Description or CustomerID are missing
    df = df.dropna(subset=['Description', 'CustomerID'])

    # Drop rows where Quantity is negative
    df = df[df['Quantity','UnitPrice', 'CustomerID'] > 0]

    # Hide preprocessing, just show the results

    st.subheader('Exploratory Data Analysis (EDA)')

    # Show dataset overview
    if st.checkbox('Show Raw Data'):
        st.write(df.head())

    # --- Sales Data Visualizations ---
    st.subheader('Sales Data Over Time')

    # Sales Data Visualization Over Years
    plt.figure(figsize=(14, 7))
    plt.scatter(df['InvoiceDate'], df['Sales'], color='blue', marker='o', s=10, alpha=0.5)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Sales Data Visualization Over Years')
    plt.xlabel('Year')
    plt.ylabel('Sales')
    plt.legend(['Individual Sales'], loc='upper left')
    st.pyplot(plt)

    # --- Pivot Table Visualization ---
    st.subheader('Average Sales by Month and Day of Week')
    pivot_table = df.pivot_table(values='Sales', index='Month', columns='DayOfWeek', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt=".0f")
    plt.title('Average Sales by Month and Day of Week')
    st.pyplot(plt)

    # --- Correlation Heatmap ---
    st.subheader('Correlation Heatmap')

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

    # --- Frequency of Categorical Features ---
    st.subheader('Frequency of Categorical Features')

    categorical_features = ['Country', 'Week', 'DayOfWeek', 'IsWeekend']
    for feature in categorical_features:
        plt.figure(figsize=(10, 4))
        chart = sns.countplot(x=feature, data=df, order=df[feature].value_counts().index)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
        plt.title(f'Frequency of {feature}')
        st.pyplot(plt)

    # --- Scatter Plot Matrix ---
    st.subheader('Scatter Plot Matrix')

    attributes = ['Sales', 'UnitPrice', 'Quantity', 'Year']
    scatter_matrix(df[attributes], figsize=(12, 8), alpha=0.2)
    st.pyplot(plt)

    # --- K-means for Customer Segmentation ---
    st.subheader('Customer Segmentation Using K-means')

    features = df[['Sales', 'Quantity', 'UnitPrice']]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    df['Cluster'] = kmeans.labels_
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='UnitPrice', y='Sales', hue='Cluster', data=df, palette='viridis')
    plt.title('Customer Segmentation by Sales and Unit Price')
    st.pyplot(plt)

    # --- Text Analysis of Descriptions ---
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

    # --- Cross-tabulation of Weekday vs. IsWeekend ---
    st.subheader('Cross-Tabulation of Weekday vs. IsWeekend')

    weekend_cross = pd.crosstab(index=df['DayOfWeek'], columns=df['IsWeekend'])
    st.write(weekend_cross)

    # --- Sales Quantity by Country Choropleth ---
    st.subheader('Sales Quantity by Country')

    country_sales = df.groupby('Country')['Quantity'].sum().reset_index()
    fig = px.choropleth(country_sales, locations="Country",
                        locationmode='country names',
                        color="Quantity",
                        hover_name="Country",
                        color_continuous_scale="Viridis",
                        title="Sales Quantity by Country")
    st.plotly_chart(fig)
