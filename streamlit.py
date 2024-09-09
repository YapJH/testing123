import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Title of the Streamlit app
st.title("Data Analysis and Forecasting App")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# If a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataset
    st.write("Dataset Preview:")
    st.write(df.head())

    # Data cleaning step (you can add more steps based on your notebook)
    st.write("Data Cleaning:")
    st.write(df.describe())

    # Select features and target for modeling
    feature_columns = st.multiselect("Select Features", df.columns)
    target_column = st.selectbox("Select Target", df.columns)

    if feature_columns and target_column:
        X = df[feature_columns]
        y = df[target_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a model (example: Linear Regression)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display metrics
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")

        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.legend()
        st.pyplot(plt)
