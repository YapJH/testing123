import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import kpss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Title
st.title("Data Analysis and Modeling App")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Show the raw data
    st.subheader("Raw Data")
    st.write(data.head())
    
    # EDA Section
    st.subheader("Exploratory Data Analysis")
    st.write("Basic Data Information")
    st.write(data.describe())
    
    # Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)
    
    # Feature Selection
    st.subheader("Feature Selection")
    feature_columns = st.multiselect("Select Features", data.columns)
    target_column = st.selectbox("Select Target", data.columns)
    
    if feature_columns and target_column:
        X = data[feature_columns]
        y = data[target_column]

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Check stationarity (KPSS Test)
        st.subheader("Stationarity Check (KPSS Test)")
        def kpss_test(series):
            statistic, p_value, n_lags, critical_values = kpss(series)
            return p_value
        
        for feature in feature_columns:
            st.write(f"KPSS Test for {feature}:")
            p_value = kpss_test(X_train[feature])
            st.write(f"P-value: {p_value}")
            if p_value < 0.05:
                st.write("Series is non-stationary.")
            else:
                st.write("Series is stationary.")
        
        # Modeling
        st.subheader("Model Training")
        model_type = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])
        
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Random Forest":
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Display performance
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")
        
        # Visualize predictions vs actual
        st.subheader("Predictions vs Actual")
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.legend()
        st.pyplot(plt)
