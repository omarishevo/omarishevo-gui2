

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import streamlit as st

def load_and_process_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Cleaning
    df = df.drop_duplicates()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Stock Index'] = df['Stock Index'].astype('category')

    # Feature engineering
    df['Price Range'] = df['Daily High'] - df['Daily Low']
    df['Price Change'] = df['Close Price'] - df['Open Price']

    # Normalization
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['Trading Volume'])
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, df_scaled

def plot_line_chart(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Close Price', hue='Stock Index')
    plt.title('Stock Index Close Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

def plot_heatmap(df_scaled):
    plt.figure(figsize=(14, 10))
    numeric_cols = df_scaled.select_dtypes(include=['float64', 'int64']).columns
    sns.heatmap(df_scaled[numeric_cols].corr(), cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap (Normalized Features)')
    plt.tight_layout()
    st.pyplot(plt)

# Streamlit App Layout
st.title("📊 Finance Dashboard")
st.sidebar.header("Upload Your CSV Data")

# File Upload
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Process the file and display status
    df, df_scaled = load_and_process_data(uploaded_file)
    st.success("Data loaded and processed successfully!")

    # Option to plot charts
    plot_option = st.sidebar.selectbox("Choose a plot", ["Select", "Line Chart", "Correlation Heatmap"])

    if plot_option == "Line Chart":
        plot_line_chart(df)

    elif plot_option == "Correlation Heatmap":
        plot_heatmap(df_scaled)
else:
    st.warning("Please upload a CSV file to get started.")

