import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st

def load_and_process_data(uploaded_file):
    # Load and preprocess the data
@st.cache_data
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

# Streamlit App Layout
st.title("📊 Finance Dashboard")
st.sidebar.header("Upload Your CSV Data")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df, df_scaled = load_and_process_data(uploaded_file)
    st.success("Data loaded and processed successfully!")

    st.subheader("Raw Data")
    st.dataframe(df.head())

    st.subheader("Processed (Normalized) Data")
    st.dataframe(df_scaled.head())
else:
    st.warning("Please upload a CSV file to get started.")
