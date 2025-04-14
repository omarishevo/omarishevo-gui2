import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Use @st.cache_data to cache the data processing function
@st.cache_data
def load_and_process_data(uploaded_file):
    # Load and preprocess the data
    df = pd.read_csv(r"C:\Users\Administrator\Desktop\class work\finance_economics_dataset.csv")

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

# File uploader widget in the sidebar
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Call the load_and_process_data function if a file is uploaded
    df, df_scaled = load_and_process_data(uploaded_file)
    st.success("Data loaded and processed successfully!")

    # Display the raw data
    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Display the processed (normalized) data
    st.subheader("Processed (Normalized) Data")
    st.dataframe(df_scaled.head())
else:
    # Show a warning if no file is uploaded
    st.warning("Please upload a CSV file to get started.")
v
