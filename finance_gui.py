import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Use @st.cache_data to cache the function to optimize performance (for Streamlit >=1.0)
@st.cache_data
def load_and_process_data(uploaded_file):
    # Load CSV file directly from uploaded_file
    df = pd.read_csv(uploaded_file)

    # Basic Cleaning
    df = df.drop_duplicates()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # Ensure 'Stock Index' exists and convert to category
    if 'Stock Index' in df.columns:
        df['Stock Index'] = df['Stock Index'].astype('category')

    # Feature engineering
    if all(col in df.columns for col in ['Daily High', 'Daily Low', 'Close Price', 'Open Price']):
        df['Price Range'] = df['Daily High'] - df['Daily Low']
        df['Price Change'] = df['Close Price'] - df['Open Price']

    # Normalize numerical columns (excluding 'Trading Volume' if it exists)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['Trading Volume'])
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, df_scaled

# ------------------ Streamlit Layout ------------------

st.title("📈 Finance & Economics Dashboard")
st.sidebar.header("Upload Your Dataset")

# File uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load and process the uploaded data
        df, df_scaled = load_and_process_data(uploaded_file)
        st.success("✅ Dataset uploaded and processed!")

        # Show raw data
        st.subheader("🗃️ Raw Data")
        st.dataframe(df.head())

        # Show processed data
        st.subheader("📉 Normalized Data")
        st.dataframe(df_scaled.head())
    except Exception as e:
        st.error(f"❌ Error loading the file: {e}")
else:
    st.info("📤 Please upload a CSV file to begin.")
