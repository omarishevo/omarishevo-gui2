import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Election Forecasting", layout="wide")
st.title("🗳 LSTM Election Forecasting Dashboard (Date Optional)")

uploaded_file = st.file_uploader(
    "Upload Historical Polling CSV (Columns: Party Vote Shares)", type="csv"
)

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Optional Date handling
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")
        st.subheader("Dataset Preview (with Date)")
    else:
        st.subheader("Dataset Preview (No Date column)")
    
    st.dataframe(df.head())

    # Select parties to forecast
    parties = st.multiselect(
        "Select Parties to Forecast", 
        options=df.columns.tolist() if "Date" not in df.columns else [c for c in df.columns if c != "Date"],
        default=df.columns.tolist() if "Date" not in df.columns else [c for c in df.columns if c != "Date"]
    )

    sequence_length = st.slider("Sequence Length (rows)", 2, 24, 6)
    forecast_periods = st.slider("Forecast Periods (future steps)", 1, 12, 3)

    predictions_dict = {}

    for party in parties:
        votes = df[party].values.reshape(-1,1)
        scaler = MinMaxScaler()
        votes_scaled = scaler.fit_transform(votes)

        X, y = create_sequences(votes_scaled, sequence_length)

        # Train-test split
        split = int(len(X)*0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Build LSTM
        model = Sequential()
        model.add(LSTM(50, input_shape=(sequence_length,1)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0)

        # Forecast future
        last_seq = votes_scaled[-sequence_length:]
        future_preds = []
        for _ in range(forecast_periods):
            next_pred = model.predict(last_seq.reshape(1,sequence_length,1))
            future_preds.append(next_pred[0,0])
            last_seq = np.append(last_seq[1:], next_pred, axis=0)

        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))
        predictions_dict[party] = future_preds.flatten()

    st.subheader("Forecasted Vote Shares")
    forecast_df = pd.DataFrame(predictions_dict)
    st.line_chart(forecast_df)
    st.write(forecast_df)

else:
    st.info("Upload a CSV file with party vote shares to begin.")
