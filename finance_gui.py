import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, filedialog, Button, Label
import tkinter as tk

def load_and_process_data():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    
    if not file_path:
        return
    
    df = pd.read_csv(file_path)
    
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

    # Store for plotting
    app_state['df'] = df
    app_state['df_scaled'] = df_scaled
    label_status.config(text="✅ Data loaded and processed!")

def plot_line_chart():
    if 'df' not in app_state:
        return
    df = app_state['df']
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Date', y='Close Price', hue='Stock Index')
    plt.title('Stock Index Close Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_heatmap():
    if 'df_scaled' not in app_state:
        return
    df_scaled = app_state['df_scaled']
    numeric_cols = df_scaled.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(14, 10))
    sns.heatmap(df_scaled[numeric_cols].corr(), cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap (Normalized Features)')
    plt.tight_layout()
    plt.show()

# App state to store data between functions
app_state = {}

# GUI Setup
root = Tk()
root.title("Finance & Economics GUI")
root.geometry("400x250")

label_title = Label(root, text="📊 Finance Dashboard", font=("Arial", 16))
label_title.pack(pady=10)

btn_load = Button(root, text="Load CSV and Process", command=load_and_process_data, width=25)
btn_load.pack(pady=10)

btn_line = Button(root, text="Plot Close Price Line Chart", command=plot_line_chart, width=25)
btn_line.pack(pady=10)

btn_heat = Button(root, text="Plot Correlation Heatmap", command=plot_heatmap, width=25)
btn_heat.pack(pady=10)

label_status = Label(root, text="", fg="green")
label_status.pack(pady=10)

root.mainloop()

