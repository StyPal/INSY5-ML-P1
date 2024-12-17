import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import glob
import matplotlib.pyplot as plt

# Configure Streamlit page
st.set_page_config(page_title="Spotify Listening Analysis", layout="wide")

# Title and description
st.title("Spotify Listening Duration Prediction")
st.write("""
This app predicts Spotify listening duration using a Random Forest Regressor. 
It evaluates model performance and provides interactive visualizations.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    json_files = glob.glob('data/spotify/*.json')
    dataframe = []
    for file in json_files:
        df = pd.read_json(file)
        dataframe.append(df)
    df = pd.concat(dataframe, ignore_index=True)

    df['ts'] = pd.to_datetime(df['ts'])
    df['year'] = df['ts'].dt.year
    df['month'] = df['ts'].dt.month
    df['day'] = df['ts'].dt.day
    df['hour'] = df['ts'].dt.hour
    df['weekday'] = df['ts'].dt.weekday
    df["listening_duration"] = df["ms_played"] / 60000
    df['prev_duration'] = df['listening_duration'].shift(1)
    df['rolling_mean'] = df['listening_duration'].rolling(window=3).mean()
    df['elapsed_seconds'] = (df['ts'] - df['ts'].min()).dt.total_seconds()

    return df

# Load data
st.sidebar.header("Data Loading")
if st.sidebar.button("Load Data"):
    df = load_data()
    st.write("### Raw Data")
    st.dataframe(df.head())
else:
    st.write("Click 'Load Data' in the sidebar to begin.")

# Feature selection and model training
if st.sidebar.button("Train Model"):
    df = load_data()
    X = df[['year', 'month', "day", "prev_duration", "rolling_mean", "weekday", 'hour']]
    y = df['listening_duration']

    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    smape = 100 * np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))
    tolerance = 0.1
    accuracy = np.mean(np.abs(y_test - y_pred) <= tolerance * y_test) * 100

    # Display metrics
    st.write("### Model Performance")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")
    st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
    st.write(f"**Symmetric MAPE (sMAPE):** {smape:.2f}%")
    st.write(f"**Accuracy (within 10% tolerance):** {accuracy:.2f}%")

    # Visualization
    st.write("### Visualizations")

    # Scatter plot: Actual vs Predicted
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Listening Duration")
    ax.set_ylabel("Predicted Listening Duration")
    ax.set_title("Actual vs Predicted Listening Duration")
    st.pyplot(fig)

    # Residual histogram
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=30, color='blue', alpha=0.7)
    ax.set_title("Residuals Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
