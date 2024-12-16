import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import glob


json_files = glob.glob('data/spotify/*.json')
dataframe = []
print("Loading data...")
for file in json_files:
    df = pd.read_json(file)
    dataframe.append(df)
print("Data finished loading...")
df = pd.concat(dataframe, ignore_index=True)

df['ts'] = pd.to_datetime(df['ts'])
df['year'] = df['ts'].dt.year
df['month'] = df['ts'].dt.month
df['day'] = df['ts'].dt.day
df['hour'] = df['ts'].dt.hour
df['weekday'] = df['ts'].dt.weekday  
df["listening_duration"] = df["ms_played"] / 60000

# df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
# df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['prev_duration'] = df['listening_duration'].shift(1)
df['rolling_mean'] = df['listening_duration'].rolling(window=3).mean()

df['elapsed_seconds'] = (df['ts'] - df['ts'].min()).dt.total_seconds()

X = df[['year', 'month', "day",  "prev_duration", "rolling_mean", "weekday", 'hour']]
y = df['listening_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
y_true = y_test
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
tolerance = 0.1  # 10%
accuracy = np.mean(np.abs(y_true - y_pred) <= tolerance * y_true) * 100

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Symmetric MAPE (sMAPE): {smape:.2f}%")
print(f"Custom Accuracy (within 10% tolerance): {accuracy:.2f}%")