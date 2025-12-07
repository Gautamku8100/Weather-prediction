import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor

# -----------------------
# User settings
# -----------------------
# model_folder = "/content/drive/MyDrive/weather_data/city_models"
model_folder = "G:/My Drive/weather_data/city_models"


city = "Chennai"

# Set your specific date here (YYYY-MM-DD)
specific_date_str = "2025-10-18"
specific_date = datetime.strptime(specific_date_str, "%Y-%m-%d")

# -----------------------
# Paths
# -----------------------
model_file = os.path.join(model_folder, f"{city}_model.pkl")
scaler_file = os.path.join(model_folder, f"{city}_scaler.pkl")

# -----------------------
# Check existence
# -----------------------
if not os.path.exists(model_file) or not os.path.exists(scaler_file):
    raise FileNotFoundError(f"Model or scaler for {city} not found!")

# -----------------------
# Load model and scaler
# -----------------------
model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

# -----------------------
# Prepare features for the specific date
# -----------------------
X_date = pd.DataFrame({
    'day': [specific_date.day],
    'month': [specific_date.month],
    'dayofweek': [specific_date.weekday()]
})

# -----------------------
# Predict
# -----------------------
y_scaled_pred = model.predict(X_date)
y_pred = scaler.inverse_transform(y_scaled_pred)

# Map predictions to feature names
features_to_predict = [
    'temperature_2m','relative_humidity_2m','dew_point_2m',
    'apparent_temperature','precipitation','rain','wind_speed_10m','cloud_cover'
]

prediction = dict(zip(features_to_predict, y_pred[0]))

print(f"🌤️ Predicted weather for {city} on {specific_date.date()}:")
for k, v in prediction.items():
    print(f"{k}: {v:.2f}")
