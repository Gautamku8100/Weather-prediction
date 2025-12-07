# 🌤️ Machine Learning-Powered Weather Forecast Dashboard

A comprehensive, Google-style web application built with **Streamlit** for displaying weather forecasts predicted by trained machine learning models.

---

## 🚀 Key Features

* **Google-Style UI:** Custom CSS for a clean, modern, and responsive user interface.
* **14-Day Forecast:** Displays daily weather predictions for temperature, precipitation, wind speed, and more.
* **Interactive Charts:** Uses Plotly to visualize temperature, precipitation, and wind/humidity trends over the forecast period.
* **ML Model Integration:** Dynamically loads and uses pre-trained `joblib` models and scalers for each city.
* **Data Download:** Allows users to download the 14-day forecast data as a CSV file.
* **Caching:** Utilizes Streamlit's `@st.cache_data` and `@st.cache_resource` for efficient model loading and data processing.

---

## ⚙️ Prerequisites

Before running the application, you need to have:

1.  **Trained ML Models:** You must have pre-trained machine learning models (e.g., a **Random Forest Regressor**) and their corresponding feature scalers (e.g., a **MinMaxScaler**).
2.  **Model Naming Convention:**
    * Model files must be named as `<city_name>\_model.pkl`.
    * Scaler files must be named as `<city_name>\_scaler.pkl`.
    * *Example: `London_model.pkl` and `London_scaler.pkl`.*
3.  **Required Libraries:**

    ```bash
    pip install streamlit pandas joblib plotly pathlib
    ```

---

## 🛠️ Setup and Configuration

### 1. Model Folder Setup

The application needs to know where your models are stored. You **must** update the `MODEL_FOLDER` variable in the script to point to the correct directory containing your `.pkl` files.

**Original Line (needs to be changed):**

```python
MODEL_FOLDER = Path("G:/My Drive/weather_data/city_models") 
Action: Replace the path with the actual, accessible path on your system (e.g., a relative path like "./models" or an absolute path).
```
### 2. Required Data
The ML models are expected to predict the following 8 features based on input features like day, month, and dayofweek:
```
temperature_2m

relative_humidity_2m

dew_point_2m

apparent_temperature

precipitation

rain

wind_speed_10m

cloud_cover
```
Ensure your models and scalers handle these 8 features correctly during the inverse transformation step.

# ▶️ How to Run the App
Save the provided Python code as a file (e.g., app.py).

Ensure your model files are in the directory specified by MODEL_FOLDER.

Open your terminal or command prompt.

Navigate to the directory where you saved app.py.

Run the Streamlit application:

Bash
```
streamlit run app.py
📂 Code Structure
app.py
```
This is the main application file, which contains all the Streamlit, data processing, and rendering logic.

## 1. Configuration & Setup
Sets up Streamlit page configuration (st.set_page_config).

Defines the custom CSS for the Google-style theme.

Defines the MODEL_FOLDER path.

## 2. WeatherPredictor Class
get_available_cities(): Scans the MODEL_FOLDER for _model.pkl files to automatically populate the city dropdown.

load_model(city): Caches and loads the specific city's _model.pkl and _scaler.pkl files using @st.cache_resource.

predict(model, scaler, date): Runs the prediction for a single day based on day, month, and dayofweek features.

predict_range(...): Generates a 14-day forecast by calling predict for a range of dates.

## 3. Rendering Functions
get_weather_condition(...): A utility to map predicted metrics (temp, precip, cloud) to a weather condition string and emoji.

render_current_weather(...): Renders the large current weather card and detail metrics using st.columns and HTML/Markdown.

render_daily_forecast(...): Renders the list of 14 daily forecast cards.

render_weather_charts(...): Generates and displays interactive Plotly charts for key weather trends (Temperature, Precipitation, Wind/Humidity).

## 4. main() Function
Handles app initialization, model folder checks, city selection (st.selectbox), date input (st.date_input), and model loading.

Manages the main layout using st.columns for the main content and side panel.