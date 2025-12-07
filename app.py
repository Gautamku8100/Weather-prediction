import streamlit as st
import pandas as pd
import os
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path # Use pathlib for better path management

# --- Configuration & Setup ---

# Set a sensible base directory for the models.
# IMPORTANT: This path must exist and contain your model files (*_model.pkl and *_scaler.pkl).
# Replace this with your actual folder path. 'city_models' is used as a relative example.
MODEL_FOLDER = Path("G:/My Drive/weather_data/city_models") 

# Page configuration
st.set_page_config(
    page_title="Weather Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Google-style design (Kept as is)
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(to bottom, #e3f2fd, #ffffff);
    }
    
    /* Header */
    .weather-header {
        background: white;
        padding: 20px 40px;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    
    .city-name {
        font-size: 2.5rem;
        font-weight: 600;
        color: #202124;
        margin: 0;
    }
    
    .current-date {
        font-size: 1.1rem;
        color: #5f6368;
        margin-top: 5px;
    }
    
    /* Current weather card */
    .current-weather {
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        text-align: center;
        margin-bottom: 30px;
    }
    
    .current-temp {
        font-size: 5rem;
        font-weight: 300;
        color: #202124;
        margin: 20px 0;
    }
    
    .weather-icon-large {
        font-size: 6rem;
    }
    
    .weather-desc {
        font-size: 1.3rem;
        color: #5f6368;
        margin: 10px 0;
    }
    
    /* Hourly forecast */
    .hourly-scroll {
        display: flex;
        overflow-x: auto;
        gap: 15px;
        padding: 20px 10px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    
    .hourly-card {
        min-width: 100px;
        background: #f8f9fa;
        padding: 20px 15px;
        border-radius: 12px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .hourly-card:hover {
        background: #e3f2fd;
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Daily forecast cards */
    .daily-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    
    .daily-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    .day-name {
        font-size: 1.3rem;
        font-weight: 600;
        color: #202124;
    }
    
    .day-date {
        font-size: 0.95rem;
        color: #5f6368;
    }
    
    /* Details grid */
    .detail-item {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }
    
    .detail-label {
        font-size: 0.85rem;
        color: #5f6368;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .detail-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #202124;
        margin-top: 5px;
    }
    
    /* Buttons */
    .stButton>button {
        background: #1a73e8;
        color: white;
        border: none;
        border-radius: 24px;
        padding: 10px 24px;
        font-size: 0.95rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #1557b0;
        box-shadow: 0 4px 12px rgba(26, 115, 232, 0.3);
    }
    
    /* Search box */
    .stSelectbox {
        border-radius: 24px;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- WeatherPredictor Class ---

class WeatherPredictor:
    """Handles model loading and daily prediction logic."""

    def __init__(self, model_folder):
        self.model_folder = Path(model_folder)
        self.features_to_predict = [
            'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
            'apparent_temperature', 'precipitation', 'rain', 
            'wind_speed_10m', 'cloud_cover'
        ]
        
    @st.cache_data
    def get_available_cities(_self):
        """Finds available city models in the specified folder."""
        if not _self.model_folder.exists():
            return []
        
        # Use glob to find files matching the pattern
        model_files = [f.name for f in _self.model_folder.glob('*_model.pkl')]
        return sorted([f.replace('_model.pkl', '') for f in model_files])
    
    @st.cache_resource
    def load_model(_self, city):
        """Loads the model and scaler for a given city."""
        model_file = _self.model_folder / f"{city}_model.pkl"
        scaler_file = _self.model_folder / f"{city}_scaler.pkl"
        
        if not model_file.exists() or not scaler_file.exists():
            return None, None
        
        try:
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            return model, scaler
        except Exception as e:
            st.error(f"Error loading model or scaler for {city}: {e}")
            return None, None
    
    def predict(self, model, scaler, date):
        """Generates a single day's prediction."""
        X_date = pd.DataFrame({
            'day': [date.day],
            'month': [date.month],
            'dayofweek': [date.weekday()]
        })
        
        try:
            y_scaled_pred = model.predict(X_date)
            y_pred = scaler.inverse_transform(y_scaled_pred)
            
            return dict(zip(self.features_to_predict, y_pred[0]))
        except Exception as e:
            # Return defaults on failure
            st.warning(f"Prediction failed for date {date}: {e}")
            return dict(zip(self.features_to_predict, [20.0, 70.0, 10.0, 22.0, 0.0, 0.0, 10.0, 50.0]))
    
    def predict_range(self, model, scaler, start_date, days=14):
        """Generates predictions for a range of days."""
        predictions = []
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            pred = self.predict(model, scaler, current_date)
            pred['date'] = current_date
            pred['day_name'] = current_date.strftime('%A')
            pred['date_str'] = current_date.strftime('%b %d')
            predictions.append(pred)
        return pd.DataFrame(predictions)

# --- Utility Functions ---

def get_weather_condition(temp, precipitation, cloud_cover):
    """Determine weather condition based on metrics."""
    temp = round(temp)
    precipitation = round(precipitation, 1)
    cloud_cover = round(cloud_cover)

    if precipitation >= 5:
        return "Heavy Rain", "🌧️"
    elif precipitation > 1:
        return "Rain", "☔"
    elif cloud_cover > 85:
        return "Overcast", "☁️"
    elif cloud_cover > 40 and precipitation > 0.1:
        return "Showers", "🌦️"
    elif cloud_cover > 40:
        return "Partly Cloudy", "⛅"
    elif temp >= 30:
        return "Hot & Sunny", "🌞"
    elif temp < 10:
        return "Chilly", "🥶"
    else:
        return "Clear/Sunny", "☀️"

# --- Rendering Functions ---

def render_current_weather(city, prediction, date):
    """Render the current weather section."""
    try:
        temp = prediction['temperature_2m']
        apparent_temp = prediction['apparent_temperature']
        humidity = prediction['relative_humidity_2m']
        wind = prediction['wind_speed_10m']
        precip = prediction['precipitation']
        cloud = prediction['cloud_cover']
    except TypeError:
        st.error("Prediction data is corrupted or missing.")
        return

    condition, icon = get_weather_condition(temp, precip, cloud)
    
    st.markdown(f"""
    <div class="current-weather">
        <div class="weather-icon-large">{icon}</div>
        <div class="current-temp">{temp:.0f}°C</div>
        <div class="weather-desc">{condition}</div>
        <div style="font-size: 1rem; color: #5f6368; margin-top: 10px;">
            Feels like {apparent_temp:.0f}°C
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    details = [
        ("💧 Humidity", f"{humidity:.0f}%", col1),
        ("💨 Wind", f"{wind:.0f} km/h", col2),
        ("🌧️ Precipitation", f"{precip:.1f} mm", col3),
        ("☁️ Cloud Cover", f"{cloud:.0f}%", col4),
    ]
    
    for label, value, col in details:
        with col:
            st.markdown(f"""
            <div class="detail-item">
                <div class="detail-label">{label}</div>
                <div class="detail-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)


def render_daily_forecast(predictions_df):
    """Render 14-day forecast."""
    st.markdown("### 14-Day Forecast")
    
    for idx, row in predictions_df.iterrows():
        try:
            temp = row['temperature_2m']
            precip = row['precipitation']
            cloud = row['cloud_cover']
            low_temp = temp - 5 # Simulated low temp
        except TypeError:
            continue

        condition, icon = get_weather_condition(temp, precip, cloud)
        
        with st.container():
            st.markdown(f"""
            <div class="daily-card">
            <div class="st-row" style="display: flex; align-items: center; justify-content: space-between;">
                <div style="flex: 2;">
                    <div class="day-name">{"Today" if idx == 0 else row['day_name']}</div>
                    <div class="day-date">{row['date_str']}</div>
                </div>
                <div style="flex: 1; text-align: center; font-size: 2.5rem;">{icon}</div>
                <div style="flex: 2; text-align: center;">
                    <div style="font-size: 1.1rem; color: #5f6368;">High / Low</div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: #202124;">
                        {temp:.0f}° / {low_temp:.0f}°
                    </div>
                </div>
                <div style="flex: 2; text-align: center;">
                    <div style="font-size: 1.1rem; color: #5f6368;">Precipitation</div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: #202124;">
                        {precip:.1f} mm
                    </div>
                </div>
                <div style="flex: 2; text-align: center;">
                    <div style="font-size: 1.1rem; color: #5f6368;">Wind</div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: #202124;">
                        {row['wind_speed_10m']:.0f} km/h
                    </div>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<hr style="margin: 0;">', unsafe_allow_html=True)

# THE MISSING FUNCTION DEFINITION IS ADDED HERE
def render_weather_charts(predictions_df):
    """Render interactive weather charts."""
    st.markdown("### Weather Trends")
    
    if predictions_df.empty:
        st.info("No data available for charts.")
        return

    tab1, tab2, tab3 = st.tabs(["🌡️ Temperature", "🌧️ Precipitation", "💨 Wind & Humidity"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['temperature_2m'],
            mode='lines+markers',
            name='Temperature',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['apparent_temperature'],
            mode='lines+markers',
            name='Feels Like',
            line=dict(color='#FFA07A', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="Temperature Forecast",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            hovermode='x unified',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12, color='black')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=predictions_df['date'],
            y=predictions_df['precipitation'],
            name='Precipitation',
            marker_color='#4A90E2'
        ))
        fig.update_layout(
            title="Precipitation Forecast",
            xaxis_title="Date",
            yaxis_title="Precipitation (mm)",
            hovermode='x unified',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12, color='black')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['wind_speed_10m'],
            mode='lines+markers',
            name='Wind Speed',
            line=dict(color='#50C878', width=3),
            marker=dict(size=8),
            yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['relative_humidity_2m'],
            mode='lines+markers',
            name='Humidity',
            line=dict(color='#87CEEB', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        fig.update_layout(
            title="Wind Speed & Humidity",
            xaxis_title="Date",
            yaxis=dict(
                title="Wind Speed (km/h)",
                title_font=dict(color='#50C878')
            ),
            yaxis2=dict(
                title="Humidity (%)",
                overlaying='y',
                side='right',
                title_font=dict(color='#87CEEB')
            ),
            hovermode='x unified',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12, color='black')
        )
        st.plotly_chart(fig, use_container_width=True)


# --- Main Application Logic ---

def main():
    # Check if the model folder exists
    if not MODEL_FOLDER.exists():
        st.error(f"❌ Model folder not found at: `{MODEL_FOLDER.resolve()}`. Please check the path and folder contents.")
        st.stop()
        
    # Initialize predictor using st.session_state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = WeatherPredictor(MODEL_FOLDER)
    
    # Top navigation bar
    col1, col2, col3 = st.columns([3, 2, 1])
    
    cities = st.session_state.predictor.get_available_cities()
    
    with col1:
        if not cities:
            st.error("❌ No city models found in the folder. Please train and save your models.")
            st.stop()
        
        selected_city = st.selectbox(
            "🔍 Search for a city",
            cities,
            label_visibility="collapsed",
            key="city_selector"
        )
    
    with col2:
        selected_date_raw = st.date_input(
            "📅 Select date",
            value=datetime.now().date(),
            label_visibility="collapsed",
            key="date_selector",
            min_value=datetime.now().date()
        )
        selected_date = datetime.combine(selected_date_raw, datetime.min.time())

    with col3:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_button"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Header
    st.markdown(f"""
    <div class="weather-header">
        <div class="city-name">📍 {selected_city}</div>
        <div class="current-date">{datetime.now().strftime('%A, %B %d, %Y • %I:%M %p')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and make predictions
    model, scaler = None, None
    with st.spinner(f'🌤️ Loading and predicting weather for {selected_city}...'):
        model, scaler = st.session_state.predictor.load_model(selected_city)
        
        if model is None:
            st.error(f"❌ Model or Scaler file not found for **{selected_city}**.")
            st.stop()
        
        # Get 14-day forecast
        predictions_df = st.session_state.predictor.predict_range(model, scaler, selected_date, days=14)
        
        if predictions_df.empty:
            st.error("Prediction failed. Dataframe is empty.")
            st.stop()
            
        current_prediction = predictions_df.iloc[0].to_dict()
    
    # --- Rendering Layout ---
    
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.markdown("## Today's Forecast", unsafe_allow_html=True)
        render_current_weather(selected_city, current_prediction, selected_date)
        

        
        render_daily_forecast(predictions_df)
    
    with col_side:
        st.markdown("## Global Stats", unsafe_allow_html=True)
        
        # Quick stats
        max_temp = predictions_df['temperature_2m'].max()
        min_temp = predictions_df['temperature_2m'].min()
        avg_temp = predictions_df['temperature_2m'].mean()
        
        st.metric("Average Temperature (14d)", f"{avg_temp:.1f}°C", f"{max_temp-min_temp:.1f}°C range")
        st.metric("Total Precipitation (14d)", f"{predictions_df['precipitation'].sum():.1f} mm")
        st.metric("Max Wind Speed (14d)", f"{predictions_df['wind_speed_10m'].max():.1f} km/h")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Mini chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['temperature_2m'],
            mode='lines',
            line=dict(color='#1a73e8', width=2),
            fill='tozeroy',
            fillcolor='rgba(26, 115, 232, 0.1)'
        ))
        fig.update_layout(
            title="14-Day Temperature Overview",
            xaxis_title="",
            yaxis_title="°C",
            height=250,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='black'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Data",
            data=csv,
            file_name=f"{selected_city}_forecast_{selected_date.strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Detailed charts section
    st.markdown("<br><br>", unsafe_allow_html=True)
    render_weather_charts(predictions_df)

if __name__ == "__main__":
    main()