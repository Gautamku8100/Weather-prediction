# 🌦 Weather Prediction Project

---

## 📂 Dataset Information

**Dataset Name:** Indian Cities Weather Dataset  
**Dataset Source:** Open-Meteo Historical Weather API  

---

## 📋 Dataset Summary

- **Rows:** 123,936  
- **Columns:** 21  
- **Numerical Features:** 20  
- **Categorical Features:** 1  
- **Datetime Features:** 0  

---

## 📊 Basic Statistics (Sample)

| Feature              | Mean   | Median | Mode  | Min   | Max    |
|----------------------|--------|--------|-------|-------|--------|
| temperature_2m       | 27.92  | 27.76  | 28.16 | 15.26 | 41.96  |
| relative_humidity_2m | 74.53  | 76.90  | 87.27 | 22.24 | 100.00 |
| dew_point_2m         | 22.61  | 23.06  | 23.96 | 10.86 | 29.36  |
| apparent_temperature | 31.63  | 31.71  | 27.37 | 15.89 | 46.94  |
| precipitation        | 0.13   | 0.00   | 0.00  | 0.00  | 29.70  |
| rain                 | 0.13   | 0.00   | 0.00  | 0.00  | 29.70  |
| snowfall             | 0.00   | 0.00   | 0.00  | 0.00  | 0.00   |
| snow_depth           | 0.00   | 0.00   | 0.00  | 0.00  | 0.00   |
| pressure_msl         | 1008.7 | 1008.7 | 1005.9| 986.9 | 1020.5 |
| surface_pressure     | 1007.8 | 1007.7 | 1005.5| 985.9 | 1019.5 |

---

## 📋 Sample Table

| temperature_2m | relative_humidity_2m | precipitation | wind_speed_10m |
|----------------|-----------------------|---------------|----------------|
| 21.06          | 94.31                 | 0.0           | 6.28           |
| 21.11          | 94.02                 | 0.0           | 6.95           |
| 22.26          | 92.35                 | 0.0           | 8.70           |
| 24.91          | 82.42                 | 0.0           | 11.21          |
| 26.41          | 71.58                 | 0.0           | 14.84          |

---

## 🔑 Key Findings

- Majority of **precipitation values are 0** (most hours have no rainfall).  
- Some features (**temperature, humidity**) vary widely and show strong **seasonal trends**.  
- **Cloud cover, humidity, and precipitation** are correlated (rainy periods = high humidity & cloud cover).  
- **Wind speed and gusts** vary by city (coastal cities like Mumbai & Chennai show stronger winds).  
- Dataset is **time-series**, so values are autocorrelated (today’s weather depends on yesterday’s).  
- **Class imbalance** if predicting rainfall/conditions (e.g., “No Rain” dominates).  

---

## 🛠 Tools Used

| Tool          | Purpose                                                   |
|---------------|-----------------------------------------------------------|
| **Pandas**    | Data handling & summary                                   |
| **Matplotlib**| Basic visualization (histograms, line plots)              |
| **Seaborn**   | Enhanced statistical visuals (correlation, distributions) |
| **Scikit-learn** | Modeling & prediction (ML)                             |

---

## 📊 Simple Visuals for Poster

### 🔹 Histogram – Temperature Distribution
- **X-axis:** Temperature (°C)  
- **Y-axis:** Count (frequency of hours)  
- **Insight:** Most temperatures cluster around seasonal averages (25–30°C).  

---

### 🔹 Pie Chart – Top 5 Weather Conditions
- **Labels:** No Rain, Light Rain, Heavy Rain, Cloudy, Clear  
- **Insight:** “No Rain” dominates, followed by “Cloudy”.  

---

### 🔹 Bar Graph – Top 10 Cities by Avg Rainfall
- **X-axis:** Cities (Delhi, Mumbai, Chennai, etc.)  
- **Y-axis:** Average rainfall (mm)  
- **Insight:** Coastal cities show much higher rainfall than inland ones.  

---

### 🔹 Scatter Plot – Temperature vs Humidity
- **X-axis:** Temperature (°C)  
- **Y-axis:** Humidity (%)  
- **Insight:** Inverse relationship (higher temperature → lower humidity).  

---

### 🔹 Box Plot – Seasonal Temperature Variation
- **X-axis:** Season (Winter, Summer, Monsoon, Post-Monsoon)  
- **Y-axis:** Temperature (°C)  
- **Insight:** Variability highest in Monsoon, lowest in Winter.  
