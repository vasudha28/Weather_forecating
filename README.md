## 📊 Project Overview

A comprehensive weather forecasting system that leverages machine learning to predict weather conditions, updates forecasts in real-time, and provides an intuitive dashboard with personalized recommendations through an AI assistant.

![image](https://github.com/user-attachments/assets/1b365bd8-d79a-476d-9344-b62db6ac25e1)



### Key Features

- **ML-Powered Weather Predictions**: Uses Random Forest models to forecast temperature, humidity, and pressure
- **Real-time Updates**: Automatically refreshes forecasts every 30 minutes
- **Interactive Dashboard**: Easy-to-read visualizations of current and forecasted weather conditions
- **AI Weather Assistant**: Provides personalized recommendations based on weather forecasts and user's profession

## 🔍 Problem Statement

Weather forecasting is critical for industries like agriculture, transportation, energy, and event planning. Accurate forecasts help organizations make informed decisions, reduce costs, and ensure safety. However, forecasting can be complex due to rapidly changing variables such as temperature, humidity, wind speed, and precipitation.

This project develops a machine learning model to predict weather conditions, schedules regular data updates, and visualizes forecast insights on a dashboard. By automating data ingestion and updates, stakeholders can access timely weather information in an intuitive format.

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Machine Learning**: Random Forest Classification and Regression models
- **APScheduler**: Handles automated model retraining and forecast updates
- **FastAPI**: Backend framework connecting frontend and ML models
- **Weather API**: Source for real-time weather data
- **Frontend**: HTML, CSS, and JavaScript for the user interface

## 📋 Project Components

### 1. ML Model for Weather Forecasting
- Supervised learning models trained on one year of historical data
- Random Forest Classification for weather condition prediction
- Random Forest Regression for temperature, humidity, and pressure forecasting

### 2. Scheduler for Regular Updates
- APScheduler runs at predefined intervals (every 30 minutes)
- Ingests new weather data from Weather API
- Processes and cleans the data
- Updates the model's input datasets and refines predictions

### 3. Interactive Dashboard
- Displays forecasts and current weather metrics
- Shows historical trends and real-time predictions
- Customized visualizations for easy interpretation

### 4. AI Weather Assistant
- Provides suggestions based on weather predictions
- Personalizes recommendations according to user's profession
- Uses current and forecasted weather data to give relevant advice

## 🚀 Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/weather-forecast-ml.git
   cd weather-forecast-ml
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the dashboard at:
   ```
   http://localhost:8000
   ```

## 💻 Usage

1. **View Weather Dashboard**:
   - Current weather conditions
   - Forecasted metrics for upcoming hours/days
   - Visual representation of weather trends

2. **Interact with AI Assistant**:
   - Select your profession for personalized recommendations
   - Get suggestions for planning your day based on weather conditions
   - Ask questions about current or forecasted weather

3. **Data Updates**:
   - System automatically refreshes data every 30 minutes
   - No manual intervention required for up-to-date forecasts

## 🔄 Data Flow

```
Weather API → Data Processing → ML Model Training → Forecast Generation → Dashboard Visualization & AI Assistant
                 ↑                                          |
                 └------ APScheduler (every 30 min) --------┘
```

## 📝 Future Enhancements

- Integration with additional weather data sources for improved accuracy
- Multilingual chat assistance for regional accessibility
- Customizable alert system for extreme weather conditions
- Detailed dashboard visualization with advanced analytics
- Mobile application for on-the-go access

## 👥 Contributors

- Vasudha https://github.com/vasudha28
- Ankit Singh https://github.com/Iankitsinghak
- Nimesh https://github.com/nimeshshukla2
---

*This project was developed as part of xto 10x Hackathon 2025.*
