from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime, timedelta
import pytz
from pydantic import BaseModel
from weather import get_current_weather, read_historical_data, prepare_data, train_rain_model, prepare_regression_data, train_regression_model, predict_future
import json
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from typing import Dict, List, Any

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# In-memory storage for weather data
weather_cache: Dict[str, Dict[str, Any]] = {}
last_update_times: Dict[str, str] = {}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, city: str):
        await websocket.accept()
        if city not in self.active_connections:
            self.active_connections[city] = []
        self.active_connections[city].append(websocket)

    def disconnect(self, websocket: WebSocket, city: str):
        if city in self.active_connections:
            self.active_connections[city].remove(websocket)

    async def broadcast_to_city(self, city: str, message: str):
        if city in self.active_connections:
            for connection in self.active_connections[city]:
                await connection.send_text(message)

manager = ConnectionManager()

# Initialize scheduler
scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def startup_event():
    scheduler.start()

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()

class CityRequest(BaseModel):
    city: str

async def update_weather_data(city: str):
    """Fetch updated weather data for a city"""
    try:
        # Get and process weather data similar to your original endpoint
        current_weather = get_current_weather(city)
        historical_data = read_historical_data('weather.csv')

        x, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(x, y)

        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 348.75, 360), ("N", 0, 11.25), ("NNE", 11.25, 33.75),
            ("NE", 33.75, 56.25), ("ENE", 56.25, 78.75), ("E", 78.75, 101.25),
            ("ESE", 101.25, 123.75), ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75),
            ("S", 168.75, 191.25), ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25),
            ("WSW", 236.25, 258.75), ("W", 258.75, 281.25), ("WNW", 281.25, 303.75),
            ("NW", 303.75, 326.25), ("NNW", 326.25, 348.75)
        ]

        compass_direction = next((point for point, start, end in compass_points if start <= wind_deg < end), "N")
        if compass_direction in le.classes_:
            compass_direction_encoded = le.transform([compass_direction])[0]
        else:
            compass_direction_encoded = le.transform([le.classes_[0]])[0]

        current_data = {
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed': current_weather['wind_Gust_Speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['current_temp']
        }

        current_df = pd.DataFrame([current_data])
        rain_prediction = rain_model.predict(current_df)[0]

        x_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        x_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
        x_pre, y_pre = prepare_regression_data(historical_data, 'Pressure')

        temp_model = train_regression_model(x_temp, y_temp)
        hum_model = train_regression_model(x_hum, y_hum)
        pre_model = train_regression_model(x_pre, y_pre)

        future_temp = predict_future(temp_model, current_weather['temp_min'])
        future_humidity = predict_future(hum_model, current_weather['humidity'])
        future_pressure = predict_future(pre_model, current_weather['pressure'])

        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)
        next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
        future_times = [(next_hour + timedelta(hours=i)).strftime('%H:00') for i in range(24)]

        # Convert zip objects to lists for JSON serialization
        future_temp_list = list(zip(future_times, future_temp))
        future_humidity_list = list(zip(future_times, future_humidity))
        future_pressure_list = list(zip(future_times, future_pressure))

        # Update the cache
        weather_data = {
            "city": city,
            "country": current_weather['country'],
            "temp": current_weather['current_temp'],
            "feels_like": current_weather['feels_like'],
            "temp_min": current_weather['temp_min'],
            "temp_max": current_weather['temp_max'],
            "humidity": current_weather['humidity'],
            "description": current_weather['description'],
            "rain": 'Yes' if rain_prediction == 1 else 'No',
            "future_temp": future_temp_list,
            "future_humidity": future_humidity_list,
            "future_pressure": future_pressure_list
        }
        
        # Store the update time in the desired timezone
        last_update_times[city] = now.strftime("%Y-%m-%d %H:%M:%S")
        weather_data["last_updated"] = last_update_times[city]
        
        # Update cache
        weather_cache[city] = weather_data
        
        # Broadcast to all connected clients for this city
        await manager.broadcast_to_city(city, json.dumps(weather_data))
        
        print(f"Updated weather data for {city} at {last_update_times[city]}")
        return weather_data
    except Exception as e:
        print(f"Error updating weather for {city}: {str(e)}")
        return None

def schedule_weather_updates(city: str):
    """Schedule regular updates for a city (every 30 minutes)"""
    job_id = f"update_{city}"
    
    # Check if job already exists
    if scheduler.get_job(job_id):
        return
    
    # Schedule job to run every 30 minutes
    scheduler.add_job(
        update_weather_data,
        'interval',
        minutes=30,
        args=[city],
        id=job_id,
        replace_existing=True
    )
    print(f"Scheduled updates for {city} every 30 minutes")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/weather")
async def weather_report_json(request: Request, payload: CityRequest):
    city = payload.city
    
    # Check if we have cached data that's recent (within 30 minutes)
    if city in weather_cache and city in last_update_times:
        # Parse the stored timestamp
        timezone = pytz.timezone('Asia/Kolkata')
        last_update = datetime.strptime(last_update_times[city], "%Y-%m-%d %H:%M:%S")
        last_update = timezone.localize(last_update)
        now = datetime.now(timezone)
        
        # If data is recent, return cached data
        if (now - last_update).total_seconds() < 1800:  # 30 minutes in seconds
            return JSONResponse(content=weather_cache[city])
    
    # Otherwise, fetch fresh data
    weather_data = await update_weather_data(city)
    
    # Schedule regular updates for this city
    schedule_weather_updates(city)
    
    return JSONResponse(content=weather_data)

@app.websocket("/ws/{city}")
async def websocket_endpoint(websocket: WebSocket, city: str):
    await manager.connect(websocket, city)
    
    # Check if we need to schedule updates for this city
    schedule_weather_updates(city)
    
    # Send initial data if available
    if city in weather_cache:
        await websocket.send_text(json.dumps(weather_cache[city]))
    
    try:
        while True:
            # Keep the connection open but don't actually process any incoming messages
            # We're just using the WebSocket for server -> client updates
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, city)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)