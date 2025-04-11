from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime, timedelta
import pytz
from pydantic import BaseModel
from weather import get_current_weather, read_historical_data, prepare_data, train_rain_model, prepare_regression_data, train_regression_model, predict_future

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


class CityRequest(BaseModel):
    city: str

@app.post("/weather")
async def weather_report_json(request: Request, payload: CityRequest):
    city = payload.city
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

    temp_model = train_regression_model(x_temp, y_temp)
    hum_model = train_regression_model(x_hum, y_hum)

    future_temp = predict_future(temp_model, current_weather['temp_min'])
    future_humidity = predict_future(hum_model, current_weather['humidity'])

    timezone = pytz.timezone('Asia/Kolkata')
    now = datetime.now(timezone)
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    future_times = [(next_hour + timedelta(hours=i)).strftime('%H:00') for i in range(5)]

    # Convert zip objects to lists for JSON serialization
    future_temp_list = list(zip(future_times, future_temp))
    future_humidity_list = list(zip(future_times, future_humidity))

    return JSONResponse(content={
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
        "future_humidity": future_humidity_list
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)