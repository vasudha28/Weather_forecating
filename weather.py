import requests 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz

API_kEY = '897d31673e6edcf0e06b286dc9992368'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_kEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'city' : data['name'],
        'current_temp': round(data['main']['temp'] ),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'wind_Gust_Speed': data['wind']['speed'],
    }


# Read Historical data

def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    print(df.head(5))
    return df

def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    #define feature variables and target variable
    x = data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']]
    y = data['RainTomorrow'] #targetvariable

    return x,y,le

def train_rain_model(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("mean squared error for rain model")

    print(mean_squared_error(y_test,y_pred))

    return model    


def prepare_regression_data(data,feature):
    x,y = [],[]

    for i in range(len(data)-1):
        x.append(data[feature].iloc[i])

        y.append(data[feature].iloc[i+1])

    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    return x,y

def train_regression_model(x,y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x,y)
    return model


def predict_future(model,current_value):
    predictions = [current_value]
    for i in range(24):  
        next_value = model.predict(np.array([[predictions[-1]]]))

        predictions.append(next_value[0])

    return predictions[1:]  