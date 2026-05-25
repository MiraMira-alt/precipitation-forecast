# Клітинка — перезаписуємо data_loader.py
code = '''import requests
import pandas as pd

DAILY_VARIABLES = [
    "precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "sunshine_duration",
    "daylight_duration",
    "snowfall_sum",
]

BASE_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
BASE_FORECAST   = "https://api.open-meteo.com/v1/forecast"

def _parse_response(data):
    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        raise ValueError("API не повернув daily/time")
    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df

def fetch_historical_data(lat, lon, start, end):
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "daily": ",".join(DAILY_VARIABLES), "timezone": "Europe/Kyiv"}
    r = requests.get(BASE_HISTORICAL, params=params, timeout=60)
    r.raise_for_status()
    df = _parse_response(r.json())
    df["target"] = ((df["precipitation_sum"].fillna(0) > 0) | (df["snowfall_sum"].fillna(0) > 0)).astype(int)
    return df

def fetch_forecast_data(lat, lon, days=7):
    params = {"latitude": lat, "longitude": lon, "daily": ",".join(DAILY_VARIABLES),
              "forecast_days": min(days, 16), "timezone": "Europe/Kyiv"}
    r = requests.get(BASE_FORECAST, params=params, timeout=30)
    r.raise_for_status()
    return _parse_response(r.json())

def geocode_city(city_name):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": city_name, "count": 1, "language": "uk"}, timeout=10)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        return None
    res = results[0]
    return {"lat": res["latitude"], "lon": res["longitude"], "name": res.get("name", city_name)}
'''
with open("data_loader.py", "w") as f:
    f.write(code)
print("OK")
