"""
data_loader.py
Завантаження даних з Open-Meteo Historical та Forecast API.
"""
import requests
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


def _parse_response(data: dict) -> pd.DataFrame:
    """Перетворює відповідь API у DataFrame."""
    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        raise ValueError("Відповідь API не містить поля 'daily' або 'time'")
    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def fetch_historical_data(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Завантажує щоденні архівні дані з Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Europe/Kyiv",
    }
    response = requests.get(BASE_HISTORICAL, params=params, timeout=60)
    response.raise_for_status()
    df = _parse_response(response.json())

    df["target"] = (
        (df["precipitation_sum"].fillna(0) > 0) |
        (df["snowfall_sum"].fillna(0) > 0)
    ).astype(int)

    return df


def fetch_forecast_data(lat: float, lon: float, days: int = 7) -> pd.DataFrame:
    """Завантажує прогнозні дані з Open-Meteo Forecast API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARIABLES),
        "forecast_days": min(days, 16),
        "timezone": "Europe/Kyiv",
    }
    response = requests.get(BASE_FORECAST, params=params, timeout=30)
    response.raise_for_status()
    return _parse_response(response.json())


def geocode_city(city_name: str):
    """Геокодування назви міста через Open-Meteo Geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = requests.get(url, params={"name": city_name, "count": 1, "language": "uk"}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return None
    r = results[0]
    return {"lat": r["latitude"], "lon": r["longitude"], "name": r.get("name", city_name)}
