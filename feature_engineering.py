"""
feature_engineering.py
Підготовка та відбір ознак для ML-моделі прогнозування опадів.
"""
import pandas as pd
import numpy as np


# Ознаки що НЕ є data leakage (не містять інформацію про поточний день опадів)
BASE_FEATURE_COLS = [
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
]


def prepare_features(df: pd.DataFrame, is_forecast: bool = False) -> pd.DataFrame:
    """
    Підготовка ознак для навчання або прогнозу.
    Додає лагові ознаки (попередні дні) та сезонні компоненти.
    Прибирає рядки з пропущеними значеннями.
    """
    df = df.copy()

    # ── 1. Сезонні ознаки (детерміновані — не є leakage) ─────────────────────
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    # Синусоїдальне кодування сезонності
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── 2. Лагові ознаки (попередній день — не є leakage) ────────────────────
    lag_cols = [
        "temperature_2m_mean",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "sunshine_duration",
    ]
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag2"] = df[col].shift(2)

    # ── 3. Ковзні середні (3 дні назад) ─────────────────────────────────────
    for col in ["temperature_2m_mean", "wind_speed_10m_max", "shortwave_radiation_sum"]:
        if col in df.columns:
            df[f"{col}_roll3"] = df[col].shift(1).rolling(3).mean()

    # ── 4. Різниця температур день/ніч ───────────────────────────────────────
    if "temperature_2m_max" in df.columns and "temperature_2m_min" in df.columns:
        df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

    # ── 5. Попередній день мав опади? (лаг target) ───────────────────────────
    if not is_forecast and "target" in df.columns:
        df["prev_day_rain"] = df["target"].shift(1).fillna(0)

    if is_forecast:
        # Для прогнозу ставимо 0 (невідомо чи був дощ попередній день)
        df["prev_day_rain"] = 0

    # ── 6. Видаляємо рядки з пропусками (через лаги) ─────────────────────────
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Повертає список ознак для навчання (без target та сирих precipitation колонок).
    """
    exclude = {
        "target",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "precipitation_hours",  # ← це теж може бути leakage для поточного дня
        "date",
    }
    return [c for c in df.columns if c not in exclude]
