# Прогноз опадів — ML-сервіс на Open-Meteo

Streamlit-застосунок для прогнозування опадів на основі реальних метеорологічних даних.

---

## Призначення проєкту

Міні-сервіс прогнозування погоди, який:
- Завантажує реальні щоденні метеодані через Open-Meteo API (Historical + Forecast)
- Навчає 4 моделі класифікації для передбачення опадів (є/немає)
- Виводить прогноз на 7 днів з ймовірністю для кожного дня
- Підтримує будь-яке місто (6 передустановлених + введення координат вручну)

---

## Які дані використовуються

Джерело: [Open-Meteo](https://open-meteo.com) — безкоштовний погодний API

Архівні дані: за замовчуванням 2009–2024 (налаштовується від 1940 до поточного року)

Цільова змінна:
- `0` — опадів немає (`precipitation_sum == 0` та `snowfall_sum == 0`)
- `1` — опади є (`precipitation_sum > 0` або `snowfall_sum > 0`)

`precipitation_sum`, `rain_sum`, `snowfall_sum` не використовуються як ознаки — це data leakage.

Ознаки для моделі:

| Ознака | Опис |
|--------|------|
| `temperature_2m_mean/max/min` | Середня/макс/мін температура (°C) |
| `apparent_temperature_mean` | Відчутна температура |
| `wind_speed_10m_max` | Максимальна швидкість вітру |
| `wind_gusts_10m_max` | Пориви вітру |
| `shortwave_radiation_sum` | Сума сонячної радіації |
| `et0_fao_evapotranspiration` | Випаровуваність (ET0) |
| `sunshine_duration` | Тривалість сонячного сяйва |
| `daylight_duration` | Тривалість світлового дня |
| `sin_doy`, `cos_doy` | Синусоїдальне кодування дня року |
| `sin_month`, `cos_month` | Синусоїдальне кодування місяця |
| `*_lag1`, `*_lag2` | Лагові ознаки (1–2 дні назад) |
| `*_roll3` | Ковзне середнє за 3 дні |
| `temp_range` | Різниця макс/мін температур |
| `prev_day_rain` | Чи були опади попереднього дня |

---

## Моделі класифікації

1. Logistic Regression — базова лінійна модель
2. Random Forest — ансамбль дерев рішень
3. Gradient Boosting — градієнтний бустинг (sklearn)
4. XGBoost — оптимізований градієнтний бустинг

Балансування класів: SMOTE (оверсемплінг меншинного класу)

Вибір фінальної моделі: за максимальним F1-score на тестовій вибірці

Метрики оцінки: Accuracy, Precision, Recall, F1, ROC-AUC, матриця помилок

---

## Структура проєкту

```
precipitation_forecast/
│
├── app.py                  — головний Streamlit-застосунок
├── data_loader.py          — завантаження даних з Open-Meteo API
├── feature_engineering.py  — підготовка ознак, лаги, сезонність
├── ml_pipeline.py          — навчання моделей, оцінка, прогноз
├── images_b64.py           — зображення для інтерфейсу (base64)
├── requirements.txt        — залежності Python
├── README.md               — цей файл
└── weather_daily.csv       — збережені дані (генерується автоматично)
```

Послідовність дій:

```
1. app.py → data_loader.py         — завантаження архівних даних
2. app.py → feature_engineering.py — підготовка ознак + лаги
3. app.py → ml_pipeline.py         — навчання 4 моделей, вибір найкращої
4. app.py → data_loader.py         — завантаження прогнозних даних (Forecast API)
5. app.py → feature_engineering.py — підготовка ознак для прогнозу
6. app.py → ml_pipeline.py         — прогноз на 7 днів
```

---

## Запуск у Google Colab

```python
# Клітинка 1
import os
os.chdir('/content')
!rm -rf precipitation-forecast
!git clone https://github.com/MiraMira-alt/precipitation-forecast.git
%cd precipitation-forecast

# Клітинка 2
!pip install streamlit pandas numpy requests scikit-learn xgboost imbalanced-learn plotly joblib pyngrok pillow

# Клітинка 3
import os, subprocess, threading, time
from pyngrok import ngrok

ngrok.set_auth_token("ВАШ_ТОКЕН")

threading.Thread(
    target=lambda: subprocess.run(["streamlit", "run", "app.py", "--server.port=8501"]),
    daemon=True
).start()

time.sleep(5)
print(ngrok.connect(8501))
```

---

## Підтримувані міста

Київ, Львів, Харків, Одеса, Дніпро, Запоріжжя, а також будь-яке місто через введення координат вручну.
