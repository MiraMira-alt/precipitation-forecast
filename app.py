import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from data_loader import fetch_historical_data, fetch_forecast_data, geocode_city
from ml_pipeline import train_models, predict_forecast
from feature_engineering import prepare_features

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="☁️ Прогноз опадів",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

:root {
    --blue-dark: #0f1b2d;
    --blue-mid:  #1a3050;
    --blue-acc:  #2d7dd2;
    --cyan:      #00d4ff;
    --rain:      #4fc3f7;
    --dry:       #ffd54f;
    --text:      #e8f4fd;
    --muted:     #8aabcc;
    --card-bg:   rgba(26,48,80,0.7);
    --border:    rgba(45,125,210,0.3);
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a1628 0%, #0f1b2d 50%, #0d2240 100%) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

h1, h2, h3 { font-family: 'Unbounded', sans-serif !important; }

h1 { 
    font-size: 2.2rem !important; 
    background: linear-gradient(90deg, var(--cyan), var(--blue-acc));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem !important;
}

h2 { font-size: 1.2rem !important; color: var(--cyan) !important; }
h3 { font-size: 1rem !important; color: var(--rain) !important; }

.block-title {
    font-family: 'Unbounded', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--cyan);
    border-left: 3px solid var(--cyan);
    padding-left: 10px;
    margin-bottom: 1rem;
}

.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    backdrop-filter: blur(10px);
    margin-bottom: 0.8rem;
}

.forecast-card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid var(--border);
    backdrop-filter: blur(10px);
    text-align: center;
    transition: transform 0.2s;
}

.forecast-card:hover { transform: translateY(-3px); }

.rain-badge {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    border: 1px solid var(--rain);
    color: var(--rain);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 500;
}

.dry-badge {
    background: linear-gradient(135deg, #4a3800, #332700);
    border: 1px solid var(--dry);
    color: var(--dry);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 500;
}

.prob-bar-container {
    background: rgba(255,255,255,0.1);
    border-radius: 20px;
    height: 8px;
    margin-top: 8px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 20px;
    transition: width 0.5s ease;
}

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--blue-acc), #1a5fa8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Unbounded', sans-serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(45,125,210,0.4) !important;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stDateInput"] > div > div > input,
[data-testid="stTextInput"] > div > div > input {
    background: rgba(26,48,80,0.8) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

[data-testid="stMetric"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

[data-testid="stMetricValue"] { color: var(--cyan) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--card-bg) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: 'Unbounded', sans-serif !important;
    font-size: 0.72rem !important;
}

.stTabs [aria-selected="true"] {
    background: var(--blue-acc) !important;
    color: white !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--blue-acc), var(--cyan)) !important;
}

hr { border-color: var(--border) !important; }

.success-box {
    background: rgba(0, 150, 100, 0.15);
    border: 1px solid rgba(0, 200, 130, 0.4);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #a0ffd6;
    font-size: 0.9rem;
}

.info-box {
    background: rgba(45,125,210,0.15);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: var(--rain);
    font-size: 0.9rem;
}

.warn-box {
    background: rgba(255, 180, 0, 0.12);
    border: 1px solid rgba(255, 180, 0, 0.35);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #ffe082;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if 'df' not in st.session_state:
    st.session_state.df = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'city_coords' not in st.session_state:
    st.session_state.city_coords = {'lat': 50.4501, 'lon': 30.5234, 'name': 'Київ'}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
    <div style="font-size:3rem; margin-bottom:0.5rem;">🌧️</div>
    <h1>Прогноз опадів</h1>
    <p style="color:#8aabcc; font-size:0.95rem; margin-top:0.3rem;">
        ML-сервіс на основі даних Open-Meteo · Щоденний прогноз для будь-якого міста
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 1: ЗАВАНТАЖЕННЯ ДАНИХ
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="block-title">📡 Блок 1 — Завантаження даних</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])

PRESET_CITIES = {
    "Київ": (50.4501, 30.5234),
    "Львів": (49.8397, 24.0297),
    "Харків": (49.9935, 36.2304),
    "Одеса": (46.4825, 30.7233),
    "Дніпро": (48.4647, 35.0462),
    "Запоріжжя": (47.8388, 35.1396),
    "Інше (ввести координати)": None,
}

with col1:
    city_choice = st.selectbox("🏙️ Місто", list(PRESET_CITIES.keys()), index=0)
    if city_choice == "Інше (ввести координати)":
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Широта", value=50.45, format="%.4f")
        with c2:
            lon = st.number_input("Довгота", value=30.52, format="%.4f")
        city_name_custom = st.text_input("Назва (для відображення)", value="Моє місто")
        st.session_state.city_coords = {'lat': lat, 'lon': lon, 'name': city_name_custom}
    else:
        lat, lon = PRESET_CITIES[city_choice]
        st.session_state.city_coords = {'lat': lat, 'lon': lon, 'name': city_choice}

with col2:
    start_date = st.date_input(
        "📅 Початок периоду",
        value=date(2009, 1, 1),
        min_value=date(2000, 1, 1),
        max_value=date.today() - timedelta(days=365),
    )

with col3:
    end_date = st.date_input(
        "📅 Кінець периоду",
        value=date(2024, 12, 31),
        min_value=date(2000, 1, 2),
        max_value=date.today() - timedelta(days=1),
    )

col_btn1, col_btn2 = st.columns([1, 3])
with col_btn1:
    load_btn = st.button("⬇️ Завантажити дані", use_container_width=True)

if load_btn:
    coords = st.session_state.city_coords
    with st.spinner(f"Завантажую дані для {coords['name']} з Open-Meteo..."):
        try:
            df = fetch_historical_data(
                lat=coords['lat'],
                lon=coords['lon'],
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
            )
            df = prepare_features(df)
            st.session_state.df = df
            df.to_csv("weather_daily.csv", index=False)
            st.markdown(f'<div class="success-box">✅ Завантажено <b>{len(df)}</b> записів для <b>{coords["name"]}</b> · Збережено у <code>weather_daily.csv</code></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Помилка завантаження: {e}")

if st.session_state.df is not None:
    df = st.session_state.df
    with st.expander("📊 Переглянути дані", expanded=False):
        tab1, tab2 = st.tabs(["Таблиця", "Розподіл класів"])
        with tab1:
            st.dataframe(df.tail(30), use_container_width=True, height=300)
        with tab2:
            rain_days = int(df['target'].sum())
            dry_days = int((df['target'] == 0).sum())
            fig_pie = go.Figure(go.Pie(
                labels=["Опади", "Без опадів"],
                values=[rain_days, dry_days],
                hole=0.55,
                marker=dict(colors=["#4fc3f7", "#ffd54f"], line=dict(color='#0f1b2d', width=2)),
                textinfo='label+percent',
                textfont=dict(color='white'),
            ))
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                margin=dict(t=20, b=20),
                annotations=[dict(text=f"<b>{len(df)}</b><br>днів", x=0.5, y=0.5,
                                  font_size=14, font_color='white', showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown(f'<div class="info-box">🌧️ Днів з опадами: <b>{rain_days}</b> ({rain_days/len(df)*100:.1f}%) &nbsp;&nbsp; ☀️ Сухих днів: <b>{dry_days}</b> ({dry_days/len(df)*100:.1f}%)</div>', unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 2: НАВЧАННЯ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="block-title">🤖 Блок 2 — Навчання моделей</div>', unsafe_allow_html=True)

if st.session_state.df is None:
    st.markdown('<div class="warn-box">⚠️ Спочатку завантажте дані у Блоці 1</div>', unsafe_allow_html=True)
else:
    train_btn = st.button("🚀 Навчити моделі", use_container_width=False)

    if train_btn:
        with st.spinner("Навчаю 4 моделі класифікації..."):
            try:
                models, best_model, metrics, feature_cols = train_models(st.session_state.df)
                st.session_state.models = models
                st.session_state.best_model = best_model
                st.session_state.metrics = metrics
                st.session_state.feature_cols = feature_cols
                joblib.dump(best_model, "best_model.pkl")
                joblib.dump(feature_cols, "feature_cols.pkl")
            except Exception as e:
                st.error(f"Помилка навчання: {e}")
                import traceback; st.code(traceback.format_exc())

    if st.session_state.metrics is not None:
        metrics = st.session_state.metrics
        best_name = max(metrics, key=lambda m: metrics[m]['f1'])

        st.markdown(f'<div class="success-box">🏆 Найкраща модель: <b>{best_name}</b></div>', unsafe_allow_html=True)
        st.markdown("")

        # Metrics table
        model_names = list(metrics.keys())
        m_cols = st.columns(len(model_names))
        for i, (name, m) in enumerate(metrics.items()):
            is_best = name == best_name
            with m_cols[i]:
                border_color = "#00d4ff" if is_best else "rgba(45,125,210,0.3)"
                crown = "👑 " if is_best else ""
                st.markdown(f"""
                <div style="background:rgba(26,48,80,0.7); border:1.5px solid {border_color};
                     border-radius:14px; padding:1rem; text-align:center;">
                    <div style="font-family:'Unbounded',sans-serif; font-size:0.7rem;
                         color:{'#00d4ff' if is_best else '#8aabcc'}; margin-bottom:0.5rem;">
                         {crown}{name}</div>
                    <div style="font-size:1.5rem; font-weight:700; color:#e8f4fd;">{m['f1']:.3f}</div>
                    <div style="font-size:0.7rem; color:#8aabcc;">F1-score</div>
                    <hr style="border-color:rgba(255,255,255,0.1); margin:0.5rem 0;">
                    <div style="font-size:0.78rem; color:#e8f4fd;">
                        Acc: {m['accuracy']:.3f}<br>
                        Prec: {m['precision']:.3f}<br>
                        Rec: {m['recall']:.3f}<br>
                        ROC-AUC: {m['roc_auc']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # Charts
        tab_feat, tab_roc, tab_cm = st.tabs(["Важливість ознак", "ROC-криві", "Матриця помилок"])

        with tab_feat:
            if 'feature_importance' in metrics[best_name]:
                fi = metrics[best_name]['feature_importance']
                fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])
                fig_fi = go.Figure(go.Bar(
                    x=list(fi_sorted.values()),
                    y=list(fi_sorted.keys()),
                    orientation='h',
                    marker=dict(
                        color=list(fi_sorted.values()),
                        colorscale=[[0, '#1a3050'], [0.5, '#2d7dd2'], [1, '#00d4ff']],
                        showscale=False,
                    ),
                ))
                fig_fi.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'), margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(autorange='reversed'),
                    height=400,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        with tab_roc:
            fig_roc = go.Figure()
            colors = ['#00d4ff', '#4fc3f7', '#ffd54f', '#ef5350']
            for i, (name, m) in enumerate(metrics.items()):
                if 'roc_fpr' in m:
                    fig_roc.add_trace(go.Scatter(
                        x=m['roc_fpr'], y=m['roc_tpr'],
                        name=f"{name} (AUC={m['roc_auc']:.3f})",
                        line=dict(color=colors[i % len(colors)], width=2),
                    ))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random',
                                          line=dict(color='gray', dash='dash', width=1)))
            fig_roc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with tab_cm:
            cm = metrics[best_name].get('confusion_matrix', [[0,0],[0,0]])
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=['Сухо (прогноз)', 'Опади (прогноз)'],
                y=['Сухо (факт)', 'Опади (факт)'],
                colorscale=[[0,'#0f1b2d'], [1,'#2d7dd2']],
                text=cm, texttemplate='<b>%{text}</b>',
                textfont=dict(size=18, color='white'),
                showscale=False,
            ))
            fig_cm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 3: ПРОГНОЗ
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="block-title">🔮 Блок 3 — Прогноз опадів</div>', unsafe_allow_html=True)

if st.session_state.best_model is None:
    st.markdown('<div class="warn-box">⚠️ Спочатку навчіть модель у Блоці 2</div>', unsafe_allow_html=True)
else:
    coords = st.session_state.city_coords
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        forecast_days = st.slider("📆 Кількість днів прогнозу", min_value=1, max_value=16, value=7)
        forecast_btn = st.button("🌦️ Отримати прогноз", use_container_width=True)

    if forecast_btn:
        with st.spinner(f"Отримую прогноз для {coords['name']}..."):
            try:
                forecast_df = fetch_forecast_data(
                    lat=coords['lat'],
                    lon=coords['lon'],
                    days=forecast_days,
                )
                forecast_df = prepare_features(forecast_df, is_forecast=True)
                results = predict_forecast(
                    forecast_df,
                    st.session_state.best_model,
                    st.session_state.feature_cols,
                )
                st.session_state.forecast_results = results
                st.session_state.forecast_df_raw = forecast_df
            except Exception as e:
                st.error(f"Помилка прогнозу: {e}")
                import traceback; st.code(traceback.format_exc())

    if 'forecast_results' in st.session_state and st.session_state.forecast_results is not None:
        results = st.session_state.forecast_results
        forecast_df_raw = st.session_state.forecast_df_raw

        st.markdown(f"### 📍 {coords['name']} · Прогноз на {len(results)} днів")

        # Forecast cards
        cols_per_row = min(len(results), 7)
        rows = [results[i:i+cols_per_row] for i in range(0, len(results), cols_per_row)]

        for row in rows:
            row_cols = st.columns(len(row))
            for i, (col, day) in enumerate(zip(row_cols, row)):
                with col:
                    prob = day['probability']
                    is_rain = day['prediction'] == 1
                    emoji = "🌧️" if is_rain else "☀️"
                    badge_class = "rain-badge" if is_rain else "dry-badge"
                    badge_text = "Опади" if is_rain else "Сухо"
                    bar_color = "#4fc3f7" if is_rain else "#ffd54f"
                    prob_pct = int(prob * 100)

                    st.markdown(f"""
                    <div class="forecast-card">
                        <div style="font-size:0.7rem; color:#8aabcc; margin-bottom:4px;">{day['date']}</div>
                        <div style="font-size:2rem; margin-bottom:6px;">{emoji}</div>
                        <div class="{badge_class}">{badge_text}</div>
                        <div style="margin-top:10px; font-size:1.2rem; font-weight:600; color:{'#4fc3f7' if is_rain else '#ffd54f'};">
                            {prob_pct}%
                        </div>
                        <div style="font-size:0.65rem; color:#8aabcc;">ймовірність опадів</div>
                        <div class="prob-bar-container">
                            <div class="prob-bar-fill" style="width:{prob_pct}%; background:{bar_color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("")

        # Summary chart
        if len(forecast_df_raw) > 0 and 'temperature_2m_mean' in forecast_df_raw.columns:
            dates = [r['date'] for r in results]
            probs = [r['probability'] * 100 for r in results]
            temps = forecast_df_raw['temperature_2m_mean'].values[:len(results)] if 'temperature_2m_mean' in forecast_df_raw.columns else [None]*len(results)

            fig_fc = make_subplots(specs=[[{"secondary_y": True}]])
            fig_fc.add_trace(go.Bar(
                x=dates, y=probs,
                name="Ймовірність опадів (%)",
                marker_color=['#4fc3f7' if r['prediction']==1 else '#ffd54f' for r in results],
                marker_opacity=0.8,
            ), secondary_y=False)

            if temps[0] is not None:
                fig_fc.add_trace(go.Scatter(
                    x=dates, y=temps,
                    name="Температура (°C)",
                    line=dict(color='#ff8a65', width=2.5),
                    mode='lines+markers',
                    marker=dict(size=7),
                ), secondary_y=True)

            fig_fc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=10, r=10, t=20, b=10),
                height=300,
            )
            fig_fc.update_yaxes(title_text="Ймовірність (%)", secondary_y=False, gridcolor='rgba(255,255,255,0.05)')
            fig_fc.update_yaxes(title_text="Температура (°C)", secondary_y=True)
            fig_fc.update_xaxes(gridcolor='rgba(255,255,255,0.05)')

            st.plotly_chart(fig_fc, use_container_width=True)

        # Detailed table
        with st.expander("📋 Детальна таблиця прогнозу"):
            table_data = []
            for r in results:
                row = {'Дата': r['date'], 'Прогноз': '🌧️ Опади' if r['prediction']==1 else '☀️ Сухо',
                       'Ймовірність': f"{r['probability']*100:.1f}%"}
                # add weather vars if available
                idx = forecast_df_raw.index[forecast_df_raw.index.astype(str) == r['date']].tolist()
                if len(idx) == 0 and len(forecast_df_raw) > 0:
                    # try by position
                    pass
                table_data.append(row)
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8aabcc; font-size:0.75rem; padding:1rem 0;">
    Дані: <a href="https://open-meteo.com" target="_blank" style="color:#4fc3f7;">Open-Meteo API</a> · 
    ML: scikit-learn, XGBoost · Візуалізація: Plotly · UI: Streamlit
</div>
""", unsafe_allow_html=True)
