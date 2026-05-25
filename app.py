import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_loader import fetch_historical_data, fetch_forecast_data
from ml_pipeline import train_models, predict_forecast
from feature_engineering import prepare_features

st.set_page_config(
    page_title="Прогноз опадів",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

:root {
    --blue-acc: #2d7dd2;
    --cyan:     #00d4ff;
    --rain:     #4fc3f7;
    --dry:      #ffd54f;
    --text:     #e8f4fd;
    --muted:    #8aabcc;
    --card-bg:  rgba(26,48,80,0.7);
    --border:   rgba(45,125,210,0.3);
}
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a1628 0%, #0f1b2d 50%, #0d2240 100%) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
h1 {
    font-family: 'Unbounded', sans-serif !important;
    font-size: 2.2rem !important;
    background: linear-gradient(90deg, var(--cyan), var(--blue-acc));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.block-title {
    font-family: 'Unbounded', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--cyan);
    border-left: 3px solid var(--cyan);
    padding-left: 10px;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}
label { color: #c8dff0 !important; font-size: 0.88rem !important; }
[data-testid="stSelectbox"] > div > div {
    background: rgba(26,48,80,0.9) !important;
    border: 1px solid rgba(45,125,210,0.5) !important;
    color: #e8f4fd !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] * { color: #e8f4fd !important; }
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--blue-acc), #1a5fa8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Unbounded', sans-serif !important;
    font-size: 0.85rem !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(45,125,210,0.4) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: var(--card-bg) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: 'Unbounded', sans-serif !important;
    font-size: 0.7rem !important;
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
hr { border-color: var(--border) !important; }
.forecast-card {
    background: var(--card-bg);
    border-radius: 14px;
    padding: 1.2rem 0.8rem;
    border: 1px solid var(--border);
    text-align: center;
}
.rain-badge {
    background: rgba(21,101,192,0.4);
    border: 1px solid var(--rain);
    color: var(--rain);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    display: inline-block;
}
.dry-badge {
    background: rgba(74,56,0,0.4);
    border: 1px solid var(--dry);
    color: var(--dry);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    display: inline-block;
}
.prob-bar-container {
    background: rgba(255,255,255,0.1);
    border-radius: 20px;
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 20px; }
.info-box {
    background: rgba(45,125,210,0.15);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: var(--rain);
    font-size: 0.88rem;
    margin-bottom: 1rem;
}
.success-box {
    background: rgba(0,150,100,0.15);
    border: 1px solid rgba(0,200,130,0.4);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: #a0ffd6;
    font-size: 0.88rem;
    margin-bottom: 0.5rem;
}
.warn-box {
    background: rgba(255,180,0,0.12);
    border: 1px solid rgba(255,180,0,0.35);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #ffe082;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# -- Session state --
for key in ['ready','df','best_model','metrics','feature_cols','results','forecast_raw']:
    if key not in st.session_state:
        st.session_state[key] = None
if 'city_coords' not in st.session_state:
    st.session_state.city_coords = {'lat': 50.4501, 'lon': 30.5234, 'name': 'Київ'}

TRAIN_START = "2009-01-01"
TRAIN_END   = "2024-12-31"

PRESET_CITIES = {
    "Київ":              (50.4501, 30.5234),
    "Львів":             (49.8397, 24.0297),
    "Харків":            (49.9935, 36.2304),
    "Одеса":             (46.4825, 30.7233),
    "Дніпро":            (48.4647, 35.0462),
    "Запоріжжя":         (47.8388, 35.1396),
    "Інше (координати)": None,
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1.5rem 0;">
    <h1>Прогноз опадів</h1>
    <p style="color:#8aabcc; font-size:0.95rem; margin-top:0.3rem;">
        ML-сервіс на основі даних Open-Meteo &middot; Щоденний прогноз для будь-якого міста
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Вибір міста + кнопка ──────────────────────────────────────────────────────
col_city, col_info, col_btn = st.columns([2, 2, 1])

with col_city:
    city_choice = st.selectbox("Місто", list(PRESET_CITIES.keys()), index=0)
    if city_choice == "Інше (координати)":
        c1, c2 = st.columns(2)
        with c1: lat = st.number_input("Широта", value=50.45, format="%.4f")
        with c2: lon = st.number_input("Довгота", value=30.52, format="%.4f")
        name = st.text_input("Назва міста", value="Моє місто")
        st.session_state.city_coords = {'lat': lat, 'lon': lon, 'name': name}
    else:
        lat, lon = PRESET_CITIES[city_choice]
        st.session_state.city_coords = {'lat': lat, 'lon': lon, 'name': city_choice}

with col_info:
    st.markdown(f"""
    <div style="background:rgba(26,48,80,0.5); border:1px solid var(--border);
         border-radius:10px; padding:0.85rem 1.2rem; margin-top:1.8rem;">
        <span style="color:#8aabcc; font-size:0.78rem;">Модель навчена на даних:</span><br>
        <span style="color:#e8f4fd; font-weight:500;">2009 — 2024</span>
        <span style="color:#8aabcc; font-size:0.75rem; margin-left:8px;">(5 840 днів)</span>
    </div>
    """, unsafe_allow_html=True)

with col_btn:
    st.markdown("<div style='margin-top:1.8rem;'>", unsafe_allow_html=True)
    go_btn = st.button("Отримати прогноз")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Основна логіка при натисканні ─────────────────────────────────────────────
if go_btn:
    coords = st.session_state.city_coords
    progress = st.progress(0, text="Завантажую історичні дані...")

    try:
        # Крок 1 — дані
        df = fetch_historical_data(lat=coords['lat'], lon=coords['lon'],
                                   start=TRAIN_START, end=TRAIN_END)
        df = prepare_features(df)
        df.to_csv("weather_daily.csv", index=True)
        st.session_state.df = df
        progress.progress(33, text="Навчаю моделі...")

        # Крок 2 — моделі
        models, best_model, metrics, feature_cols = train_models(df)
        st.session_state.best_model  = best_model
        st.session_state.metrics     = metrics
        st.session_state.feature_cols = feature_cols
        joblib.dump(best_model,   "best_model.pkl")
        joblib.dump(feature_cols, "feature_cols.pkl")
        progress.progress(66, text="Отримую прогноз...")

        # Крок 3 — прогноз
        forecast_df = fetch_forecast_data(lat=coords['lat'], lon=coords['lon'], days=7)
        forecast_df = prepare_features(forecast_df, is_forecast=True)
        results = predict_forecast(forecast_df, best_model, feature_cols)
        st.session_state.results      = results
        st.session_state.forecast_raw = forecast_df
        st.session_state.ready        = True
        progress.progress(100, text="Готово!")

    except Exception as e:
        progress.empty()
        st.error(f"Помилка: {e}")
        import traceback; st.code(traceback.format_exc())

# ── Результати ────────────────────────────────────────────────────────────────
if st.session_state.ready:
    coords   = st.session_state.city_coords
    results  = st.session_state.results
    raw      = st.session_state.forecast_raw
    metrics  = st.session_state.metrics
    best_name = max(metrics, key=lambda m: metrics[m]['f1'])

    st.markdown("---")
    st.markdown(f'<div class="success-box">Модель навчена успішно &nbsp;|&nbsp; Найкраща: <b>{best_name}</b> &nbsp;|&nbsp; F1 = {metrics[best_name]["f1"]:.3f}</div>', unsafe_allow_html=True)

    # ── Прогноз — картки ─────────────────────────────────────────────────────
    st.markdown('<div class="block-title">Прогноз на 7 днів</div>', unsafe_allow_html=True)

    row_cols = st.columns(len(results))
    for col, day in zip(row_cols, results):
        prob     = day['probability']
        is_rain  = day['prediction'] == 1
        badge    = "rain-badge" if is_rain else "dry-badge"
        label    = "Опади" if is_rain else "Сухо"
        bar_col  = "#4fc3f7" if is_rain else "#ffd54f"
        pct      = int(prob * 100)
        date_short = day['date'][5:]  # MM-DD
        with col:
            st.markdown(f"""
            <div class="forecast-card">
                <div style="font-size:0.68rem; color:#8aabcc; margin-bottom:5px;">{date_short}</div>
                <div class="{badge}">{label}</div>
                <div style="margin-top:8px; font-size:1.25rem; font-weight:700;
                     color:{'#4fc3f7' if is_rain else '#ffd54f'};">{pct}%</div>
                <div style="font-size:0.62rem; color:#8aabcc;">ймовірність</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width:{pct}%; background:{bar_col};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Графік прогнозу ───────────────────────────────────────────────────────
    dates = [r['date'] for r in results]
    probs = [r['probability'] * 100 for r in results]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=dates, y=probs, name="Ймовірність опадів (%)",
        marker_color=['#4fc3f7' if r['prediction']==1 else '#ffd54f' for r in results],
        marker_opacity=0.85,
    ), secondary_y=False)
    if 'temperature_2m_mean' in raw.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=raw['temperature_2m_mean'].values[:len(results)],
            name="Температура (°C)",
            line=dict(color='#ff8a65', width=2.5),
            mode='lines+markers', marker=dict(size=7),
        ), secondary_y=True)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=10,r=10,t=10,b=10), height=280,
    )
    fig.update_yaxes(title_text="Ймовірність (%)", secondary_y=False, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(title_text="Температура (°C)", secondary_y=True)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    st.plotly_chart(fig, use_container_width=True)

    # ── Таблиця прогнозу ──────────────────────────────────────────────────────
    with st.expander("Детальна таблиця прогнозу"):
        rows = []
        for r in results:
            row = {
                'Дата': r['date'],
                'Прогноз': 'Очікуються опади' if r['prediction']==1 else 'Опадів не очікується',
                'Ймовірність': f"{r['probability']*100:.1f}%",
            }
            if r['date'] in [str(i)[:10] for i in raw.index]:
                idx = pd.Timestamp(r['date'])
                if idx in raw.index:
                    row['Температура (°C)'] = f"{raw.loc[idx, 'temperature_2m_mean']:.1f}" if 'temperature_2m_mean' in raw.columns else '—'
                    row['Вітер (км/год)']   = f"{raw.loc[idx, 'wind_speed_10m_max']:.1f}" if 'wind_speed_10m_max' in raw.columns else '—'
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")

    # ── Деталі моделей (приховано) ────────────────────────────────────────────
    with st.expander("Деталі навчання моделей"):
        st.markdown('<div class="block-title">Порівняння моделей</div>', unsafe_allow_html=True)

        m_cols = st.columns(len(metrics))
        for i, (name, m) in enumerate(metrics.items()):
            is_best = (name == best_name)
            border  = "#00d4ff" if is_best else "rgba(45,125,210,0.3)"
            with m_cols[i]:
                st.markdown(f"""
                <div style="background:rgba(26,48,80,0.7); border:1.5px solid {border};
                     border-radius:14px; padding:1rem; text-align:center; margin-bottom:1rem;">
                    <div style="font-family:'Unbounded',sans-serif; font-size:0.68rem;
                         color:{'#00d4ff' if is_best else '#8aabcc'}; margin-bottom:0.5rem;">
                         {'* ' if is_best else ''}{name}</div>
                    <div style="font-size:1.4rem; font-weight:700; color:#e8f4fd;">{m['f1']:.3f}</div>
                    <div style="font-size:0.68rem; color:#8aabcc; margin-bottom:0.5rem;">F1-score</div>
                    <div style="font-size:0.76rem; color:#c8dff0; line-height:1.8;">
                        Accuracy: {m['accuracy']:.3f}<br>
                        Precision: {m['precision']:.3f}<br>
                        Recall: {m['recall']:.3f}<br>
                        ROC-AUC: {m['roc_auc']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        tab_feat, tab_roc, tab_cm = st.tabs(["Важливість ознак", "ROC-криві", "Матриця помилок"])

        with tab_feat:
            if 'feature_importance' in metrics[best_name]:
                fi = metrics[best_name]['feature_importance']
                fi_s = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])
                fig_fi = go.Figure(go.Bar(
                    x=list(fi_s.values()), y=list(fi_s.keys()), orientation='h',
                    marker=dict(color=list(fi_s.values()),
                                colorscale=[[0,'#1a3050'],[0.5,'#2d7dd2'],[1,'#00d4ff']],
                                showscale=False),
                ))
                fig_fi.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'), margin=dict(l=10,r=10,t=10,b=10),
                    yaxis=dict(autorange='reversed'), height=380,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        with tab_roc:
            fig_roc = go.Figure()
            colors = ['#00d4ff','#4fc3f7','#ffd54f','#ef5350']
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
                legend=dict(bgcolor='rgba(0,0,0,0)'), margin=dict(l=10,r=10,t=10,b=10),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with tab_cm:
            cm = metrics[best_name].get('confusion_matrix', [[0,0],[0,0]])
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=['Сухо (прогноз)','Опади (прогноз)'],
                y=['Сухо (факт)','Опади (факт)'],
                colorscale=[[0,'#0f1b2d'],[1,'#2d7dd2']],
                text=cm, texttemplate='<b>%{text}</b>',
                textfont=dict(size=18, color='white'), showscale=False,
            ))
            fig_cm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), margin=dict(l=10,r=10,t=10,b=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8aabcc; font-size:0.75rem; padding:0.5rem 0 1rem 0;">
    Дані: <a href="https://open-meteo.com" target="_blank" style="color:#4fc3f7;">Open-Meteo API</a>
    &middot; ML: scikit-learn, XGBoost &middot; Streamlit
</div>
""", unsafe_allow_html=True)
