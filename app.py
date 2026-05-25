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
from images_b64 import UMBRELLA_B64, SUN_B64

st.set_page_config(
    page_title="Прогноз опадів",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DAYS_UK = {0:"Пн", 1:"Вт", 2:"Ср", 3:"Чт", 4:"Пт", 5:"Сб", 6:"Нд"}

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
    background: rgba(26,48,80,0.9) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] { color: #a0c4e8 !important; font-family: 'Unbounded', sans-serif !important; font-size: 0.7rem !important; }
.stTabs [aria-selected="true"] { background: var(--blue-acc) !important; color: white !important; border-radius: 8px !important; }

/* Expander — примусово темний фон і світлий текст скрізь */
[data-testid="stExpander"] { background: #0d1f38 !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
[data-testid="stExpander"] * { color: #e8f4fd !important; background-color: transparent !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { background: #0d1f38 !important; }
[data-testid="stExpander"] summary span { color: #e8f4fd !important; }
[data-testid="stExpander"] th { color: #8aabcc !important; }
[data-testid="stExpander"] td { color: #e8f4fd !important; }

/* Radio горизонтально */
[data-testid="stRadio"] > div { flex-direction: row !important; flex-wrap: wrap; gap: 8px; }
[data-testid="stRadio"] > div > label {
    background: rgba(26,48,80,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    color: #e8f4fd !important;
    cursor: pointer;
}
hr { border-color: var(--border) !important; }
.forecast-card {
    background: var(--card-bg);
    border-radius: 14px;
    padding: 1rem 0.6rem;
    border: 1px solid var(--border);
    text-align: center;
}
.rain-badge { background: rgba(21,101,192,0.4); border: 1px solid var(--rain); color: var(--rain); border-radius: 20px; padding: 3px 10px; font-size: 0.75rem; display: inline-block; }
.dry-badge  { background: rgba(74,56,0,0.4);  border: 1px solid var(--dry);  color: var(--dry);  border-radius: 20px; padding: 3px 10px; font-size: 0.75rem; display: inline-block; }
.prob-bar-container { background: rgba(255,255,255,0.1); border-radius: 20px; height: 6px; margin-top: 8px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 20px; }
.success-box { background: rgba(0,150,100,0.15); border: 1px solid rgba(0,200,130,0.4); border-radius: 10px; padding: 0.7rem 1rem; color: #a0ffd6; font-size: 0.88rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

for key in ['ready','df','models','best_model','metrics','feature_cols','results','forecast_raw']:
    if key not in st.session_state:
        st.session_state[key] = None
if 'city_coords' not in st.session_state:
    st.session_state.city_coords = {'lat': 50.4501, 'lon': 30.5234, 'name': 'Київ'}

PRESET_CITIES = {
    "Київ":              (50.4501, 30.5234),
    "Львів":             (49.8397, 24.0297),
    "Харків":            (49.9935, 36.2304),
    "Одеса":             (46.4825, 30.7233),
    "Дніпро":            (48.4647, 35.0462),
    "Запоріжжя":         (47.8388, 35.1396),
    "Інше (координати)": None,
}

# Дефолтний період
DEFAULT_START = "2009-01-01"
DEFAULT_END   = "2024-12-31"

if "train_start" not in st.session_state:
    st.session_state.train_start = DEFAULT_START
if "train_end" not in st.session_state:
    st.session_state.train_end = DEFAULT_END
if "show_date_settings" not in st.session_state:
    st.session_state.show_date_settings = False

st.markdown("""
<div style="text-align:center; padding: 2rem 0 1.5rem 0;">
    <h1>Прогноз опадів</h1>
    <p style="color:#8aabcc; font-size:0.95rem; margin-top:0.3rem;">
        ML-сервіс на основі даних Open-Meteo &middot; Щоденний прогноз для будь-якого міста
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

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
    from datetime import datetime
    ts = st.session_state.train_start
    te = st.session_state.train_end
    days_count = (datetime.strptime(te, "%Y-%m-%d") - datetime.strptime(ts, "%Y-%m-%d")).days
    st.markdown(f"""
    <div style="background:rgba(26,48,80,0.5); border:1px solid var(--border);
         border-radius:10px; padding:0.85rem 1.2rem; margin-top:1.8rem;">
        <span style="color:#8aabcc; font-size:0.78rem;">Модель навчена на даних:</span><br>
        <span style="color:#e8f4fd; font-weight:500;">{ts[:4]} &mdash; {te[:4]}</span>
        <span style="color:#8aabcc; font-size:0.75rem; margin-left:8px;">({days_count:,} днів)</span>
    </div>
    """, unsafe_allow_html=True)

with col_btn:
    st.markdown("<div style='margin-top:1.8rem;'>", unsafe_allow_html=True)
    go_btn = st.button("Отримати прогноз")
    st.markdown("</div>", unsafe_allow_html=True)

# Кнопка розширених налаштувань
with st.expander("Налаштування даних для навчання"):
    st.markdown("""
    <div style="color:#8aabcc; font-size:0.82rem; margin-bottom:1rem;">
        Open-Meteo надає архівні дані з <b style='color:#e8f4fd'>1940</b> до приблизно 
        <b style='color:#e8f4fd'>5 днів тому</b>. 
        Чим більший діапазон — тим точніша модель, але довше завантаження.
    </div>
    """, unsafe_allow_html=True)

    from datetime import date, timedelta
    max_end = date.today() - timedelta(days=5)

    dc1, dc2, dc3 = st.columns([1, 1, 1])
    with dc1:
        new_start = st.date_input("Початок навчання",
            value=date(2009, 1, 1),
            min_value=date(1940, 1, 1),
            max_value=date(2020, 12, 31),
            key="date_start_picker")
    with dc2:
        new_end = st.date_input("Кінець навчання",
            value=date(2024, 12, 31),
            min_value=date(1941, 1, 1),
            max_value=max_end,
            key="date_end_picker")
    with dc3:
        st.markdown("<div style='margin-top:1.8rem;'>", unsafe_allow_html=True)
        if st.button("Застосувати", key="apply_dates"):
            if new_start < new_end:
                st.session_state.train_start = new_start.strftime("%Y-%m-%d")
                st.session_state.train_end   = new_end.strftime("%Y-%m-%d")
                st.success(f"Встановлено: {new_start} — {new_end}")
            else:
                st.error("Початок має бути раніше кінця")
        if st.button("Скинути до стандартних", key="reset_dates"):
            st.session_state.train_start = DEFAULT_START
            st.session_state.train_end   = DEFAULT_END
            st.success("Скинуто: 2009 — 2024")
        st.markdown("</div>", unsafe_allow_html=True)

    # Підказки
    st.markdown("""
    <div style="display:flex; gap:10px; margin-top:0.5rem; flex-wrap:wrap;">
        <span style="background:rgba(45,125,210,0.2); border:1px solid var(--border); border-radius:8px;
              padding:4px 12px; font-size:0.75rem; color:#8aabcc;">
            1940–2024 &mdash; максимум даних (ERA5)
        </span>
        <span style="background:rgba(45,125,210,0.2); border:1px solid var(--border); border-radius:8px;
              padding:4px 12px; font-size:0.75rem; color:#8aabcc;">
            2000–2024 &mdash; баланс точності і швидкості
        </span>
        <span style="background:rgba(45,125,210,0.2); border:1px solid var(--border); border-radius:8px;
              padding:4px 12px; font-size:0.75rem; color:#8aabcc;">
            2009–2024 &mdash; стандартне налаштування
        </span>
    </div>
    """, unsafe_allow_html=True)

if go_btn:
    coords = st.session_state.city_coords
    progress = st.progress(0, text="Завантажую історичні дані...")
    try:
        df = fetch_historical_data(lat=coords['lat'], lon=coords['lon'], start=st.session_state.train_start, end=st.session_state.train_end)
        df = prepare_features(df)
        df.to_csv("weather_daily.csv", index=True)
        st.session_state.df = df
        progress.progress(33, text="Навчаю моделі...")

        models, best_model, metrics, feature_cols = train_models(df)
        st.session_state.models       = models
        st.session_state.best_model   = best_model
        st.session_state.metrics      = metrics
        st.session_state.feature_cols = feature_cols
        joblib.dump(best_model,   "best_model.pkl")
        joblib.dump(feature_cols, "feature_cols.pkl")
        progress.progress(66, text="Отримую прогноз...")

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

if st.session_state.ready:
    coords    = st.session_state.city_coords
    results   = st.session_state.results
    raw       = st.session_state.forecast_raw
    metrics   = st.session_state.metrics
    best_name = max(metrics, key=lambda m: metrics[m]['f1'])

    st.markdown("---")
    st.markdown(
        f'<div class="success-box">Модель навчена успішно &nbsp;|&nbsp; '
        f'Найкраща: <b>{best_name}</b> &nbsp;|&nbsp; F1 = {metrics[best_name]["f1"]:.3f}</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="block-title">Прогноз на 7 днів</div>', unsafe_allow_html=True)

    row_cols = st.columns(len(results))
    for col, day in zip(row_cols, results):
        prob    = day['probability']
        is_rain = day['prediction'] == 1
        badge   = "rain-badge" if is_rain else "dry-badge"
        label   = "Опади" if is_rain else "Сухо"
        bar_col = "#4fc3f7" if is_rain else "#ffd54f"
        pct     = int(prob * 100)
        # День тижня
        dt      = pd.Timestamp(day['date'])
        dow     = DAYS_UK[dt.dayofweek]
        date_s  = day['date'][5:]
        # Картинка
        img_tag = (
            f'<img src="data:image/png;base64,{UMBRELLA_B64}" style="width:52px;height:52px;object-fit:contain;margin:4px 0;">'
            if is_rain else
            f'<img src="data:image/png;base64,{SUN_B64}" style="width:52px;height:52px;object-fit:contain;margin:4px 0;">'
        )
        with col:
            st.markdown(f"""
            <div class="forecast-card">
                <div style="font-size:0.8rem; font-weight:600; color:#c8dff0;">{dow}</div>
                <div style="font-size:0.65rem; color:#8aabcc; margin-bottom:6px;">{date_s}</div>
                {img_tag}
                <div style="margin:6px 0;"><div class="{badge}">{label}</div></div>
                <div style="font-size:1.2rem; font-weight:700; color:{'#4fc3f7' if is_rain else '#ffd54f'};">{pct}%</div>
                <div style="font-size:0.6rem; color:#8aabcc;">ймовірність</div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width:{pct}%; background:{bar_col};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Графік
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
        font=dict(color='#e8f4fd'),
        legend=dict(bgcolor='rgba(15,27,45,0.85)', font=dict(color='#e8f4fd', size=12),
                    bordercolor='rgba(45,125,210,0.3)', borderwidth=1),
        margin=dict(l=10,r=10,t=10,b=10), height=280,
    )
    fig.update_yaxes(title_text="Ймовірність (%)", secondary_y=False, gridcolor='rgba(255,255,255,0.05)', color='#e8f4fd')
    fig.update_yaxes(title_text="Температура (°C)", secondary_y=True, color='#e8f4fd')
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', color='#e8f4fd')
    st.plotly_chart(fig, use_container_width=True)

    # Детальна таблиця
    with st.expander("Детальна таблиця прогнозу"):
        rows = []
        for r in results:
            dt  = pd.Timestamp(r['date'])
            row = {
                'День': DAYS_UK[dt.dayofweek],
                'Дата': r['date'],
                'Прогноз': 'Очікуються опади' if r['prediction']==1 else 'Опадів не очікується',
                'Ймовірність': f"{r['probability']*100:.1f}%",
            }
            if dt in raw.index:
                row['Температура (°C)'] = f"{raw.loc[dt,'temperature_2m_mean']:.1f}" if 'temperature_2m_mean' in raw.columns else '—'
                row['Вітер (км/год)']   = f"{raw.loc[dt,'wind_speed_10m_max']:.1f}"  if 'wind_speed_10m_max'  in raw.columns else '—'
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")

    # Деталі моделей
    with st.expander("Деталі навчання моделей"):
        st.markdown('<div class="block-title">Порівняння моделей</div>', unsafe_allow_html=True)
        m_cols = st.columns(len(metrics))
        for i, (name, m) in enumerate(metrics.items()):
            is_best = (name == best_name)
            border  = "#00d4ff" if is_best else "rgba(45,125,210,0.3)"
            with m_cols[i]:
                st.markdown(f"""
                <div style="background:rgba(26,48,80,0.8); border:1.5px solid {border};
                     border-radius:14px; padding:1rem; text-align:center; margin-bottom:1rem;">
                    <div style="font-family:'Unbounded',sans-serif; font-size:0.68rem;
                         color:{'#00d4ff' if is_best else '#8aabcc'}; margin-bottom:0.5rem;">
                         {'* ' if is_best else ''}{name}</div>
                    <div style="font-size:1.4rem; font-weight:700; color:#e8f4fd;">{m['f1']:.3f}</div>
                    <div style="font-size:0.68rem; color:#8aabcc; margin-bottom:0.5rem;">F1-score</div>
                    <div style="font-size:0.76rem; color:#c8dff0; line-height:1.8;">
                        Accuracy: {m['accuracy']:.3f}<br>Precision: {m['precision']:.3f}<br>
                        Recall: {m['recall']:.3f}<br>ROC-AUC: {m['roc_auc']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="block-title">Детальний аналіз моделі</div>', unsafe_allow_html=True)
        selected_model = st.radio("Оберіть модель:", list(metrics.keys()),
                                   index=list(metrics.keys()).index(best_name), horizontal=True)
        sel_m = metrics[selected_model]

        tab_feat, tab_roc, tab_cm = st.tabs(["Важливість ознак", "ROC-крива", "Матриця помилок"])

        with tab_feat:
            if 'feature_importance' in sel_m:
                fi   = sel_m['feature_importance']
                fi_s = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])
                fig_fi = go.Figure(go.Bar(
                    x=list(fi_s.values()), y=list(fi_s.keys()), orientation='h',
                    marker=dict(color=list(fi_s.values()),
                                colorscale=[[0,'#1a3050'],[0.5,'#2d7dd2'],[1,'#00d4ff']], showscale=False),
                ))
                fig_fi.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e8f4fd'), margin=dict(l=10,r=10,t=10,b=10),
                    yaxis=dict(autorange='reversed', color='#e8f4fd'),
                    xaxis=dict(color='#e8f4fd'), height=380,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        with tab_roc:
            fig_roc = go.Figure()
            for i, (name, m) in enumerate(metrics.items()):
                if 'roc_fpr' in m and name != selected_model:
                    fig_roc.add_trace(go.Scatter(
                        x=m['roc_fpr'], y=m['roc_tpr'],
                        name=f"{name} (AUC={m['roc_auc']:.3f})",
                        line=dict(color='rgba(150,170,200,0.35)', width=1.5, dash='dot'),
                    ))
            if 'roc_fpr' in sel_m:
                fig_roc.add_trace(go.Scatter(
                    x=sel_m['roc_fpr'], y=sel_m['roc_tpr'],
                    name=f"{selected_model} (AUC={sel_m['roc_auc']:.3f})",
                    line=dict(color='#00d4ff', width=3),
                ))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random',
                                          line=dict(color='rgba(150,150,150,0.5)', dash='dash', width=1)))
            fig_roc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8f4fd'),
                xaxis=dict(title='False Positive Rate', color='#e8f4fd', gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(title='True Positive Rate', color='#e8f4fd', gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(bgcolor='rgba(15,27,45,0.85)', font=dict(color='#e8f4fd'),
                            bordercolor='rgba(45,125,210,0.3)', borderwidth=1),
                margin=dict(l=10,r=10,t=10,b=10),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with tab_cm:
            cm = sel_m.get('confusion_matrix', [[0,0],[0,0]])
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=['Сухо (прогноз)','Опади (прогноз)'], y=['Сухо (факт)','Опади (факт)'],
                colorscale=[[0,'#0f1b2d'],[1,'#2d7dd2']],
                text=cm, texttemplate='<b>%{text}</b>',
                textfont=dict(size=20, color='white'), showscale=False,
            ))
            fig_cm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8f4fd'),
                xaxis=dict(color='#e8f4fd'), yaxis=dict(color='#e8f4fd'),
                margin=dict(l=10,r=10,t=10,b=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8aabcc; font-size:0.75rem; padding:0.5rem 0 1rem 0;">
    Дані: <a href="https://open-meteo.com" target="_blank" style="color:#4fc3f7;">Open-Meteo API</a>
    &middot; ML: scikit-learn, XGBoost &middot; Streamlit
</div>
""", unsafe_allow_html=True)
