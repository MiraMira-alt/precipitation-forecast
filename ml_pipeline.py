"""
ml_pipeline.py
Повний ML-пайплайн: навчання 4 моделей, оцінка, відбір найкращої, прогноз.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from feature_engineering import get_feature_columns


def train_models(df: pd.DataFrame):
    """
    Навчає 4 моделі класифікації, оцінює кожну, повертає найкращу.
    
    Returns:
        models (dict): всі навчені моделі
        best_model: найкраща модель (за F1)
        metrics (dict): метрики кожної моделі
        feature_cols (list): список ознак
    """
    # ── Підготовка ────────────────────────────────────────────────────────────
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Балансування класів через SMOTE
    try:
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    except Exception:
        X_train_bal, y_train_bal = X_train, y_train

    # ── Моделі ───────────────────────────────────────────────────────────────
    model_definitions = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=5, subsample=0.8, random_state=42,
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=6, subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
                random_state=42, eval_metric="logloss", verbosity=0,
            )),
        ]),
    }

    models = {}
    metrics = {}

    for name, pipeline in model_definitions.items():
        pipeline.fit(X_train_bal, y_train_bal)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        m = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, y_prob),
            "roc_fpr":   fpr.tolist(),
            "roc_tpr":   tpr.tolist(),
            "confusion_matrix": cm,
        }

        # Важливість ознак
        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            fi = dict(zip(feature_cols, clf.feature_importances_))
            m["feature_importance"] = fi
        elif hasattr(clf, "coef_"):
            fi = dict(zip(feature_cols, np.abs(clf.coef_[0])))
            m["feature_importance"] = fi

        models[name] = pipeline
        metrics[name] = m

    # ── Найкраща модель за F1 ─────────────────────────────────────────────────
    best_name = max(metrics, key=lambda n: metrics[n]["f1"])
    best_model = models[best_name]

    return models, best_model, metrics, feature_cols


def predict_forecast(forecast_df: pd.DataFrame, model, feature_cols: list) -> list:
    """
    Робить прогноз для кожного дня прогнозного датафрейму.
    
    Returns:
        list of dicts: [{date, prediction, probability}, ...]
    """
    # Знаходимо спільні колонки
    available = [c for c in feature_cols if c in forecast_df.columns]
    missing = [c for c in feature_cols if c not in forecast_df.columns]

    X = forecast_df[available].copy()
    # Якщо деяких колонок немає — заповнюємо нулями
    for col in missing:
        X[col] = 0.0
    X = X[feature_cols].values

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = []
    for i, idx in enumerate(forecast_df.index):
        results.append({
            "date": str(idx.date()) if hasattr(idx, 'date') else str(idx)[:10],
            "prediction": int(predictions[i]),
            "probability": float(probabilities[i]),
        })
    return results
