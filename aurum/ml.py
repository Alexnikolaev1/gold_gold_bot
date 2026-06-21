import pickle
from dataclasses import dataclass
from pathlib import Path

import logging
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from aurum.config import (
    BUY_MODEL_PATH,
    COST_POINTS,
    FEATURE_COLS,
    META_PATH,
    MODEL_DIR,
    SCALER_PATH,
    SELL_MODEL_PATH,
    TARGET_HORIZON_BARS,
)
from aurum.features import calculate_indicators

logger = logging.getLogger(__name__)


@dataclass
class ModelMeta:
    buy_accuracy: float
    sell_accuracy: float
    buy_precision: float
    sell_precision: float
    samples: int
    horizon_bars: int


def models_exist() -> bool:
    return all(p.exists() for p in (BUY_MODEL_PATH, SELL_MODEL_PATH, SCALER_PATH))


def create_target_buy(df: pd.DataFrame, horizon: int = TARGET_HORIZON_BARS, cost: float = COST_POINTS) -> pd.Series:
    future_close = df["Close"].shift(-horizon)
    return (future_close > (df["Close"] + cost)).astype(int)


def create_target_sell(df: pd.DataFrame, horizon: int = TARGET_HORIZON_BARS, cost: float = COST_POINTS) -> pd.Series:
    future_close = df["Close"].shift(-horizon)
    return (future_close < (df["Close"] - cost)).astype(int)


def _build_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )


def _holdout_metrics(model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    split = int(len(X) * 0.8)
    if split < 50 or len(X) - split < 20:
        return 0.0, 0.0
    preds = model.predict(X[split:])
    probs = model.predict_proba(X[split:])[:, 1]
    acc = float((preds == y[split:]).mean())
    pos_mask = probs >= 0.5
    prec = float((y[split:][pos_mask] == 1).mean()) if pos_mask.sum() > 0 else 0.0
    return acc, prec


def train_models(df: pd.DataFrame) -> ModelMeta:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df_feat = calculate_indicators(df)
    X = df_feat[FEATURE_COLS].iloc[:-TARGET_HORIZON_BARS]
    y_buy = create_target_buy(df_feat).iloc[:-TARGET_HORIZON_BARS]
    y_sell = create_target_sell(df_feat).iloc[:-TARGET_HORIZON_BARS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    buy_base = _build_classifier()
    sell_base = _build_classifier()

    buy_model = CalibratedClassifierCV(buy_base, method="isotonic", cv=3)
    sell_model = CalibratedClassifierCV(sell_base, method="isotonic", cv=3)

    buy_model.fit(X_scaled, y_buy)
    sell_model.fit(X_scaled, y_sell)

    buy_acc, buy_prec = _holdout_metrics(buy_model, X_scaled, y_buy.to_numpy())
    sell_acc, sell_prec = _holdout_metrics(sell_model, X_scaled, y_sell.to_numpy())

    meta = ModelMeta(
        buy_accuracy=buy_acc,
        sell_accuracy=sell_acc,
        buy_precision=buy_prec,
        sell_precision=sell_prec,
        samples=len(X),
        horizon_bars=TARGET_HORIZON_BARS,
    )

    with open(BUY_MODEL_PATH, "wb") as f:
        pickle.dump(buy_model, f)
    with open(SELL_MODEL_PATH, "wb") as f:
        pickle.dump(sell_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    clear_model_cache()
    return meta


def ensure_models() -> ModelMeta | None:
    """Train ML models if missing (Railway / fresh deploy / telegram worker)."""
    if models_exist():
        return load_meta()
    from aurum.data import fetch_training_data

    logger.info("ML models missing — fetching history and training...")
    df = fetch_training_data()
    if df.empty:
        logger.error("Cannot train ML: no market data from Yahoo Finance")
        return None
    meta = train_models(df)
    logger.info(
        "ML trained on %d bars | buy_prec=%.1f%% sell_prec=%.1f%%",
        meta.samples,
        meta.buy_precision * 100,
        meta.sell_precision * 100,
    )
    return meta


def load_meta() -> ModelMeta | None:
    if not META_PATH.exists():
        return None
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def predict_confidence(row: pd.Series) -> tuple[float, float]:
    if not models_exist():
        return 0.5, 0.5

    buy_model, sell_model, scaler = _load_artifacts()
    feat_vector = pd.DataFrame([row[FEATURE_COLS]])
    scaled = scaler.transform(feat_vector)

    p_buy = float(buy_model.predict_proba(scaled)[0][1])
    p_sell = float(sell_model.predict_proba(scaled)[0][1])
    return p_buy, p_sell


_artifact_cache: dict | None = None


def clear_model_cache() -> None:
    global _artifact_cache
    _artifact_cache = None


def _load_artifacts():
    global _artifact_cache
    if _artifact_cache is None:
        with open(BUY_MODEL_PATH, "rb") as f:
            buy_model = pickle.load(f)
        with open(SELL_MODEL_PATH, "rb") as f:
            sell_model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        _artifact_cache = {"buy": buy_model, "sell": sell_model, "scaler": scaler}
    return _artifact_cache["buy"], _artifact_cache["sell"], _artifact_cache["scaler"]


def predict_confidence_batch(df_feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Batch inference for backtesting — loads models once."""
    if not models_exist():
        n = len(df_feat)
        return np.full(n, 0.5), np.full(n, 0.5)

    buy_model, sell_model, scaler = _load_artifacts()
    X = scaler.transform(df_feat[FEATURE_COLS])
    p_buy = buy_model.predict_proba(X)[:, 1]
    p_sell = sell_model.predict_proba(X)[:, 1]
    return p_buy, p_sell
