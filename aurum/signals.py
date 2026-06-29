from dataclasses import dataclass

import numpy as np
import pandas as pd

from aurum.config import CONFIDENCE_LOOKBACK, TOTAL_RULES
from aurum.thresholds import get_entry_thresholds
from aurum.ml import predict_confidence, predict_confidence_batch


def _technical_p_buy_sell(row: pd.Series) -> tuple[float, float]:
    """Composite technical score for crypto (no per-asset ML)."""
    buy = 0.35
    sell = 0.35
    if row["EMA_9"] > row["EMA_21"]:
        buy += 0.08
    else:
        sell += 0.08
    if row["EMA_50"] > row["EMA_200"]:
        buy += 0.08
    else:
        sell += 0.08
    if row["MACD_Diff"] > 0:
        buy += 0.1
    else:
        sell += 0.1
    if row["EMA_9_Slope"] > 0:
        buy += 0.06
    else:
        sell += 0.06
    if row["Close"] > row["POC"]:
        buy += 0.08
    else:
        sell += 0.08
    rsi = float(row["RSI"])
    if rsi < 40:
        buy += 0.1
    elif rsi > 60:
        sell += 0.1
    if row["BB_Squeeze"] > 0.02:
        if row["Close"] > row["EMA_9"]:
            buy += 0.05
        else:
            sell += 0.05
    total = buy + sell
    return buy / total, sell / total


def technical_scores_batch(df_feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(df_feat)
    pb = np.zeros(n)
    ps = np.zeros(n)
    for i in range(n):
        pb[i], ps[i] = _technical_p_buy_sell(df_feat.iloc[i])
    return pb, ps


@dataclass
class SignalResult:
    signal: str
    confidence: float
    p_buy: float
    p_sell: float
    conf_buy: float
    conf_sell: float
    rules_buy: int
    rules_sell: int
    rule_details: list[str]


def _count_rules(row: pd.Series, *, include_macro: bool = True) -> tuple[int, int, list[str]]:
    rules_buy = 0
    rules_sell = 0
    details: list[str] = []

    if row["EMA_9"] > row["EMA_21"] and row["EMA_50"] > row["EMA_200"]:
        rules_buy += 1
        details.append("✓ EMA тренд бычий")
    elif row["EMA_9"] < row["EMA_21"] and row["EMA_50"] < row["EMA_200"]:
        rules_sell += 1
        details.append("✓ EMA тренд медвежий")

    if row["MACD_Diff"] > 0 and row["EMA_9_Slope"] > 0:
        rules_buy += 1
        details.append("✓ MACD + наклон EMA9")
    elif row["MACD_Diff"] < 0 and row["EMA_9_Slope"] < 0:
        rules_sell += 1
        details.append("✓ MACD − наклон EMA9")

    if row["Close"] > row["POC"] and row["Close"] > row["VAL"]:
        rules_buy += 1
        details.append("✓ Цена выше POC/VAL")
    elif row["Close"] < row["POC"] and row["Close"] < row["VAH"]:
        rules_sell += 1
        details.append("✓ Цена ниже POC/VAH")

    if row["RSI"] < row["RSI_Oversold"]:
        rules_buy += 1
        details.append("✓ RSI перепродан (адаптивный)")
    elif row["RSI"] > row["RSI_Overbought"]:
        rules_sell += 1
        details.append("✓ RSI перекуплен (адаптивный)")

    if include_macro:
        if row["Corr_DXY_Z"] < -1.0:
            rules_buy += 1
            details.append("✓ DXY корреляция: слабый доллар → золото")
        elif row["Corr_DXY_Z"] > 1.0:
            rules_sell += 1
            details.append("✓ DXY корреляция: сильный доллар → золото")

    if row["BB_Squeeze"] > 0.02:
        if row["Close"] > row["EMA_9"]:
            rules_buy += 1
            details.append("✓ BB expansion + цена выше EMA9")
        else:
            rules_sell += 1
            details.append("✓ BB expansion + цена ниже EMA9")

    return rules_buy, rules_sell, details


def percentile_confidence(
    df: pd.DataFrame, p_buy: float, p_sell: float, *, use_ml: bool = True
) -> tuple[float, float]:
    """Percentile rank of current scores vs recent history (0–1)."""
    tail = df.iloc[max(0, len(df) - CONFIDENCE_LOOKBACK) :]
    if len(tail) < 50:
        return p_buy, p_sell

    if use_ml:
        pb_all, ps_all = predict_confidence_batch(tail)
    else:
        pb_all, ps_all = technical_scores_batch(tail)
    conf_buy = float(np.mean(pb_all <= p_buy))
    conf_sell = float(np.mean(ps_all <= p_sell))
    return conf_buy, conf_sell


def process_signals(
    df: pd.DataFrame,
    *,
    use_ml: bool | None = None,
    include_macro: bool | None = None,
) -> SignalResult:
    if use_ml is None:
        from aurum.ml import models_exist

        use_ml = models_exist()

    if include_macro is None:
        include_macro = use_ml

    thr = get_entry_thresholds(use_ml=use_ml)

    row = df.iloc[-1]
    if use_ml:
        p_buy, p_sell = predict_confidence(row)
    else:
        p_buy, p_sell = _technical_p_buy_sell(row)
    conf_buy, conf_sell = percentile_confidence(df, p_buy, p_sell, use_ml=use_ml)
    rules_buy, rules_sell, details = _count_rules(row, include_macro=include_macro)

    signal = "HOLD"
    confidence = max(conf_buy, conf_sell)

    if (
        conf_buy >= thr.confidence
        and rules_buy >= thr.min_rules
        and conf_buy > conf_sell
    ):
        signal = "BUY"
        confidence = conf_buy
    elif (
        conf_sell >= thr.confidence
        and rules_sell >= thr.min_rules
        and conf_sell > conf_buy
    ):
        signal = "SELL"
        confidence = conf_sell

    return SignalResult(
        signal=signal,
        confidence=confidence,
        p_buy=p_buy,
        p_sell=p_sell,
        conf_buy=conf_buy,
        conf_sell=conf_sell,
        rules_buy=rules_buy,
        rules_sell=rules_sell,
        rule_details=details,
    )


def consensus_score(result: SignalResult) -> float:
    if result.signal == "BUY":
        active = result.rules_buy
    elif result.signal == "SELL":
        active = result.rules_sell
    else:
        active = max(result.rules_buy, result.rules_sell)
    return active / TOTAL_RULES
