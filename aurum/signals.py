from dataclasses import dataclass

import numpy as np
import pandas as pd

from aurum.config import CONFIDENCE_LOOKBACK, CONFIDENCE_THRESHOLD, MIN_RULES_CONSENSUS, TOTAL_RULES
from aurum.ml import predict_confidence, predict_confidence_batch


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


def _count_rules(row: pd.Series) -> tuple[int, int, list[str]]:
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


def percentile_confidence(df: pd.DataFrame, p_buy: float, p_sell: float) -> tuple[float, float]:
    """Percentile rank of current scores vs recent history (0–1)."""
    tail = df.iloc[max(0, len(df) - CONFIDENCE_LOOKBACK) :]
    if len(tail) < 50:
        return p_buy, p_sell

    pb_all, ps_all = predict_confidence_batch(tail)
    conf_buy = float(np.mean(pb_all <= p_buy))
    conf_sell = float(np.mean(ps_all <= p_sell))
    return conf_buy, conf_sell


def process_signals(df: pd.DataFrame) -> SignalResult:
    row = df.iloc[-1]
    p_buy, p_sell = predict_confidence(row)
    conf_buy, conf_sell = percentile_confidence(df, p_buy, p_sell)
    rules_buy, rules_sell, details = _count_rules(row)

    signal = "HOLD"
    confidence = max(conf_buy, conf_sell)

    if conf_buy >= CONFIDENCE_THRESHOLD and rules_buy >= MIN_RULES_CONSENSUS and conf_buy > conf_sell:
        signal = "BUY"
        confidence = conf_buy
    elif conf_sell >= CONFIDENCE_THRESHOLD and rules_sell >= MIN_RULES_CONSENSUS and conf_sell > conf_buy:
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
