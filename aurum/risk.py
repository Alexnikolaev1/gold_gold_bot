from dataclasses import dataclass

import pandas as pd


@dataclass
class TradeLevels:
    tp1: float
    tp2: float
    sl: float
    risk_reward_tp1: float
    risk_reward_tp2: float
    risk_per_oz: float


def calculate_trade_levels(df: pd.DataFrame, signal: str) -> TradeLevels:
    row = df.iloc[-1]
    atr = float(row["ATR"])
    close = float(row["Close"])
    risk_mult = 1.5 if row["BB_Squeeze"] < 0.015 else 2.0

    if signal == "BUY":
        raw_sl = close - (atr * risk_mult)
        sl = max(raw_sl, float(row["Swing_Low"]) - 0.5)
        tp1 = close + (atr * 1.5)
        tp2 = close + (atr * 2.5)
    elif signal == "SELL":
        raw_sl = close + (atr * risk_mult)
        sl = min(raw_sl, float(row["Swing_High"]) + 0.5)
        tp1 = close - (atr * 1.5)
        tp2 = close - (atr * 2.5)
    else:
        return TradeLevels(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    tp1, tp2, sl = round(tp1, 1), round(tp2, 1), round(sl, 1)
    risk_per_oz = abs(close - sl)
    reward1 = abs(tp1 - close)
    reward2 = abs(tp2 - close)
    rr1 = reward1 / risk_per_oz if risk_per_oz > 0 else 0.0
    rr2 = reward2 / risk_per_oz if risk_per_oz > 0 else 0.0

    return TradeLevels(tp1=tp1, tp2=tp2, sl=sl, risk_reward_tp1=rr1, risk_reward_tp2=rr2, risk_per_oz=risk_per_oz)


def position_size_oz(deposit: float, risk_pct: float, risk_per_oz: float) -> float:
    if risk_per_oz <= 0:
        return 0.0
    risk_usd = deposit * (risk_pct / 100.0)
    return risk_usd / risk_per_oz
