from dataclasses import dataclass

import numpy as np
import pandas as pd

from aurum.config import CONFIDENCE_LOOKBACK, CONFIDENCE_THRESHOLD, COST_POINTS, MIN_RULES_CONSENSUS, TARGET_HORIZON_BARS
from aurum.features import calculate_indicators
from aurum.ml import models_exist, predict_confidence_batch
from aurum.risk import calculate_trade_levels
from aurum.signals import _count_rules, percentile_confidence


@dataclass
class BacktestResult:
    total_trades: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    max_drawdown: float
    sharpe: float
    avg_rr: float
    equity_curve: pd.Series
    trades: pd.DataFrame


def _simulate_signal_at_row(
    row: pd.Series,
    p_buy: float,
    p_sell: float,
    conf_buy: float,
    conf_sell: float,
    confidence_threshold: float,
    min_rules: int,
) -> str:
    rules_buy, rules_sell, _ = _count_rules(row)
    if conf_buy >= confidence_threshold and rules_buy >= min_rules and conf_buy > conf_sell:
        return "BUY"
    if conf_sell >= confidence_threshold and rules_sell >= min_rules and conf_sell > conf_buy:
        return "SELL"
    return "HOLD"


def run_backtest(
    df: pd.DataFrame,
    deposit: float = 50_000,
    risk_pct: float = 1.0,
    confidence_threshold: float | None = None,
    min_rules: int | None = None,
) -> BacktestResult | None:
    confidence_threshold = CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
    min_rules = MIN_RULES_CONSENSUS if min_rules is None else min_rules
    if not models_exist() or len(df) < 300:
        return None

    df_feat = calculate_indicators(df)
    n = len(df_feat) - TARGET_HORIZON_BARS
    if n < 50:
        return None

    p_buy_all, p_sell_all = predict_confidence_batch(df_feat.iloc[:n])

    # Precompute rolling percentile confidence
    lookback = min(CONFIDENCE_LOOKBACK, n)
    conf_buy_all = np.zeros(n)
    conf_sell_all = np.zeros(n)
    for i in range(n):
        start = max(0, i - lookback + 1)
        pb_window = p_buy_all[start : i + 1]
        ps_window = p_sell_all[start : i + 1]
        conf_buy_all[i] = float(np.mean(pb_window <= p_buy_all[i])) if len(pb_window) else 0.5
        conf_sell_all[i] = float(np.mean(ps_window <= p_sell_all[i])) if len(ps_window) else 0.5

    trades = []
    equity = [deposit]

    for i in range(n):
        row = df_feat.iloc[i]
        p_buy, p_sell = float(p_buy_all[i]), float(p_sell_all[i])
        conf_buy, conf_sell = float(conf_buy_all[i]), float(conf_sell_all[i])
        signal = _simulate_signal_at_row(row, p_buy, p_sell, conf_buy, conf_sell, confidence_threshold, min_rules)
        if signal == "HOLD":
            continue

        levels = calculate_trade_levels(df_feat.iloc[: i + 1], signal)
        if levels.risk_per_oz <= 0:
            continue

        oz = (deposit * risk_pct / 100.0) / levels.risk_per_oz
        entry = float(row["Close"])
        future = df_feat.iloc[i + 1 : i + 1 + TARGET_HORIZON_BARS]

        pnl = 0.0
        outcome = "TIMEOUT"
        for _, bar in future.iterrows():
            high, low = float(bar["High"]), float(bar["Low"])
            if signal == "BUY":
                if low <= levels.sl:
                    pnl = (levels.sl - entry - COST_POINTS) * oz
                    outcome = "SL"
                    break
                if high >= levels.tp2:
                    pnl = (levels.tp2 - entry - COST_POINTS) * oz
                    outcome = "TP2"
                    break
                if high >= levels.tp1:
                    pnl = (levels.tp1 - entry - COST_POINTS) * oz
                    outcome = "TP1"
                    break
            else:
                if high >= levels.sl:
                    pnl = (entry - levels.sl - COST_POINTS) * oz
                    outcome = "SL"
                    break
                if low <= levels.tp2:
                    pnl = (entry - levels.tp2 - COST_POINTS) * oz
                    outcome = "TP2"
                    break
                if low <= levels.tp1:
                    pnl = (entry - levels.tp1 - COST_POINTS) * oz
                    outcome = "TP1"
                    break

        if outcome == "TIMEOUT" and not future.empty:
            exit_price = float(future.iloc[-1]["Close"])
            pnl = ((exit_price - entry) if signal == "BUY" else (entry - exit_price)) * oz - COST_POINTS * oz

        deposit += pnl
        equity.append(deposit)
        trades.append(
            {
                "time": df_feat.index[i],
                "signal": signal,
                "entry": entry,
                "sl": levels.sl,
                "tp1": levels.tp1,
                "tp2": levels.tp2,
                "pnl": pnl,
                "outcome": outcome,
                "confidence": conf_buy if signal == "BUY" else conf_sell,
            }
        )

    if not trades:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, pd.Series(equity), pd.DataFrame())

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(wins) / len(trades_df)
    gross_profit = wins["pnl"].sum() if not wins.empty else 0
    gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    eq = pd.Series(equity)
    returns = eq.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 78)) if returns.std() > 0 else 0

    dd = (eq - eq.cummax()) / eq.cummax()
    max_dd = abs(dd.min()) if len(dd) else 0

    avg_rr = trades_df.apply(
        lambda t: abs(t["tp1"] - t["entry"]) / abs(t["entry"] - t["sl"]) if abs(t["entry"] - t["sl"]) > 0 else 0,
        axis=1,
    ).mean()

    return BacktestResult(
        total_trades=len(trades_df),
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_pnl=trades_df["pnl"].sum(),
        max_drawdown=max_dd,
        sharpe=sharpe,
        avg_rr=avg_rr,
        equity_curve=eq,
        trades=trades_df,
    )


def run_backtest_sweep(
    df: pd.DataFrame,
    deposit: float = 50_000,
    risk_pct: float = 1.0,
    thresholds: tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90),
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        bt = run_backtest(df, deposit, risk_pct, confidence_threshold=thr)
        if bt is None:
            continue
        rows.append(
            {
                "Порог confidence": f"{thr:.0%}",
                "Сделок": bt.total_trades,
                "Win Rate": f"{bt.win_rate:.1%}",
                "Profit Factor": f"{bt.profit_factor:.2f}",
                "PnL ($)": f"{bt.total_pnl:,.0f}",
                "Max DD": f"{bt.max_drawdown:.1%}",
            }
        )
    return pd.DataFrame(rows)
