#!/usr/bin/env python3
"""
Honest edge analysis — compares UI backtest vs live-like rules.

  python analyze_edge.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aurum.backtest import run_backtest, run_backtest_sweep
from aurum.config import (
    CONFIDENCE_LOOKBACK,
    CONFIDENCE_THRESHOLD,
    COST_POINTS,
    DEFAULT_DEPOSIT,
    DEFAULT_RISK_PCT,
    MIN_RR_TP1,
    MIN_RULES_CONSENSUS,
    TARGET_HORIZON_BARS,
    TRADE_COOLDOWN_BARS,
)
from aurum.data import fetch_training_data
from aurum.features import calculate_indicators
from aurum.ml import load_meta, models_exist, predict_confidence_batch, train_models
from aurum.risk import calculate_trade_levels, position_size_oz
from aurum.signals import _count_rules


@dataclass
class LiveSimResult:
    trades: pd.DataFrame
    total_pnl: float
    win_rate: float
    profit_factor: float
    max_drawdown: float


def _ensure_models() -> None:
    if models_exist():
        return
    print("Обучение ML-моделей (первый запуск)...")
    df = fetch_training_data()
    if df.empty:
        raise RuntimeError("Нет данных для обучения")
    train_models(df)


def _simulate_outcome(signal: str, entry: float, levels, future: pd.DataFrame, oz: float) -> tuple[float, str]:
    for _, bar in future.iterrows():
        high, low = float(bar["High"]), float(bar["Low"])
        if signal == "BUY":
            if low <= levels.sl:
                return (levels.sl - entry - COST_POINTS) * oz, "SL"
            if high >= levels.tp2:
                return (levels.tp2 - entry - COST_POINTS) * oz, "TP2"
            if high >= levels.tp1:
                return (levels.tp1 - entry - COST_POINTS) * oz, "TP1"
        else:
            if high >= levels.sl:
                return (entry - levels.sl - COST_POINTS) * oz, "SL"
            if low <= levels.tp2:
                return (entry - levels.tp2 - COST_POINTS) * oz, "TP2"
            if low <= levels.tp1:
                return (entry - levels.tp1 - COST_POINTS) * oz, "TP1"
    if future.empty:
        return 0.0, "TIMEOUT"
    exit_price = float(future.iloc[-1]["Close"])
    pnl = ((exit_price - entry) if signal == "BUY" else (entry - exit_price)) * oz - COST_POINTS * oz
    return pnl, "TIMEOUT"


def run_live_like_backtest(
    df: pd.DataFrame,
    deposit: float = DEFAULT_DEPOSIT,
    risk_pct: float = DEFAULT_RISK_PCT,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> LiveSimResult | None:
    """One position, cooldown, MIN_RR — mirrors alert / execution filters."""
    if not models_exist() or len(df) < 300:
        return None

    df_feat = calculate_indicators(df)
    n = len(df_feat) - TARGET_HORIZON_BARS
    if n < 50:
        return None

    p_buy_all, p_sell_all = predict_confidence_batch(df_feat.iloc[:n])
    lookback = min(CONFIDENCE_LOOKBACK, n)

    balance = deposit
    equity = [balance]
    trades: list[dict] = []
    in_trade = False
    cooldown_until = -1
    last_closed_idx = -TRADE_COOLDOWN_BARS - 1

    i = max(start_idx, 200)
    end = end_idx if end_idx is not None else n
    while i < end:
        if in_trade or i <= cooldown_until:
            i += 1
            continue

        start = max(0, i - lookback + 1)
        conf_buy = float(np.mean(p_buy_all[start : i + 1] <= p_buy_all[i]))
        conf_sell = float(np.mean(p_sell_all[start : i + 1] <= p_sell_all[i]))
        row = df_feat.iloc[i]
        rules_buy, rules_sell, _ = _count_rules(row)

        signal = "HOLD"
        confidence = max(conf_buy, conf_sell)
        if conf_buy >= confidence_threshold and rules_buy >= MIN_RULES_CONSENSUS and conf_buy > conf_sell:
            signal, confidence = "BUY", conf_buy
        elif conf_sell >= confidence_threshold and rules_sell >= MIN_RULES_CONSENSUS and conf_sell > conf_buy:
            signal, confidence = "SELL", conf_sell

        if signal == "HOLD":
            i += 1
            continue

        levels = calculate_trade_levels(df_feat.iloc[: i + 1], signal)
        if levels.risk_reward_tp1 < MIN_RR_TP1:
            i += 1
            continue

        oz = position_size_oz(balance, risk_pct, levels.risk_per_oz)
        if oz <= 0:
            i += 1
            continue

        entry = float(row["Close"])
        future = df_feat.iloc[i + 1 : i + 1 + TARGET_HORIZON_BARS]
        pnl, outcome = _simulate_outcome(signal, entry, levels, future, oz)
        balance += pnl
        equity.append(balance)
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
                "confidence": confidence,
                "rr": levels.risk_reward_tp1,
            }
        )
        in_trade = False
        last_closed_idx = i
        cooldown_until = i + TRADE_COOLDOWN_BARS
        i = cooldown_until + 1

    if not trades:
        return LiveSimResult(pd.DataFrame(), 0.0, 0.0, 0.0, 0.0)

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    wr = len(wins) / len(trades_df)
    gp = wins["pnl"].sum() if not wins.empty else 0.0
    gl = abs(losses["pnl"].sum()) if not losses.empty else 1e-9
    pf = gp / gl
    eq = pd.Series(equity)
    dd = abs(((eq - eq.cummax()) / eq.cummax()).min()) if len(eq) > 1 else 0.0
    return LiveSimResult(trades_df, trades_df["pnl"].sum(), wr, pf, dd)


def _side_stats(trades: pd.DataFrame, side: str) -> str:
    s = trades[trades["signal"] == side]
    if s.empty:
        return f"  {side}: 0 сделок"
    return f"  {side}: {len(s)} сделок | win rate {(s.pnl > 0).mean():.1%} | PnL ${s.pnl.sum():,.0f}"


def _verdict(ui_wr: float, ui_pf: float, live: LiveSimResult, oos: LiveSimResult | None) -> str:
    lines = ["=" * 50, "VERDICT", "=" * 50]
    if live.trades.empty:
        lines.append("[!] Live-like: no trades at 90% + R:R>=1.2 filters.")
        lines.append("    Telegram alerts will be VERY rare.")
        return "\n".join(lines)

    n = len(live.trades)
    lines.append(f"Live-like: {n} trades | WR {live.win_rate:.1%} | PF {live.profit_factor:.2f} | PnL ${live.total_pnl:,.0f}")

    if n < 30:
        lines.append(f"[!] Too few trades ({n} < 30) — statistics NOT reliable.")
    elif live.win_rate >= 0.55 and live.profit_factor >= 1.3:
        lines.append("[OK] Historical edge hint — confirm on demo 1-2 months.")
    else:
        lines.append("[X] Weak or no edge on live-like rules.")

    if ui_wr > live.win_rate + 0.15:
        lines.append(
            f"[!] UI backtest overstates WR ({ui_wr:.0%} vs {live.win_rate:.0%}) — do not trust tab numbers blindly."
        )

    if oos and not oos.trades.empty:
        lines.append(
            f"Out-of-sample (last 20%): {len(oos.trades)} trades | WR {oos.win_rate:.1%} | PnL ${oos.total_pnl:,.0f}"
        )
    elif oos:
        lines.append("Out-of-sample: 0 trades — 90% threshold too strict on recent history.")

    lines.append("\nRecommendation: paper/demo 4-8 weeks, 1% risk, $300 as learning capital.")
    return "\n".join(lines)


def main() -> int:
    _ensure_models()
    meta = load_meta()
    df = fetch_training_data()
    if df.empty:
        print("ERROR: нет рыночных данных")
        return 1

    deposit = DEFAULT_DEPOSIT
    print("=" * 50)
    print("AURUM — ЧЕСТНЫЙ АНАЛИЗ EDGE")
    print("=" * 50)
    if meta:
        print(f"ML holdout: Buy precision {meta.buy_precision:.1%} | Sell precision {meta.sell_precision:.1%}")
        print(f"Обучающих баров: {meta.samples} | История: {len(df)} баров (1h train interval)")
    print(f"Депозит: ${deposit:.0f} | Риск: {DEFAULT_RISK_PCT}% | Порог: {CONFIDENCE_THRESHOLD:.0%}")
    print()

    ui = run_backtest(df, deposit=deposit, risk_pct=DEFAULT_RISK_PCT)
    print("--- 1) БЭКТЕСТ UI (упрощённый, как вкладка «Бэктест») ---")
    if ui and ui.total_trades:
        print(f"Сделок: {ui.total_trades} | Win rate: {ui.win_rate:.1%} | PF: {ui.profit_factor:.2f}")
        print(f"PnL: ${ui.total_pnl:,.0f} | Max DD: {ui.max_drawdown:.1%}")
        print("[!] No cooldown, no 1-position limit, no R:R>=1.2 filter — OPTIMISTIC")
    else:
        print("Нет сделок")

    print()
    print("--- 2) LIVE-LIKE (как Telegram-алерты / run_trader) ---")
    print(f"Фильтры: confidence≥{CONFIDENCE_THRESHOLD:.0%}, {MIN_RULES_CONSENSUS}/6 правил, R:R≥{MIN_RR_TP1}, cooldown {TRADE_COOLDOWN_BARS} баров")
    live = run_live_like_backtest(df, deposit=deposit)
    if live and not live.trades.empty:
        print(f"Сделок: {len(live.trades)} | Win rate: {live.win_rate:.1%} | PF: {live.profit_factor:.2f}")
        print(f"PnL: ${live.total_pnl:,.0f} | Max DD: {live.max_drawdown:.1%}")
        print(f"Expectancy/сделка: ${live.trades.pnl.mean():,.2f}")
        t0, t1 = live.trades["time"].min(), live.trades["time"].max()
        days = max((pd.Timestamp(t1) - pd.Timestamp(t0)).days, 1)
        print(f"Частота: ~{len(live.trades) / days * 30:.1f} сделок/месяц")
        print(_side_stats(live.trades, "BUY"))
        print(_side_stats(live.trades, "SELL"))
    else:
        print("Нет сделок при live-like правилах")

    print()
    print("--- 3) OUT-OF-SAMPLE (последние 20% времени) ---")
    cut = int(len(df) * 0.8)
    oos_df = df.iloc[cut:]
    df_feat_full = calculate_indicators(df)
    oos_start = int(len(df_feat_full) * 0.8)
    oos = run_live_like_backtest(df, deposit=deposit, start_idx=oos_start)
    if oos and not oos.trades.empty:
        print(f"Сделок: {len(oos.trades)} | Win rate: {oos.win_rate:.1%} | PnL: ${oos.total_pnl:,.0f}")
    else:
        print("0 сделок — модель/фильтры не давали входов на свежей истории")

    print()
    print("--- 4) СВЕП ПОРОГА (UI-метод, для сравнения) ---")
    sweep = run_backtest_sweep(df, deposit=deposit)
    if not sweep.empty:
        print(sweep.to_string(index=False))

    print()
    ui_wr = ui.win_rate if ui else 0.0
    ui_pf = ui.profit_factor if ui else 0.0
    print(_verdict(ui_wr, ui_pf, live or LiveSimResult(pd.DataFrame(), 0, 0, 0, 0), oos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
