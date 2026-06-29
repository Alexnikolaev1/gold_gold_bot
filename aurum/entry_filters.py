"""Strict entry filters — alerts only on maximum-confidence setups."""

from aurum.signals import SignalResult
from aurum.thresholds import get_entry_thresholds


def passes_strong_entry(result: SignalResult, *, use_ml: bool) -> bool:
    """True only when direction is clear and confidence is in the top tier."""
    if result.signal not in ("BUY", "SELL"):
        return False

    thr = get_entry_thresholds(use_ml=use_ml)
    if result.confidence < thr.confidence:
        return False

    active = result.rules_buy if result.signal == "BUY" else result.rules_sell
    opposing = result.rules_sell if result.signal == "BUY" else result.rules_buy
    if active < thr.min_rules or opposing > thr.max_opposing:
        return False

    if result.signal == "BUY":
        gap = result.conf_buy - result.conf_sell
    else:
        gap = result.conf_sell - result.conf_buy
    if gap < thr.min_conf_gap:
        return False

    return True
