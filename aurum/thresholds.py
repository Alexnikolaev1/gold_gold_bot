"""Per-asset entry thresholds (no signal imports — avoids circular deps)."""

from dataclasses import dataclass

from aurum.config import (
    CONFIDENCE_THRESHOLD,
    CRYPTO_CONFIDENCE_THRESHOLD,
    CRYPTO_MAX_OPPOSING_RULES,
    CRYPTO_MIN_CONF_GAP,
    CRYPTO_MIN_RR_TP1,
    CRYPTO_MIN_RULES_CONSENSUS,
    GOLD_MAX_OPPOSING_RULES,
    GOLD_MIN_CONF_GAP,
    MIN_RR_TP1,
    MIN_RULES_CONSENSUS,
)


@dataclass(frozen=True)
class EntryThresholds:
    confidence: float
    min_rules: int
    max_opposing: int
    min_conf_gap: float
    min_rr: float


def get_entry_thresholds(*, use_ml: bool) -> EntryThresholds:
    """Gold (ML): 90% + 3/6. Crypto (technical): 95% + 4/6 — stricter by design."""
    if use_ml:
        return EntryThresholds(
            confidence=CONFIDENCE_THRESHOLD,
            min_rules=MIN_RULES_CONSENSUS,
            max_opposing=GOLD_MAX_OPPOSING_RULES,
            min_conf_gap=GOLD_MIN_CONF_GAP,
            min_rr=MIN_RR_TP1,
        )
    return EntryThresholds(
        confidence=CRYPTO_CONFIDENCE_THRESHOLD,
        min_rules=CRYPTO_MIN_RULES_CONSENSUS,
        max_opposing=CRYPTO_MAX_OPPOSING_RULES,
        min_conf_gap=CRYPTO_MIN_CONF_GAP,
        min_rr=CRYPTO_MIN_RR_TP1,
    )
