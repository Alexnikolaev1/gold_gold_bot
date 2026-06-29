"""Hash Hedge asset list and Yahoo Finance ticker mapping."""

import json
import logging
from functools import lru_cache
from pathlib import Path

from aurum.config import BASE_DIR, DATA_DIR

logger = logging.getLogger(__name__)

COINS_PATH = DATA_DIR / "hashhedge_coins.json"
FALLBACK_COINS_PATH = BASE_DIR / "data" / "hashhedge_coins.json"

# Hash Hedge pair on platform is typically SYMBOL/USDT
COMMODITY_YAHOO = {
    "XAU": ("GC=F", "XAUUSD=X", "MGC=F"),
    "XAG": ("SI=F", "XAGUSD=X"),
    "XPT": ("PL=F",),
    "XPD": ("PA=F",),
}
STOCK_YAHOO = {"TSLA": "TSLA"}
CRYPTO_YAHOO_OVERRIDES = {
    "1INCH": "1INCH-USD",
    "DODOX": "DODO-USD",
    "BEAMX": "BEAM-USD",
}
METALS = frozenset({"XAU", "XAG", "XPT", "XPD"})


@lru_cache(maxsize=1)
def load_hashhedge_symbols() -> tuple[str, ...]:
    """Liquid Hash Hedge assets only — micro-caps and unreliable tickers excluded."""
    path = COINS_PATH if COINS_PATH.exists() else FALLBACK_COINS_PATH
    if not path.exists():
        logger.warning("hashhedge_coins.json not found")
        return ("XAU", "BTC", "ETH")
    raw = json.loads(path.read_text(encoding="utf-8"))
    symbols = [str(s).upper().strip() for s in raw if str(s).strip()]
    return tuple(dict.fromkeys(symbols))


# Removed from full Hash Hedge list (no reliable Yahoo data or too illiquid for max-confidence alerts):
# RATS, BOME, HMSTR, USTC, SPELL, PEOPLE, WLFI, LINEA, ASTER, PUMP, CATI, MON, NFP, ACE,
# TSLA, XPT, XPD, micro-cap alts with thin history — see git history for full 144 list.


def yahoo_ticker_candidates(symbol: str) -> tuple[str, ...]:
    """Ordered Yahoo tickers to try for a Hash Hedge symbol."""
    sym = symbol.upper()
    if sym in COMMODITY_YAHOO:
        return COMMODITY_YAHOO[sym]
    if sym in STOCK_YAHOO:
        return (STOCK_YAHOO[sym],)
    if sym in CRYPTO_YAHOO_OVERRIDES:
        return (CRYPTO_YAHOO_OVERRIDES[sym], f"{sym}-USD")
    return (f"{sym}-USD",)


def hashhedge_pair_label(symbol: str) -> str:
    sym = symbol.upper()
    if sym in METALS or sym == "TSLA":
        return sym
    return f"{sym}/USDT"


def is_metal(symbol: str) -> bool:
    return symbol.upper() in METALS


def uses_ml_model(symbol: str) -> bool:
    """Only gold uses trained XGBoost; crypto uses technical percentile scoring."""
    return symbol.upper() == "XAU"
