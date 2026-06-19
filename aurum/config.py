import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Market instruments (free Yahoo Finance tickers)
TICKER_GOLD = os.getenv("TICKER_GOLD", "GC=F")
GOLD_FALLBACK_TICKERS = os.getenv("GOLD_FALLBACK_TICKERS", "XAUUSD=X,MGC=F").split(",")
TICKER_DXY = os.getenv("TICKER_DXY", "DX-Y.NYB")
TICKER_US10Y = os.getenv("TICKER_US10Y", "^TNX")

# Data intervals
BAR_INTERVAL = "5m"
TRAIN_INTERVAL = "1h"
HIST_DAYS = 29
TRAIN_DAYS = 365
REALTIME_DAYS = 5

# ML artifacts
MODEL_DIR = BASE_DIR / "models"
BUY_MODEL_PATH = MODEL_DIR / "aurum_buy_model.pkl"
SELL_MODEL_PATH = MODEL_DIR / "aurum_sell_model.pkl"
SCALER_PATH = MODEL_DIR / "aurum_scaler.pkl"
META_PATH = MODEL_DIR / "aurum_meta.pkl"

# Signal thresholds (percentile rank vs recent model scores)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.90"))
MIN_RULES_CONSENSUS = int(os.getenv("MIN_RULES_CONSENSUS", "3"))
CONFIDENCE_LOOKBACK = int(os.getenv("CONFIDENCE_LOOKBACK", "500"))
TOTAL_RULES = 6

# Trade parameters
TARGET_HORIZON_BARS = 6
COST_POINTS = 0.15
MIN_BARS_FOR_FEATURES = 200
TP1_CLOSE_FRACTION = float(os.getenv("TP1_CLOSE_FRACTION", "0.5"))

# Risk defaults
MIN_DEPOSIT = float(os.getenv("MIN_DEPOSIT", "50"))
DEFAULT_DEPOSIT = float(os.getenv("DEFAULT_DEPOSIT", os.getenv("DEPOSIT", "300")))
DEFAULT_RISK_PCT = 1.0
MAX_RISK_PCT = 5.0
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "1"))
TRADE_COOLDOWN_BARS = int(os.getenv("TRADE_COOLDOWN_BARS", "6"))
MIN_RR_TP1 = float(os.getenv("MIN_RR_TP1", "1.2"))

# FxPro / cTrader Open API
CTRADER_CLIENT_ID = os.getenv("CTRADER_CLIENT_ID", "")
CTRADER_CLIENT_SECRET = os.getenv("CTRADER_CLIENT_SECRET", "")
CTRADER_REDIRECT_URI = os.getenv("CTRADER_REDIRECT_URI", "http://localhost:8501")
CTRADER_ACCESS_TOKEN = os.getenv("CTRADER_ACCESS_TOKEN", "")
CTRADER_REFRESH_TOKEN = os.getenv("CTRADER_REFRESH_TOKEN", "")
CTRADER_ACCOUNT_ID = int(os.getenv("CTRADER_ACCOUNT_ID", "0") or "0")
CTRADER_HOST = os.getenv("CTRADER_HOST", "demo").lower()  # demo | live
CTRADER_SYMBOL = os.getenv("CTRADER_SYMBOL", "XAUUSD")
CTRADER_TOKENS_PATH = DATA_DIR / "ctrader_tokens.json"
EXECUTION_STATE_PATH = DATA_DIR / "execution_state.json"

# Trading mode: paper | demo | live
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()
AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"

# Telegram (optional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# UI / worker
AUTO_REFRESH_SEC = int(os.getenv("AUTO_REFRESH_SEC", "300"))
TRADER_LOOP_SEC = int(os.getenv("TRADER_LOOP_SEC", "300"))

FEATURE_COLS = [
    "RSI",
    "MACD_Diff",
    "ATR",
    "BB_Squeeze",
    "EMA_9_Slope",
    "EMA_50_Slope",
    "Corr_DXY_Z",
    "Corr_US10Y_Z",
    "POC_Dist",
    "Vol_Ratio",
]
