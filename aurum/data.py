import datetime
import logging

import pandas as pd
import yfinance as yf

from aurum.config import (
    BAR_INTERVAL,
    GOLD_FALLBACK_TICKERS,
    HIST_DAYS,
    REALTIME_DAYS,
    TICKER_DXY,
    TICKER_GOLD,
    TICKER_US10Y,
)

logger = logging.getLogger(__name__)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _download_ticker(
    ticker: str,
    start: datetime.datetime,
    end: datetime.datetime,
    interval: str = BAR_INTERVAL,
) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        return _flatten_columns(df)
    except Exception as exc:
        logger.warning("Download failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _download_gold_with_fallback(start: datetime.datetime, end: datetime.datetime, interval: str) -> pd.DataFrame:
    tickers = [TICKER_GOLD] + [t for t in GOLD_FALLBACK_TICKERS if t != TICKER_GOLD]
    for ticker in tickers:
        df = _download_ticker(ticker, start, end, interval)
        if not df.empty and len(df) >= 50:
            logger.info("Gold data from %s (%d bars)", ticker, len(df))
            return df
        logger.warning("Insufficient data from %s", ticker)
    return pd.DataFrame()


def align_and_clean_data(
    df_gold: pd.DataFrame,
    df_dxy: pd.DataFrame,
    df_us10y: pd.DataFrame,
) -> pd.DataFrame:
    if df_gold.empty:
        return pd.DataFrame()

    df_gold = _flatten_columns(df_gold)
    df_dxy = _flatten_columns(df_dxy)
    df_us10y = _flatten_columns(df_us10y)

    df_gold = df_gold[["Open", "High", "Low", "Close", "Volume"]].dropna()

    dxy_close = df_dxy["Close"].reindex(df_gold.index).ffill().bfill() if not df_dxy.empty else df_gold["Close"]
    us10y_close = df_us10y["Close"].reindex(df_gold.index).ffill().bfill() if not df_us10y.empty else df_gold["Close"]

    df_gold = df_gold.copy()
    df_gold["DXY_Close"] = dxy_close
    df_gold["US10Y_Close"] = us10y_close
    return df_gold


def _fetch_aligned(start_dt: datetime.datetime, end_dt: datetime.datetime, interval: str) -> pd.DataFrame:
    return align_and_clean_data(
        _download_gold_with_fallback(start_dt, end_dt, interval),
        _download_ticker(TICKER_DXY, start_dt, end_dt, interval),
        _download_ticker(TICKER_US10Y, start_dt, end_dt, interval),
    )


def fetch_historical_data() -> pd.DataFrame:
    end_dt = datetime.datetime.now()
    start_dt = end_dt - datetime.timedelta(days=HIST_DAYS)
    return _fetch_aligned(start_dt, end_dt, BAR_INTERVAL)


def fetch_training_data() -> pd.DataFrame:
    return fetch_historical_data()


def fetch_realtime_data() -> pd.DataFrame:
    end_dt = datetime.datetime.now()
    start_dt = end_dt - datetime.timedelta(days=REALTIME_DAYS)
    return _fetch_aligned(start_dt, end_dt, BAR_INTERVAL)


def fetch_analysis_data() -> pd.DataFrame:
    """Market data with enough bars for 200-period indicators (alerts, /status, trader)."""
    end_dt = datetime.datetime.now()
    start_dt = end_dt - datetime.timedelta(days=HIST_DAYS)
    df = _fetch_aligned(start_dt, end_dt, BAR_INTERVAL)
    if not df.empty and len(df) >= 50:
        return df
    logger.warning("Analysis data sparse (%d bars), retrying extended window", len(df))
    start_dt = end_dt - datetime.timedelta(days=max(HIST_DAYS, 60))
    return _fetch_aligned(start_dt, end_dt, BAR_INTERVAL)


def is_comex_session_active(ts: datetime.datetime | None = None) -> tuple[bool, str]:
    """COMEX gold electronic session ~ Sun 18:00 ET – Fri 17:00 ET."""
    ts = ts or datetime.datetime.now(datetime.timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        et = ts.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        return True, "Сессия: данные доступны"

    wd = et.weekday()
    hour = et.hour + et.minute / 60

    if wd == 5:
        return False, "COMEX закрыт (суббота)"
    if wd == 6 and hour < 18:
        return False, "COMEX откроется в воскресенье 18:00 ET"
    if wd == 4 and hour >= 17:
        return False, "COMEX закрыт (пятница после 17:00 ET)"

    return True, f"COMEX активна • {et.strftime('%H:%M ET')}"
