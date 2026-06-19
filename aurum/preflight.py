"""System health and pre-flight checks."""

from dataclasses import dataclass, field

from aurum.config import (
    CTRADER_CLIENT_ID,
    CTRADER_CLIENT_SECRET,
    CTRADER_TOKENS_PATH,
    MODEL_DIR,
    TRADING_MODE,
)
from aurum.ml import models_exist


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def critical_ok(self) -> bool:
        critical = {"ml_models", "market_data"}
        return all(c.ok for c in self.checks if c.name in critical)


def run_preflight() -> PreflightReport:
    report = PreflightReport()

    report.checks.append(
        CheckResult("ml_models", models_exist(), "ML models trained" if models_exist() else "Run training first")
    )

    try:
        from aurum.data import fetch_realtime_data

        df = fetch_realtime_data()
        report.checks.append(
            CheckResult(
                "market_data",
                not df.empty and len(df) >= 200,
                f"{len(df)} bars loaded" if not df.empty else "Yahoo Finance unavailable",
            )
        )
    except Exception as exc:
        report.checks.append(CheckResult("market_data", False, str(exc)))

    if TRADING_MODE in ("demo", "live"):
        has_creds = bool(CTRADER_CLIENT_ID and CTRADER_CLIENT_SECRET)
        has_tokens = CTRADER_TOKENS_PATH.exists()
        report.checks.append(
            CheckResult(
                "ctrader_oauth",
                has_creds and has_tokens,
                "OAuth ready" if has_creds and has_tokens else "Configure cTrader API credentials",
            )
        )
    else:
        report.checks.append(CheckResult("ctrader_oauth", True, "Paper mode — broker optional"))

    try:
        import xgboost  # noqa: F401
        import streamlit  # noqa: F401

        report.checks.append(CheckResult("dependencies", True, "Core packages OK"))
    except ImportError as exc:
        report.checks.append(CheckResult("dependencies", False, str(exc)))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report.checks.append(CheckResult("data_dir", True, "Writable data directory"))

    return report
