"""
goat_live/config.py
Load configuration from .env (or environment variables) and return a config dict.
"""

import os
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    _dotenv_available = True
except ImportError:
    _dotenv_available = False


def load_config() -> dict:
    """Load and validate configuration from .env / environment variables."""
    # Look for .env in goat_live/ first, then repo root
    _here = Path(__file__).resolve().parent
    _env_path = _here / ".env"
    if not _env_path.exists():
        _env_path = _here.parent / ".env"

    if _dotenv_available:
        load_dotenv(dotenv_path=_env_path, override=False)

    cfg = {
        # Bybit API credentials
        "api_key": os.getenv("BYBIT_API_KEY", ""),
        "api_secret": os.getenv("BYBIT_API_SECRET", ""),
        # Trading parameters
        "symbol": os.getenv("GOAT_SYMBOL", "ONDO/USDT:USDT"),
        "timeframe": os.getenv("GOAT_TIMEFRAME", "1m"),
        "notional_usd": float(os.getenv("GOAT_NOTIONAL_USD", "20")),
        "rr_ratio": float(os.getenv("GOAT_RR_RATIO", "3.0")),
        "pivot_len": int(os.getenv("GOAT_PIVOT_LEN", "2")),
        "leverage": int(os.getenv("GOAT_LEVERAGE", "1")),
        # Safety
        "dry_run": os.getenv("GOAT_DRY_RUN", "false").lower() in ("1", "true", "yes"),
        # Logging
        "log_level": os.getenv("GOAT_LOG_LEVEL", "INFO").upper(),
        # Internal
        "warmup_bars": int(os.getenv("GOAT_WARMUP_BARS", "500")),
        "poll_interval_sec": float(os.getenv("GOAT_POLL_INTERVAL_SEC", "10")),
    }

    return cfg


def setup_logging(cfg: dict) -> None:
    """Configure root logger based on cfg['log_level']."""
    level = getattr(logging, cfg.get("log_level", "INFO"), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
