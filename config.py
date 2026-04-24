"""User-editable configuration for the 17 LABS workflow."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_TITLE = "17 LABS"
START_DATE = "2020-01-01"

VN_TICKERS = ["FPT", "VCB"]
US_TICKERS: list[str] = []
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "FET-USD", "BNB-USD"]
COMMODITY_TICKERS: list[str] = []

TRADING_DAYS_PER_YEAR = 252
NUM_PORTFOLIOS = 50_000
RANDOM_SEED = 42

RISK_FREE_RATE = 0.05
TARGET_RETURN = 0.83
TARGET_TOLERANCE = 0.02

SELECTED_PROFILE = "Balanced"
RISK_PROFILES = {
    "Conservative": {"Stocks": 0.50, "Crypto": 0.20, "Commodities": 0.30},
    "Balanced": {"Stocks": 0.40, "Crypto": 0.40, "Commodities": 0.20},
    "Aggressive": {"Stocks": 0.30, "Crypto": 0.60, "Commodities": 0.10},
}

VNSTOCK_SOURCES = ("KBS", "VCI", "TCBS", "SSI")
VNSTOCK_API_KEY = os.getenv("VNSTOCK_API_KEY", "")

