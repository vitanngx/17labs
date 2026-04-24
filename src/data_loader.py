"""Market data ingestion utilities for 17 LABS."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import yfinance as yf
from vnstock import Vnstock

from config import VNSTOCK_API_KEY, VNSTOCK_SOURCES
from src.utils import clean_price_frame


def _download_yfinance_prices(tickers: list[str], start_date: str) -> pd.DataFrame:
    """Download close prices for Yahoo Finance tickers."""
    if not tickers:
        return pd.DataFrame()

    downloaded = yf.download(
        tickers,
        start=start_date,
        progress=False,
        auto_adjust=False,
    )
    close_prices = downloaded["Close"]

    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame(name=tickers[0])

    if isinstance(close_prices.columns, pd.MultiIndex):
        close_prices.columns = close_prices.columns.get_level_values(0)

    return clean_price_frame(close_prices)


def _download_vnstock_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download close prices for a single Vietnamese ticker."""
    last_error: Exception | None = None

    for source in VNSTOCK_SOURCES:
        try:
            stock_client = Vnstock().stock(symbol=ticker, source=source)
            history = stock_client.quote.history(
                start=start_date,
                end=end_date,
                interval="1D",
            )

            if history is None or history.empty:
                continue

            time_column = "time" if "time" in history.columns else "date"
            cleaned = history[[time_column, "close"]].copy()
            cleaned[time_column] = pd.to_datetime(cleaned[time_column])
            cleaned = cleaned.rename(columns={time_column: "date", "close": ticker})
            cleaned = cleaned.set_index("date")
            return clean_price_frame(cleaned)
        except Exception as error:  # pragma: no cover - data vendor failure paths
            last_error = error

    if last_error is not None:
        print(f"Skipping {ticker}: unable to fetch data from vnstock sources.")
    return pd.DataFrame()


def get_multi_asset_data(
    vn_tickers: list[str],
    us_tickers: list[str],
    crypto_tickers: list[str],
    commodity_tickers: list[str],
    start_date: str = "2020-01-01",
) -> pd.DataFrame:
    """Download and align close-price data across the configured asset universe."""
    if VNSTOCK_API_KEY:
        os.environ.setdefault("VNSTOCK_API_KEY", VNSTOCK_API_KEY)

    global_tickers = us_tickers + crypto_tickers + commodity_tickers
    global_prices = _download_yfinance_prices(global_tickers, start_date=start_date)

    end_date = datetime.now().strftime("%Y-%m-%d")
    vietnam_frames = [
        frame
        for ticker in vn_tickers
        if not (frame := _download_vnstock_prices(ticker, start_date=start_date, end_date=end_date)).empty
    ]

    if vietnam_frames:
        vietnam_prices = pd.concat(vietnam_frames, axis=1)
    else:
        vietnam_prices = pd.DataFrame()

    if global_prices.empty and vietnam_prices.empty:
        raise ValueError("No price data could be downloaded for the configured assets.")

    if global_prices.empty:
        combined = vietnam_prices
    elif vietnam_prices.empty:
        combined = global_prices
    else:
        combined = pd.concat([global_prices, vietnam_prices], axis=1)

    return clean_price_frame(combined)

