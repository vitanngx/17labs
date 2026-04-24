"""Shared helpers used across the 17 LABS modules."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def clean_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a forward-filled price matrix with duplicate dates removed."""
    cleaned = prices.copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
    cleaned = cleaned.sort_index()
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.ffill().dropna(how="all")
    cleaned = cleaned[cleaned.index.dayofweek < 5]
    return cleaned


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily percentage returns from a price matrix."""
    return clean_price_frame(prices).pct_change().dropna(how="all")


def annualize_returns(returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """Convert daily mean returns into annualized returns."""
    return returns.mean() * trading_days


def annualize_volatility(returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """Convert daily volatility into annualized volatility."""
    return returns.std() * np.sqrt(trading_days)


def normalize(values: Iterable[float]) -> np.ndarray:
    """Normalize a numeric vector to sum to one."""
    array = np.asarray(list(values), dtype=float)
    total = array.sum()
    if total <= 0:
        raise ValueError("Values must sum to a positive number.")
    return array / total


def format_weight_summary(weights: pd.Series, minimum_weight: float = 0.0) -> str:
    """Format portfolio weights for console output."""
    filtered = weights[weights >= minimum_weight].sort_values(ascending=False)
    lines = [f"- {ticker}: {weight:.2%}" for ticker, weight in filtered.items()]
    return "\n".join(lines) if lines else "- No weights above the threshold."

