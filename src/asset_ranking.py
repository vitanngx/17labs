"""Asset ranking logic based on annualized return, volatility, and Sharpe ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import annualize_returns, annualize_volatility, compute_daily_returns


def rank_assets_by_sharpe(prices: pd.DataFrame, rf_rate: float = 0.05) -> pd.DataFrame:
    """Rank assets by annualized Sharpe ratio."""
    returns = compute_daily_returns(prices)
    annual_returns = annualize_returns(returns)
    annual_volatility = annualize_volatility(returns)

    sharpe_ratio = (annual_returns - rf_rate) / annual_volatility.replace(0, np.nan)

    ranking = pd.DataFrame(
        {
            "annual_return": annual_returns,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
        }
    )
    ranking = ranking.sort_values(by="sharpe_ratio", ascending=False)
    return ranking

