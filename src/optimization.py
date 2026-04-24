"""Monte Carlo portfolio optimization with group budgets and asset constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils import annualize_returns, compute_daily_returns, normalize


@dataclass(frozen=True)
class PortfolioSelection:
    """Summary statistics and weights for a single portfolio."""

    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: pd.Series


@dataclass(frozen=True)
class SimulationResult:
    """Container for Monte Carlo simulation outputs."""

    volatilities: np.ndarray
    returns: np.ndarray
    sharpe_ratios: np.ndarray
    weights: pd.DataFrame
    best_portfolio: PortfolioSelection
    total_attempts: int
    failed_attempts: int


def get_weight_constraints(n_assets: int) -> tuple[float, float]:
    """Return the maximum and minimum weight per asset."""
    constraints = {
        2: (0.80, 0.00),
        3: (0.60, 0.20),
        4: (0.55, 0.15),
        5: (0.50, 0.12),
        6: (0.45, 0.10),
        7: (0.40, 0.08),
        8: (0.35, 0.06),
        9: (0.30, 0.04),
        10: (0.25, 0.02),
    }
    return constraints.get(n_assets, (1.0 / n_assets + 0.2, 0.01))


def build_asset_groups(
    tickers: list[str],
    stock_tickers: list[str],
    crypto_tickers: list[str],
    commodity_tickers: list[str],
) -> dict[str, list[str]]:
    """Assign active tickers to portfolio groups."""
    groups = {
        "Stocks": [ticker for ticker in tickers if ticker in stock_tickers],
        "Crypto": [ticker for ticker in tickers if ticker in crypto_tickers],
        "Commodities": [ticker for ticker in tickers if ticker in commodity_tickers],
    }
    return {group_name: names for group_name, names in groups.items() if names}


def normalize_group_budget(
    risk_budget: dict[str, float],
    active_groups: dict[str, list[str]],
) -> dict[str, float]:
    """Renormalize group budgets if some configured groups are inactive."""
    total = sum(risk_budget[group_name] for group_name in active_groups)
    if total <= 0:
        raise ValueError("Risk budget must allocate positive weight to active groups.")
    return {group_name: risk_budget[group_name] / total for group_name in active_groups}


def simulate_portfolios(
    prices: pd.DataFrame,
    risk_budget: dict[str, float],
    stock_tickers: list[str],
    crypto_tickers: list[str],
    commodity_tickers: list[str],
    rf_rate: float = 0.05,
    num_portfolios: int = 50_000,
    max_failed_attempts: int = 500_000,
    random_seed: int | None = None,
) -> SimulationResult:
    """Run constrained Monte Carlo portfolio simulation."""
    clean_prices = prices.ffill()
    returns = compute_daily_returns(clean_prices)
    annual_returns = annualize_returns(returns)
    covariance = returns.cov() * 252

    tickers = list(clean_prices.columns)
    n_assets = len(tickers)
    max_weight, min_weight = get_weight_constraints(n_assets)

    active_groups = build_asset_groups(
        tickers=tickers,
        stock_tickers=stock_tickers,
        crypto_tickers=crypto_tickers,
        commodity_tickers=commodity_tickers,
    )
    if not active_groups:
        raise ValueError("No active asset groups were found for the provided tickers.")

    final_budget = normalize_group_budget(risk_budget, active_groups)
    rng = np.random.default_rng(random_seed)

    accepted_volatility: list[float] = []
    accepted_return: list[float] = []
    accepted_sharpe: list[float] = []
    weight_records: list[np.ndarray] = []
    failed_attempts = 0

    while len(weight_records) < num_portfolios:
        weights = np.zeros(n_assets, dtype=float)

        for group_name, group_tickers in active_groups.items():
            group_weights = normalize(rng.random(len(group_tickers)))
            indices = [tickers.index(ticker) for ticker in group_tickers]
            weights[indices] = group_weights * final_budget[group_name]

        if np.all(weights <= max_weight + 1e-4) and np.all(weights >= min_weight - 1e-4):
            portfolio_return = float(weights @ annual_returns.values)
            portfolio_volatility = float(np.sqrt(weights.T @ covariance.values @ weights))
            sharpe_ratio = np.nan
            if portfolio_volatility > 0:
                sharpe_ratio = (portfolio_return - rf_rate) / portfolio_volatility

            accepted_volatility.append(portfolio_volatility)
            accepted_return.append(portfolio_return)
            accepted_sharpe.append(sharpe_ratio)
            weight_records.append(weights.copy())
            continue

        failed_attempts += 1
        if failed_attempts > max_failed_attempts:
            raise ValueError(
                "Constraints and group budgets did not produce enough valid portfolios. "
                "Review the asset count or budget settings."
            )

    weight_frame = pd.DataFrame(weight_records, columns=tickers)
    sharpes = np.asarray(accepted_sharpe, dtype=float)
    best_index = int(np.nanargmax(sharpes))
    best_portfolio = PortfolioSelection(
        expected_return=accepted_return[best_index],
        volatility=accepted_volatility[best_index],
        sharpe_ratio=accepted_sharpe[best_index],
        weights=weight_frame.iloc[best_index],
    )

    return SimulationResult(
        volatilities=np.asarray(accepted_volatility, dtype=float),
        returns=np.asarray(accepted_return, dtype=float),
        sharpe_ratios=sharpes,
        weights=weight_frame,
        best_portfolio=best_portfolio,
        total_attempts=len(weight_records) + failed_attempts,
        failed_attempts=failed_attempts,
    )

