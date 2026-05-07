"""Reusable analysis workflow for CLI and UI entry points."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.asset_ranking import rank_assets_by_sharpe
from src.data_loader import get_multi_asset_data
from src.optimization import SimulationResult, simulate_portfolios
from src.risk_analysis import DistributionAnalysis, analyze_return_distribution
from src.target_portfolio import select_target_return_portfolio
from src.optimization import PortfolioSelection


@dataclass(frozen=True)
class AnalysisConfig:
    """Inputs required to run one portfolio analysis."""

    vn_tickers: list[str]
    us_tickers: list[str]
    crypto_tickers: list[str]
    commodity_tickers: list[str]
    start_date: str
    risk_budget: dict[str, float]
    selected_profile: str
    risk_free_rate: float
    target_return: float
    target_tolerance: float
    num_portfolios: int
    random_seed: int | None


@dataclass(frozen=True)
class AnalysisResult:
    """Outputs produced by the end-to-end analysis."""

    prices: pd.DataFrame
    ranking: pd.DataFrame
    simulation: SimulationResult
    target_portfolio: PortfolioSelection | None
    selected_portfolio: PortfolioSelection
    distribution: DistributionAnalysis


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Run the complete 17 LABS workflow from prices to risk distribution."""
    prices = get_multi_asset_data(
        vn_tickers=config.vn_tickers,
        us_tickers=config.us_tickers,
        crypto_tickers=config.crypto_tickers,
        commodity_tickers=config.commodity_tickers,
        start_date=config.start_date,
    )
    ranking = rank_assets_by_sharpe(prices, rf_rate=config.risk_free_rate)
    simulation = simulate_portfolios(
        prices=prices,
        risk_budget=config.risk_budget,
        stock_tickers=config.vn_tickers + config.us_tickers,
        crypto_tickers=config.crypto_tickers,
        commodity_tickers=config.commodity_tickers,
        rf_rate=config.risk_free_rate,
        num_portfolios=config.num_portfolios,
        random_seed=config.random_seed,
    )
    target_portfolio = select_target_return_portfolio(
        simulation=simulation,
        target_return=config.target_return,
        tolerance=config.target_tolerance,
    )
    selected_portfolio = target_portfolio or simulation.best_portfolio
    distribution = analyze_return_distribution(
        expected_return=selected_portfolio.expected_return,
        volatility=selected_portfolio.volatility,
    )

    return AnalysisResult(
        prices=prices,
        ranking=ranking,
        simulation=simulation,
        target_portfolio=target_portfolio,
        selected_portfolio=selected_portfolio,
        distribution=distribution,
    )
