"""Run the 17 LABS workflow end-to-end."""

from __future__ import annotations

import matplotlib.pyplot as plt

from config import (
    COMMODITY_TICKERS,
    CRYPTO_TICKERS,
    NUM_PORTFOLIOS,
    PROJECT_TITLE,
    RANDOM_SEED,
    RISK_FREE_RATE,
    RISK_PROFILES,
    SELECTED_PROFILE,
    START_DATE,
    TARGET_RETURN,
    TARGET_TOLERANCE,
    US_TICKERS,
    VN_TICKERS,
)
from src.asset_ranking import rank_assets_by_sharpe
from src.data_loader import get_multi_asset_data
from src.optimization import simulate_portfolios
from src.risk_analysis import analyze_return_distribution
from src.target_portfolio import select_target_return_portfolio
from src.utils import format_weight_summary
from src.visualization import (
    plot_asset_ranking,
    plot_return_distribution,
    plot_simulation_frontier,
)


def main() -> None:
    """Execute the default analysis workflow."""
    prices = get_multi_asset_data(
        vn_tickers=VN_TICKERS,
        us_tickers=US_TICKERS,
        crypto_tickers=CRYPTO_TICKERS,
        commodity_tickers=COMMODITY_TICKERS,
        start_date=START_DATE,
    )

    ranking = rank_assets_by_sharpe(prices, rf_rate=RISK_FREE_RATE)
    simulation = simulate_portfolios(
        prices=prices,
        risk_budget=RISK_PROFILES[SELECTED_PROFILE],
        stock_tickers=VN_TICKERS + US_TICKERS,
        crypto_tickers=CRYPTO_TICKERS,
        commodity_tickers=COMMODITY_TICKERS,
        rf_rate=RISK_FREE_RATE,
        num_portfolios=NUM_PORTFOLIOS,
        random_seed=RANDOM_SEED,
    )
    target_portfolio = select_target_return_portfolio(
        simulation=simulation,
        target_return=TARGET_RETURN,
        tolerance=TARGET_TOLERANCE,
    )
    selected_portfolio = target_portfolio or simulation.best_portfolio
    distribution = analyze_return_distribution(
        expected_return=selected_portfolio.expected_return,
        volatility=selected_portfolio.volatility,
    )

    print(f"\n{PROJECT_TITLE} price matrix shape: {prices.shape}")
    print("\nAsset ranking (annualized):")
    print(ranking.round(4).to_string())

    print(f"\nBest portfolio for profile '{SELECTED_PROFILE}':")
    print(f"Expected return: {simulation.best_portfolio.expected_return:.2%}")
    print(f"Volatility: {simulation.best_portfolio.volatility:.2%}")
    print(f"Sharpe ratio: {simulation.best_portfolio.sharpe_ratio:.3f}")
    print(format_weight_summary(simulation.best_portfolio.weights))

    if target_portfolio is None:
        print(
            f"\nNo portfolio was found within {TARGET_RETURN:.2%} +/- "
            f"{TARGET_TOLERANCE:.2%}. Using the max-Sharpe portfolio for risk analysis."
        )
    else:
        print(f"\nTarget-return portfolio around {TARGET_RETURN:.2%}:")
        print(f"Expected return: {target_portfolio.expected_return:.2%}")
        print(f"Volatility: {target_portfolio.volatility:.2%}")
        print(f"Sharpe ratio: {target_portfolio.sharpe_ratio:.3f}")
        print(format_weight_summary(target_portfolio.weights, minimum_weight=0.01))

    print("\nReturn distribution summary:")
    print(f"Probability of profit: {distribution.probability_profit:.2%}")
    print(f"Probability of loss: {distribution.probability_loss:.2%}")
    print(
        "68% interval: "
        f"{distribution.interval_68[0]:.2%} to {distribution.interval_68[1]:.2%}"
    )
    print(
        "95% interval: "
        f"{distribution.interval_95[0]:.2%} to {distribution.interval_95[1]:.2%}"
    )

    plot_asset_ranking(ranking, rf_rate=RISK_FREE_RATE)
    plot_simulation_frontier(simulation, target_portfolio=target_portfolio)
    title = "Target Portfolio Return Distribution" if target_portfolio else "Max-Sharpe Return Distribution"
    plot_return_distribution(distribution, title=title)
    plt.show()


if __name__ == "__main__":
    main()

