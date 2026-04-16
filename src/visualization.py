"""Plotting utilities for 17 LABS."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from src.optimization import PortfolioSelection, SimulationResult
from src.risk_analysis import DistributionAnalysis


def plot_asset_ranking(ranking, rf_rate: float = 0.05):
    """Plot Sharpe ratios for the ranked assets."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ranking)))
    bars = ax.bar(ranking.index, ranking["sharpe_ratio"], color=colors, edgecolor="black", alpha=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Asset Sharpe Ratio Ranking")
    ax.set_ylabel(f"Sharpe Ratio (Rf = {rf_rate:.0%})")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig, ax


def plot_simulation_frontier(
    simulation: SimulationResult,
    target_portfolio: PortfolioSelection | None = None,
):
    """Plot the simulated portfolio cloud and highlighted portfolios."""
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        simulation.volatilities,
        simulation.returns,
        c=simulation.sharpe_ratios,
        cmap="viridis",
        s=12,
        alpha=0.35,
    )
    fig.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    ax.scatter(
        simulation.best_portfolio.volatility,
        simulation.best_portfolio.expected_return,
        marker="*",
        color="red",
        s=350,
        edgecolors="black",
        label="Max Sharpe",
        zorder=5,
    )

    if target_portfolio is not None:
        ax.scatter(
            target_portfolio.volatility,
            target_portfolio.expected_return,
            marker="P",
            color="cyan",
            s=220,
            edgecolors="black",
            label="Target Return",
            zorder=5,
        )

    ax.set_title("Monte Carlo Portfolio Cloud")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Expected Return")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


def plot_return_distribution(analysis: DistributionAnalysis, title: str):
    """Plot the normal-distribution approximation for selected portfolio returns."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = np.linspace(
        analysis.expected_return - 4 * analysis.volatility,
        analysis.expected_return + 4 * analysis.volatility,
        1000,
    )

    if analysis.volatility > 0:
        density = np.exp(-0.5 * ((x_values - analysis.expected_return) / analysis.volatility) ** 2)
        density /= analysis.volatility * np.sqrt(2 * np.pi)
    else:
        density = np.zeros_like(x_values)

    ax.plot(x_values, density, color="black", linewidth=2)
    ax.fill_between(
        x_values[x_values >= 0],
        density[x_values >= 0],
        color="green",
        alpha=0.3,
        label=f"Profit ({analysis.probability_profit:.1%})",
    )
    ax.fill_between(
        x_values[x_values < 0],
        density[x_values < 0],
        color="red",
        alpha=0.3,
        label=f"Loss ({analysis.probability_loss:.1%})",
    )

    ax.axvline(
        analysis.expected_return,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Expected Return ({analysis.expected_return:.2%})",
    )
    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Annual Return")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig, ax

