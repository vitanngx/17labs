"""Probability analysis for annual portfolio returns."""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class DistributionAnalysis:
    """Summary of a normal-distribution approximation for portfolio returns."""

    expected_return: float
    volatility: float
    probability_profit: float
    probability_loss: float
    interval_68: tuple[float, float]
    interval_95: tuple[float, float]


def analyze_return_distribution(
    expected_return: float,
    volatility: float,
) -> DistributionAnalysis:
    """Estimate profit probability and confidence intervals using a normal model."""
    if volatility <= 0:
        probability_profit = 1.0 if expected_return > 0 else 0.0
        probability_loss = 1.0 - probability_profit
        interval_68 = (expected_return, expected_return)
        interval_95 = (expected_return, expected_return)
        return DistributionAnalysis(
            expected_return=expected_return,
            volatility=volatility,
            probability_profit=probability_profit,
            probability_loss=probability_loss,
            interval_68=interval_68,
            interval_95=interval_95,
        )

    z_score = (0 - expected_return) / volatility
    probability_loss = float(stats.norm.cdf(z_score))
    probability_profit = 1.0 - probability_loss
    interval_68 = (expected_return - volatility, expected_return + volatility)
    interval_95 = (expected_return - 1.96 * volatility, expected_return + 1.96 * volatility)

    return DistributionAnalysis(
        expected_return=expected_return,
        volatility=volatility,
        probability_profit=probability_profit,
        probability_loss=probability_loss,
        interval_68=interval_68,
        interval_95=interval_95,
    )

