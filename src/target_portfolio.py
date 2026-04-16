"""Target-return portfolio selection helpers."""

from __future__ import annotations

import numpy as np

from src.optimization import PortfolioSelection, SimulationResult


def select_target_return_portfolio(
    simulation: SimulationResult,
    target_return: float,
    tolerance: float,
) -> PortfolioSelection | None:
    """Return the lowest-volatility portfolio within the target-return window."""
    mask = (
        (simulation.returns >= target_return - tolerance)
        & (simulation.returns <= target_return + tolerance)
    )
    if not np.any(mask):
        return None

    candidate_indices = np.where(mask)[0]
    min_volatility_index = candidate_indices[np.argmin(simulation.volatilities[mask])]

    return PortfolioSelection(
        expected_return=float(simulation.returns[min_volatility_index]),
        volatility=float(simulation.volatilities[min_volatility_index]),
        sharpe_ratio=float(simulation.sharpe_ratios[min_volatility_index]),
        weights=simulation.weights.iloc[min_volatility_index],
    )

