"""Streamlit dashboard for the 17 LABS portfolio workflow."""

from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

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
from src.optimization import PortfolioSelection
from src.visualization import (
    plot_asset_ranking,
    plot_return_distribution,
    plot_simulation_frontier,
)
from src.workflow import AnalysisConfig, AnalysisResult, run_analysis


def parse_tickers(raw_value: str) -> list[str]:
    """Parse comma-separated tickers from sidebar text inputs."""
    return [ticker.strip().upper() for ticker in raw_value.split(",") if ticker.strip()]


def join_tickers(tickers: list[str]) -> str:
    """Format tickers as a comma-separated input default."""
    return ", ".join(tickers)


def format_percent(value: float) -> str:
    """Format decimal returns and weights as percentages."""
    return f"{value:.2%}"


def portfolio_weights_frame(portfolio: PortfolioSelection) -> pd.DataFrame:
    """Return sorted portfolio weights for display."""
    return (
        portfolio.weights.sort_values(ascending=False)
        .rename("weight")
        .to_frame()
        .assign(weight_percent=lambda frame: frame["weight"].map(format_percent))
    )


@st.cache_data(show_spinner=False, ttl=3600)
def run_cached_analysis(
    vn_tickers: tuple[str, ...],
    us_tickers: tuple[str, ...],
    crypto_tickers: tuple[str, ...],
    commodity_tickers: tuple[str, ...],
    start_date: str,
    profile_name: str,
    risk_free_rate: float,
    target_return: float,
    target_tolerance: float,
    num_portfolios: int,
    random_seed: int | None,
) -> AnalysisResult:
    """Cache one analysis run for repeated UI renders."""
    analysis_config = AnalysisConfig(
        vn_tickers=list(vn_tickers),
        us_tickers=list(us_tickers),
        crypto_tickers=list(crypto_tickers),
        commodity_tickers=list(commodity_tickers),
        start_date=start_date,
        risk_budget=RISK_PROFILES[profile_name],
        selected_profile=profile_name,
        risk_free_rate=risk_free_rate,
        target_return=target_return,
        target_tolerance=target_tolerance,
        num_portfolios=num_portfolios,
        random_seed=random_seed,
    )
    return run_analysis(analysis_config)


def render_metric_row(result: AnalysisResult) -> None:
    """Render the most important selected-portfolio metrics."""
    selected = result.selected_portfolio
    distribution = result.distribution
    col_return, col_volatility, col_sharpe, col_profit = st.columns(4)

    col_return.metric("Expected return", format_percent(selected.expected_return))
    col_volatility.metric("Volatility", format_percent(selected.volatility))
    col_sharpe.metric("Sharpe ratio", f"{selected.sharpe_ratio:.3f}")
    col_profit.metric("Profit probability", format_percent(distribution.probability_profit))


def render_weight_chart(portfolio: PortfolioSelection) -> None:
    """Render selected portfolio weights as a compact bar chart."""
    weights = portfolio.weights.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = plt.cm.Set2(range(len(weights)))
    ax.barh(weights.index, weights.values, color=colors, edgecolor="#1f2937", linewidth=0.7)
    ax.set_xlabel("Portfolio weight")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.set_title("Selected Portfolio Weights")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_portfolio_summary(result: AnalysisResult) -> None:
    """Render portfolio weights and target status."""
    selected_label = "Target-return portfolio" if result.target_portfolio else "Max-Sharpe portfolio"
    st.subheader(selected_label)

    if result.target_portfolio is None:
        st.warning("No portfolio matched the target-return window. Showing the max-Sharpe portfolio.")

    col_chart, col_table = st.columns([1.15, 0.85])
    with col_chart:
        render_weight_chart(result.selected_portfolio)
    with col_table:
        weights = portfolio_weights_frame(result.selected_portfolio)
        st.dataframe(
            weights[["weight_percent"]],
            width="stretch",
            height=360,
            column_config={"weight_percent": st.column_config.TextColumn("Weight")},
        )


def render_ranking(result: AnalysisResult) -> None:
    """Render asset-level ranking table and chart."""
    chart_col, table_col = st.columns([1.15, 0.85])
    with chart_col:
        fig, _ = plot_asset_ranking(result.ranking, rf_rate=RISK_FREE_RATE)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with table_col:
        ranking = result.ranking.copy()
        ranking["annual_return"] = ranking["annual_return"].map(format_percent)
        ranking["annual_volatility"] = ranking["annual_volatility"].map(format_percent)
        ranking["sharpe_ratio"] = ranking["sharpe_ratio"].map(lambda value: f"{value:.3f}")
        st.dataframe(
            ranking,
            width="stretch",
            height=420,
            column_config={
                "annual_return": st.column_config.TextColumn("Return"),
                "annual_volatility": st.column_config.TextColumn("Volatility"),
                "sharpe_ratio": st.column_config.TextColumn("Sharpe"),
            },
        )


def render_simulation(result: AnalysisResult) -> None:
    """Render Monte Carlo frontier and run diagnostics."""
    fig, _ = plot_simulation_frontier(result.simulation, target_portfolio=result.target_portfolio)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    attempt_col, failed_col, assets_col = st.columns(3)
    attempt_col.metric("Accepted portfolios", f"{len(result.simulation.weights):,}")
    failed_col.metric("Rejected attempts", f"{result.simulation.failed_attempts:,}")
    assets_col.metric("Assets loaded", str(result.prices.shape[1]))


def render_distribution(result: AnalysisResult) -> None:
    """Render selected portfolio return distribution."""
    title = "Target Portfolio Return Distribution" if result.target_portfolio else "Max-Sharpe Return Distribution"
    fig, _ = plot_return_distribution(result.distribution, title=title)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    interval_68, interval_95 = result.distribution.interval_68, result.distribution.interval_95
    col_68, col_95, col_loss = st.columns(3)
    col_68.metric("68% interval", f"{format_percent(interval_68[0])} to {format_percent(interval_68[1])}")
    col_95.metric("95% interval", f"{format_percent(interval_95[0])} to {format_percent(interval_95[1])}")
    col_loss.metric("Loss probability", format_percent(result.distribution.probability_loss))


def configure_page() -> None:
    """Apply dashboard page settings and lightweight styling."""
    st.set_page_config(page_title=PROJECT_TITLE, layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: rgba(255, 255, 255, 0.05);
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the Streamlit dashboard."""
    configure_page()

    st.title(PROJECT_TITLE)
    st.caption("Portfolio optimization dashboard")

    with st.sidebar:
        st.image("Logo.png", use_container_width=True)
        st.header("Configuration")
        with st.form("analysis_settings"):
            vn_tickers = parse_tickers(st.text_input("Vietnam stocks", value=join_tickers(VN_TICKERS)))
            us_tickers = parse_tickers(st.text_input("US stocks", value=join_tickers(US_TICKERS)))
            crypto_tickers = parse_tickers(st.text_input("Crypto", value=join_tickers(CRYPTO_TICKERS)))
            commodity_tickers = parse_tickers(st.text_input("Commodities", value=join_tickers(COMMODITY_TICKERS)))

            selected_date = st.date_input("Start date", value=date.fromisoformat(START_DATE))
            profile_name = st.selectbox(
                "Risk profile",
                options=list(RISK_PROFILES),
                index=list(RISK_PROFILES).index(SELECTED_PROFILE),
            )
            risk_free_rate = st.number_input(
                "Risk-free rate",
                min_value=0.0,
                max_value=0.25,
                value=float(RISK_FREE_RATE),
                step=0.005,
                format="%.3f",
            )
            target_return = st.number_input(
                "Target return",
                min_value=-1.0,
                max_value=3.0,
                value=float(TARGET_RETURN),
                step=0.01,
                format="%.2f",
            )
            target_tolerance = st.number_input(
                "Target tolerance",
                min_value=0.0,
                max_value=1.0,
                value=float(TARGET_TOLERANCE),
                step=0.005,
                format="%.3f",
            )
            num_portfolios = st.slider(
                "Simulations",
                min_value=2_000,
                max_value=100_000,
                value=int(NUM_PORTFOLIOS),
                step=2_000,
            )
            submitted = st.form_submit_button("Run analysis", width="stretch")

    if not submitted and "analysis_result" not in st.session_state:
        submitted = True

    if not any([vn_tickers, us_tickers, crypto_tickers, commodity_tickers]):
        st.error("Add at least one asset before running analysis.")
        return

    if submitted:
        with st.spinner("Running portfolio analysis..."):
            st.session_state.analysis_result = run_cached_analysis(
                tuple(vn_tickers),
                tuple(us_tickers),
                tuple(crypto_tickers),
                tuple(commodity_tickers),
                selected_date.isoformat(),
                profile_name,
                risk_free_rate,
                target_return,
                target_tolerance,
                num_portfolios,
                RANDOM_SEED,
            )

    result: AnalysisResult = st.session_state.analysis_result
    st.caption(f"Price matrix: {result.prices.shape[0]:,} rows x {result.prices.shape[1]:,} assets")

    render_metric_row(result)
    st.divider()
    render_portfolio_summary(result)

    frontier_tab, ranking_tab, distribution_tab, prices_tab = st.tabs(
        ["Simulation", "Asset Ranking", "Distribution", "Prices"]
    )
    with frontier_tab:
        render_simulation(result)
    with ranking_tab:
        render_ranking(result)
    with distribution_tab:
        render_distribution(result)
    with prices_tab:
        st.dataframe(result.prices.tail(250), width="stretch", height=520)


if __name__ == "__main__":
    main()
