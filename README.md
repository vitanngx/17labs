# 17 LABS

17 LABS is a quantitative portfolio optimization prototype that combines multi-asset market data, ranking, Monte Carlo simulation, and probability analysis into a simple Python workflow. The project is designed as a clean, GitHub-ready refactor of an exploratory notebook and can be used as an educational decision-support tool for experimenting with cross-asset portfolio construction.

## Main Features

- Multi-asset data ingestion for Vietnamese equities, international assets, crypto, and commodities
- Annualized asset ranking using return, volatility, and Sharpe ratio
- Markowitz-style Monte Carlo portfolio simulation
- Risk-profile budget allocation across asset groups
- Asset-level weight constraints
- Target-return portfolio filtering
- Probability and distribution analysis for selected portfolios
- Simple Matplotlib visualizations for ranking, portfolio clouds, and return distributions

## Tech Stack

- Python
- pandas
- numpy
- scipy
- matplotlib
- yfinance
- vnstock
- python-dotenv

## Project Structure

```text
17-labs/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── main.py
├── config.py
├── data/
│   └── .gitkeep
├── notebooks/
│   └── 17labs_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── asset_ranking.py
│   ├── optimization.py
│   ├── target_portfolio.py
│   ├── risk_analysis.py
│   ├── visualization.py
│   └── utils.py
└── images/
    └── .gitkeep
```

## Setup

1. Create and activate a virtual environment.
2. Install the project dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example environment file and add your own credentials:

```bash
cp .env.example .env
```

## Environment Variables

The project does not keep secrets in source control. Add the following variable to `.env`:

```bash
VNSTOCK_API_KEY=your_vnstock_api_key_here
```

## Example Usage

Run the default workflow:

```bash
python main.py
```

Edit `config.py` to change the asset universe, risk profile, risk-free rate, simulation count, or target-return settings.

## Quantitative Logic Overview

The workflow first downloads close-price data and aligns it into a clean price matrix. It then converts prices into daily returns, annualizes return and volatility, and ranks individual assets by Sharpe ratio. Portfolio construction uses Monte Carlo simulation with group-level risk budgets and asset-level min/max constraints, then selects the highest-Sharpe portfolio under those rules. A second filter searches for portfolios near a target annual return and keeps the lowest-volatility candidate. The selected portfolio is finally analyzed under a normal-distribution assumption to estimate profit probability, loss probability, and one-year confidence bands.

## Disclaimer

17 LABS is an educational and decision-support prototype. It is not investment advice, not a production trading system, and not a guarantee of future performance.

