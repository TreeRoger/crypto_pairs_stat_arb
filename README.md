# Crypto Pairs Statistical Arbitrage (Hourly CoinGecko Data)

This repository contains a complete statistical arbitrage research framework for cryptocurrency markets. The project collects hourly OHLC data from the CoinGecko API, selects mean-reverting pairs, generates trading signals, constructs a dollar-neutral portfolio, and evaluates performance through a backtesting engine.

The goal of this work is to study whether cryptocurrency pairs exhibit stable long-term relationships and whether those relationships can be exploited through a systematic spread-trading strategy.

---

## Project Overview

The system is organized into four stages:

### 1. Pair Selection
The engine evaluates all candidate asset pairs and filters them using:

- Ordinary Least Squares hedge ratios  
- Augmented Dickey–Fuller tests for stationarity  
- Estimated half-life of mean reversion  
- Variance ratio screens  
- Return correlation thresholds  
- Per-asset exposure limits  

Only pairs that pass all statistical and structural checks are admitted into the trading universe.

### 2. Signal Generation
For each selected pair, the framework computes:

- Rolling hedge ratios (alpha and beta)  
- Rolling spreads  
- Rolling z-scores  
- Mean-reversion entry and exit signals  

Signals are created in a format that can be directly consumed by the portfolio construction module.

### 3. Portfolio Construction
The portfolio allocates capital across pairs in a dollar-neutral manner. It includes:

- Risk caps per pair  
- Global gross exposure caps  
- Long and short spread positioning  
- A state-machine approach to entering and exiting trades  

The resulting weight matrix is designed to be realistic and stable over long backtests.

### 4. Backtesting
The backtest engine computes:

- Hourly portfolio returns  
- Turnover and transaction cost effects  
- Net returns after costs  
- Sharpe ratio  
- Maximum drawdown  
- Information ratio relative to a benchmark  
- Estimated average holding period  

This produces a transparent and replicable performance assessment.

---

## Directory Structure

crypto_pairs_stat_arb/
│
├── src/
│ ├── run_backtest.py
│ ├── pairs.py
│ ├── signals.py
│ ├── portfolio.py
│ ├── backtest.py
│ ├── metrics.py
│ └── data/
│ ├── coingecko_client.py
│ ├── download_hourly.py
│ ├── load_hourly_prices.py
│ └── init.py
│
├── data/
│ ├── hourly_raw/ (ignored in git)
│ └── prices/ (ignored in git)
│
├── requirements.txt
├── .gitignore
└── README.md


---

## Installation

Install all required packages:

```bash
pip install -r requirements.txt
