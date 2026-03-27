"""
Data pipeline for fetching and processing market data.
Uses yfinance for historical price data, with CSV fallback for demo mode.
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Top 50 S&P 500 tickers by market cap for demo
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
    "ACN", "TMO", "ABT", "DHR", "CRM", "NKE", "TXN", "NEE", "PM",
    "UNP", "RTX", "HON", "LOW", "INTC", "QCOM", "AMGN", "IBM", "CAT",
    "GS", "BA", "SBUX", "BLK", "AMD"
]


def fetch_market_data(tickers=None, start_date=None, end_date=None, use_cache=True):
    """
    Fetch historical adjusted close prices.
    Returns: DataFrame of adjusted close prices, or generates synthetic data if yfinance fails.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

    cache_path = os.path.join(DATA_DIR, 'price_cache.csv')
    meta_path = os.path.join(DATA_DIR, 'cache_meta.json')

    # Try cache first
    if use_cache and os.path.exists(cache_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if set(meta.get('tickers', [])) == set(tickers):
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                print(f"[DATA] Loaded {len(df.columns)} tickers from cache ({len(df)} days)")
                return df
        except Exception:
            pass

    # Try yfinance
    try:
        import yfinance as yf
        print(f"[DATA] Downloading {len(tickers)} tickers from Yahoo Finance...")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)

        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        else:
            prices = data

        prices = prices.dropna(axis=1, how='all').dropna()

        if len(prices.columns) < 5:
            raise ValueError("Too few tickers returned")

        # Cache it
        os.makedirs(DATA_DIR, exist_ok=True)
        prices.to_csv(cache_path)
        with open(meta_path, 'w') as f:
            json.dump({'tickers': list(prices.columns), 'date': datetime.now().isoformat()}, f)

        print(f"[DATA] Downloaded {len(prices.columns)} tickers, {len(prices)} trading days")
        return prices

    except Exception as e:
        print(f"[DATA] yfinance failed ({e}), generating synthetic market data...")
        return _generate_synthetic_data(tickers, start_date, end_date)


def _generate_synthetic_data(tickers, start_date, end_date):
    """Generate realistic synthetic stock price data for demo/testing."""
    np.random.seed(42)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    n_assets = len(tickers)

    # Sector-based correlation structure
    sector_map = {
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
                 'CRM', 'CSCO', 'TXN', 'INTC', 'QCOM', 'IBM', 'AMD', 'ACN'],
        'health': ['UNH', 'JNJ', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'DHR', 'AMGN'],
        'finance': ['JPM', 'V', 'MA', 'BRK-B', 'GS', 'BLK'],
        'consumer': ['PG', 'PEP', 'KO', 'COST', 'WMT', 'MCD', 'NKE', 'SBUX', 'HD', 'LOW'],
        'industrial': ['XOM', 'CVX', 'HON', 'CAT', 'BA', 'RTX', 'UNP', 'NEE', 'PM'],
    }

    # Annual returns and vols by sector
    sector_params = {
        'tech': (0.15, 0.28), 'health': (0.10, 0.20), 'finance': (0.11, 0.22),
        'consumer': (0.08, 0.16), 'industrial': (0.09, 0.19),
    }

    prices_data = {}
    for ticker in tickers:
        sector = 'tech'  # default
        for s, members in sector_map.items():
            if ticker in members:
                sector = s
                break

        annual_ret, annual_vol = sector_params[sector]
        daily_ret = annual_ret / 252
        daily_vol = annual_vol / np.sqrt(252)

        # GBM with mean reversion
        log_returns = np.random.normal(daily_ret, daily_vol, n_days)
        # Add market factor
        market_noise = np.random.normal(0, daily_vol * 0.3, n_days)
        log_returns += market_noise

        prices = 100 * np.exp(np.cumsum(log_returns))
        prices_data[ticker] = prices

    df = pd.DataFrame(prices_data, index=dates[:n_days])
    print(f"[DATA] Generated synthetic data: {len(df.columns)} tickers, {len(df)} days")
    return df


def compute_returns(prices):
    """Compute daily log returns from price data."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_statistics(prices):
    """Compute return statistics needed for optimization."""
    returns = compute_returns(prices)
    mean_returns = returns.mean().values  # daily mean returns
    cov_matrix = returns.cov().values     # daily covariance matrix
    tickers = list(returns.columns)

    return {
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix,
        'tickers': tickers,
        'n_assets': len(tickers),
        'annual_factor': 252,
        'returns_df': returns,
    }


if __name__ == '__main__':
    prices = fetch_market_data()
    stats = compute_statistics(prices)
    print(f"\nAssets: {stats['n_assets']}")
    print(f"Cov matrix shape: {stats['cov_matrix'].shape}")
    print(f"Annualized returns range: [{stats['mean_returns'].min()*252:.2%}, {stats['mean_returns'].max()*252:.2%}]")
