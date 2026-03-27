"""
Walk-forward backtesting for the portfolio optimizer.

Splits historical price data into a training window and an out-of-sample
test window, optimizes on the training window, then evaluates on the test
window.  Also computes an equal-weight and SPY/benchmark comparison.
"""

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    return prices_df.pct_change().dropna()


def _portfolio_daily_returns(weights: np.ndarray, daily_ret: pd.DataFrame) -> pd.Series:
    """Apply fixed weights to a returns dataframe → portfolio daily return series."""
    return (daily_ret * weights).sum(axis=1)


def _perf_stats(daily_series: pd.Series, rf_daily: float = 0.04 / 252) -> dict:
    """Annualised performance stats from a daily return series."""
    cum = (1 + daily_series).cumprod()
    n = len(daily_series)
    annual_return = float(cum.iloc[-1] ** (252 / n) - 1)
    annual_vol = float(daily_series.std() * np.sqrt(252))
    sharpe = float((annual_return - 0.04) / annual_vol) if annual_vol > 0 else 0.0

    # maximum drawdown
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    cumulative_return = float(cum.iloc[-1] - 1)

    return {
        "cumulative_return": round(cumulative_return, 4),
        "annual_return": round(annual_return, 4),
        "annual_volatility": round(annual_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 4),
        "n_days": n,
    }


# ── main backtest function ────────────────────────────────────────────────────

def run_backtest(
    prices_df: pd.DataFrame,
    train_days: int = 500,
    risk_free: float = 0.04,
    n_iterations: int = 1000,
) -> dict:
    """
    Walk-forward backtest.

    Parameters
    ----------
    prices_df:    DataFrame with dates as index, tickers as columns.
    train_days:   Number of trading days used for optimisation.
    risk_free:    Annual risk-free rate (default 4 %).
    n_iterations: Gradient optimisation iterations on training window.

    Returns
    -------
    dict with keys: train_stats, optimized, equal_weight, dates, daily_returns
    """
    # Import here to avoid circular imports
    from data_pipeline import compute_statistics
    from optimizer import optimize_sharpe_gpu

    total_days = len(prices_df)
    test_days = total_days - train_days
    if test_days < 30:
        raise ValueError(
            f"Need at least 30 test days; got {test_days} "
            f"(total={total_days}, train={train_days})"
        )

    # ── split ──────────────────────────────────────────────────────
    train_prices = prices_df.iloc[:train_days]
    test_prices = prices_df.iloc[train_days - 1:]  # overlap 1 day for return continuity

    # ── compute statistics on TRAINING window only ─────────────────
    train_stats = compute_statistics(train_prices)

    # ── optimise on TRAINING window ────────────────────────────────
    opt_result = optimize_sharpe_gpu(
        train_stats["mean_returns"],
        train_stats["cov_matrix"],
        n_iterations=n_iterations,
        risk_free_rate=risk_free,
    )
    opt_weights = np.array(opt_result["weights"])
    tickers = train_stats["tickers"]
    n_assets = len(tickers)

    # ── evaluate on TEST window ────────────────────────────────────
    test_ret = _daily_returns(test_prices)

    # Optimised portfolio
    opt_daily = _portfolio_daily_returns(opt_weights, test_ret)

    # Equal-weight portfolio
    ew_weights = np.ones(n_assets) / n_assets
    ew_daily = _portfolio_daily_returns(ew_weights, test_ret)

    # In-sample optimised stats (for reference / overfitting comparison)
    train_ret = _daily_returns(train_prices)
    opt_daily_train = _portfolio_daily_returns(opt_weights, train_ret)

    optimized_test = _perf_stats(opt_daily)
    optimized_train = _perf_stats(opt_daily_train)
    equal_weight_test = _perf_stats(ew_daily)

    # Top holdings
    top_holdings = sorted(
        zip(tickers, opt_weights.tolist()), key=lambda x: -x[1]
    )[:5]

    return {
        "split": {
            "train_days": train_days,
            "test_days": len(test_ret),
            "train_start": str(train_prices.index[0].date()),
            "train_end": str(train_prices.index[-1].date()),
            "test_start": str(test_prices.index[1].date()),
            "test_end": str(test_prices.index[-1].date()),
        },
        "optimized_weights": {t: round(w, 4) for t, w in zip(tickers, opt_weights.tolist())},
        "top_holdings": top_holdings,
        "in_sample": optimized_train,
        "out_of_sample": {
            "optimized": optimized_test,
            "equal_weight": equal_weight_test,
            "sharpe_lift": round(
                optimized_test["sharpe_ratio"] - equal_weight_test["sharpe_ratio"], 4
            ),
        },
        # Daily series as lists for the API / notebook
        "daily_returns": {
            "dates": [str(d.date()) for d in test_ret.index],
            "optimized": opt_daily.tolist(),
            "equal_weight": ew_daily.tolist(),
        },
        # Cumulative return curves
        "cumulative": {
            "dates": [str(d.date()) for d in test_ret.index],
            "optimized": (1 + opt_daily).cumprod().tolist(),
            "equal_weight": (1 + ew_daily).cumprod().tolist(),
        },
    }


# ── rolling walk-forward (multiple windows) ──────────────────────────────────

def rolling_backtest(
    prices_df: pd.DataFrame,
    train_days: int = 500,
    step_days: int = 63,   # rebalance ~quarterly
    n_iterations: int = 500,
    risk_free: float = 0.04,
) -> dict:
    """
    Rolling walk-forward: re-optimise every `step_days` trading days,
    evaluate on the next window of `step_days` days.

    Returns a combined out-of-sample equity curve covering the full test period.
    """
    from data_pipeline import compute_statistics
    from optimizer import optimize_sharpe_gpu

    total_days = len(prices_df)
    all_opt_daily: list[float] = []
    all_ew_daily: list[float] = []
    all_dates: list[str] = []
    windows: list[dict] = []

    start = 0
    while start + train_days + step_days <= total_days:
        train_prices = prices_df.iloc[start : start + train_days]
        test_prices = prices_df.iloc[start + train_days - 1 : start + train_days + step_days]

        train_stats = compute_statistics(train_prices)
        opt_result = optimize_sharpe_gpu(
            train_stats["mean_returns"],
            train_stats["cov_matrix"],
            n_iterations=n_iterations,
            risk_free_rate=risk_free,
        )
        opt_weights = np.array(opt_result["weights"])
        n_assets = len(train_stats["tickers"])
        ew_weights = np.ones(n_assets) / n_assets

        test_ret = _daily_returns(test_prices)
        opt_daily = _portfolio_daily_returns(opt_weights, test_ret)
        ew_daily = _portfolio_daily_returns(ew_weights, test_ret)

        all_opt_daily.extend(opt_daily.tolist())
        all_ew_daily.extend(ew_daily.tolist())
        all_dates.extend([str(d.date()) for d in test_ret.index])
        windows.append({
            "train_start": str(train_prices.index[0].date()),
            "train_end": str(train_prices.index[-1].date()),
            "test_start": str(test_ret.index[0].date()),
            "test_end": str(test_ret.index[-1].date()),
            "opt_sharpe_train": round(opt_result["sharpe"], 4),
        })

        start += step_days

    opt_series = pd.Series(all_opt_daily)
    ew_series = pd.Series(all_ew_daily)

    return {
        "windows": windows,
        "combined_out_of_sample": {
            "optimized": _perf_stats(opt_series),
            "equal_weight": _perf_stats(ew_series),
        },
        "cumulative": {
            "dates": all_dates,
            "optimized": (1 + opt_series).cumprod().tolist(),
            "equal_weight": (1 + ew_series).cumprod().tolist(),
        },
    }
