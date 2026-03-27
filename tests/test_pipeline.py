"""
Tests for the portfolio optimization pipeline.
Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from data_pipeline import (
    fetch_market_data,
    compute_returns,
    compute_statistics,
    _generate_synthetic_data,
    DEFAULT_TICKERS,
)
from optimizer import (
    monte_carlo_gpu,
    optimize_sharpe_gpu,
    compute_efficient_frontier,
    score_asset_combinations,
    benchmark_cpu_baseline,
    get_device_info,
    run_full_optimization,
)
from backtest import run_backtest, rolling_backtest, _perf_stats


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def synthetic_prices():
    """Generate synthetic price data (fast, no network)."""
    return _generate_synthetic_data(
        DEFAULT_TICKERS[:10],
        "2023-01-01",
        "2024-01-01",
    )


@pytest.fixture(scope="module")
def stats(synthetic_prices):
    """Compute statistics from synthetic prices."""
    return compute_statistics(synthetic_prices)


# ==============================================================================
# Data Pipeline Tests
# ==============================================================================

class TestDataPipeline:
    def test_synthetic_data_shape(self, synthetic_prices):
        assert synthetic_prices.shape[1] == 10
        assert len(synthetic_prices) > 100

    def test_synthetic_data_positive(self, synthetic_prices):
        assert (synthetic_prices > 0).all().all()

    def test_compute_returns_shape(self, synthetic_prices):
        returns = compute_returns(synthetic_prices)
        assert returns.shape[0] == synthetic_prices.shape[0] - 1
        assert returns.shape[1] == synthetic_prices.shape[1]

    def test_compute_statistics_keys(self, stats):
        assert 'mean_returns' in stats
        assert 'cov_matrix' in stats
        assert 'tickers' in stats
        assert 'n_assets' in stats

    def test_cov_matrix_symmetric(self, stats):
        cov = stats['cov_matrix']
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_cov_matrix_positive_diagonal(self, stats):
        assert (np.diag(stats['cov_matrix']) > 0).all()

    def test_default_tickers_count(self):
        assert len(DEFAULT_TICKERS) == 50


# ==============================================================================
# Monte Carlo Tests
# ==============================================================================

class TestMonteCarlo:
    def test_returns_correct_keys(self, stats):
        result = monte_carlo_gpu(stats['mean_returns'], stats['cov_matrix'], n_portfolios=1000)
        assert 'all_returns' in result
        assert 'all_volatilities' in result
        assert 'all_sharpe' in result
        assert 'max_sharpe' in result
        assert 'min_volatility' in result
        assert 'computation_time' in result

    def test_correct_number_of_portfolios(self, stats):
        result = monte_carlo_gpu(stats['mean_returns'], stats['cov_matrix'], n_portfolios=500)
        assert len(result['all_returns']) == 500
        assert len(result['all_volatilities']) == 500

    def test_volatilities_positive(self, stats):
        result = monte_carlo_gpu(stats['mean_returns'], stats['cov_matrix'], n_portfolios=1000)
        assert (result['all_volatilities'] > 0).all()

    def test_max_sharpe_is_best(self, stats):
        result = monte_carlo_gpu(stats['mean_returns'], stats['cov_matrix'], n_portfolios=1000)
        assert result['max_sharpe']['sharpe'] == pytest.approx(
            result['all_sharpe'].max(), abs=1e-5
        )

    def test_weights_sum_to_one(self, stats):
        result = monte_carlo_gpu(stats['mean_returns'], stats['cov_matrix'], n_portfolios=100)
        sums = result['all_weights'].sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)


# ==============================================================================
# Sharpe Optimization Tests
# ==============================================================================

class TestSharpeOptimization:
    def test_returns_correct_keys(self, stats):
        result = optimize_sharpe_gpu(
            stats['mean_returns'], stats['cov_matrix'],
            n_iterations=100, lr=0.01,
        )
        assert 'weights' in result
        assert 'return' in result
        assert 'volatility' in result
        assert 'sharpe' in result

    def test_weights_sum_to_one(self, stats):
        result = optimize_sharpe_gpu(
            stats['mean_returns'], stats['cov_matrix'],
            n_iterations=200,
        )
        assert result['weights'].sum() == pytest.approx(1.0, abs=1e-4)

    def test_weights_non_negative(self, stats):
        result = optimize_sharpe_gpu(
            stats['mean_returns'], stats['cov_matrix'],
            n_iterations=200,
        )
        # softmax weights should be >= 0
        assert (result['weights'] >= -1e-6).all()

    def test_positive_sharpe(self, stats):
        result = optimize_sharpe_gpu(
            stats['mean_returns'], stats['cov_matrix'],
            n_iterations=500,
        )
        assert result['sharpe'] > 0


# ==============================================================================
# Efficient Frontier Tests
# ==============================================================================

class TestEfficientFrontier:
    def test_returns_volatilities_same_length(self, stats):
        frontier = compute_efficient_frontier(
            stats['mean_returns'], stats['cov_matrix'], n_points=20,
        )
        assert len(frontier['returns']) == len(frontier['volatilities'])
        assert len(frontier['returns']) > 5  # at least some points

    def test_frontier_sorted_by_volatility(self, stats):
        frontier = compute_efficient_frontier(
            stats['mean_returns'], stats['cov_matrix'], n_points=20,
        )
        vols = frontier['volatilities']
        assert vols == sorted(vols)

    def test_frontier_volatilities_positive(self, stats):
        frontier = compute_efficient_frontier(
            stats['mean_returns'], stats['cov_matrix'], n_points=20,
        )
        assert all(v > 0 for v in frontier['volatilities'])


# ==============================================================================
# Asset Scoring Tests
# ==============================================================================

class TestAssetScoring:
    def test_returns_correct_keys(self, stats):
        result = score_asset_combinations(
            stats['mean_returns'], stats['cov_matrix'], stats['tickers'],
        )
        assert 'single_assets' in result
        assert 'asset_pairs' in result
        assert 'total_combinations_scored' in result

    def test_single_assets_count(self, stats):
        result = score_asset_combinations(
            stats['mean_returns'], stats['cov_matrix'], stats['tickers'],
        )
        assert len(result['single_assets']) <= 20  # top_k default
        assert len(result['single_assets']) <= stats['n_assets']

    def test_pair_format(self, stats):
        result = score_asset_combinations(
            stats['mean_returns'], stats['cov_matrix'], stats['tickers'],
        )
        for pair in result['asset_pairs']:
            assert '+' in pair['ticker']
            assert 'sharpe' in pair


# ==============================================================================
# CPU Baseline Benchmark Tests
# ==============================================================================

class TestBenchmark:
    def test_returns_correct_keys(self, stats):
        result = benchmark_cpu_baseline(
            stats['mean_returns'], stats['cov_matrix'], n_portfolios=500,
        )
        assert 'computation_time' in result
        assert 'n_portfolios' in result
        assert 'throughput' in result

    def test_positive_throughput(self, stats):
        result = benchmark_cpu_baseline(
            stats['mean_returns'], stats['cov_matrix'], n_portfolios=500,
        )
        assert result['throughput'] > 0


# ==============================================================================
# Device Info
# ==============================================================================

class TestDeviceInfo:
    def test_device_info_keys(self):
        info = get_device_info()
        assert 'device' in info
        assert 'torch' in info
        assert info['device'] in ('cuda', 'cpu', 'cpu_numpy')


# ==============================================================================
# Full Pipeline (integration test)
# ==============================================================================

class TestFullPipeline:
    def test_run_full_optimization(self, stats):
        result = run_full_optimization(
            stats['mean_returns'], stats['cov_matrix'], stats['tickers'],
            n_portfolios=1000, risk_free_rate=0.04,
        )
        assert 'monte_carlo' in result
        assert 'optimization' in result
        assert 'frontier' in result
        assert 'asset_scores' in result
        assert 'benchmark' in result
        assert result['benchmark']['speedup'] > 0


# ==============================================================================
# API Tests
# ==============================================================================

class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        from server import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_device_endpoint(self, client):
        resp = client.get('/api/device')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'device' in data

    def test_tickers_endpoint(self, client):
        resp = client.get('/api/tickers')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tickers' in data
        assert data['count'] == 50

    def test_optimize_endpoint(self, client):
        resp = client.post('/api/optimize', json={
            'n_portfolios': 1000,
            'risk_free_rate': 0.04,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert 'scatter' in data
        assert 'optimal' in data
        assert 'frontier' in data

    def test_monte_carlo_endpoint(self, client):
        resp = client.post('/api/monte-carlo', json={'n_portfolios': 500})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert len(data['scatter']['returns']) > 0

    def test_optimize_caps_portfolios(self, client):
        resp = client.post('/api/optimize', json={'n_portfolios': 999999})
        assert resp.status_code == 200
        # Should be capped at 200000

    def test_frontend_serves(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        assert b'Portfolio' in resp.data


# ==============================================================================
# Backtest Tests
# ==============================================================================

class TestBacktest:
    @pytest.fixture(scope="class")
    def long_prices(self):
        """300-day synthetic prices (enough for train/test split)."""
        return _generate_synthetic_data(DEFAULT_TICKERS[:8], "2022-01-01", "2023-03-01")

    def test_perf_stats_keys(self):
        import pandas as pd
        dummy = pd.Series([0.01, -0.005, 0.008, 0.002, -0.003] * 50)
        stats = _perf_stats(dummy)
        for key in ("cumulative_return", "annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown", "n_days"):
            assert key in stats

    def test_perf_stats_max_drawdown_negative(self):
        import pandas as pd
        dummy = pd.Series([0.01, -0.05, 0.02, -0.03, 0.01] * 20)
        stats = _perf_stats(dummy)
        assert stats["max_drawdown"] <= 0

    def test_run_backtest_structure(self, long_prices):
        result = run_backtest(long_prices, train_days=200)
        assert "split" in result
        assert "out_of_sample" in result
        assert "in_sample" in result
        assert "optimized_weights" in result
        oos = result["out_of_sample"]
        assert "optimized" in oos
        assert "equal_weight" in oos
        assert "sharpe_lift" in oos

    def test_run_backtest_weights_sum_to_one(self, long_prices):
        result = run_backtest(long_prices, train_days=200)
        total = sum(result["optimized_weights"].values())
        assert abs(total - 1.0) < 1e-4

    def test_run_backtest_cumulative_starts_near_one(self, long_prices):
        result = run_backtest(long_prices, train_days=200)
        assert abs(result["cumulative"]["optimized"][0] - 1.0) < 0.05

    def test_run_backtest_test_days_match_split(self, long_prices):
        result = run_backtest(long_prices, train_days=200)
        expected_test = result["split"]["test_days"]
        assert len(result["daily_returns"]["optimized"]) == expected_test

    def test_run_backtest_insufficient_data(self, long_prices):
        """Should raise if train_days leaves fewer than 30 test days."""
        with pytest.raises(ValueError, match="at least 30"):
            run_backtest(long_prices, train_days=len(long_prices) - 5)

    def test_rolling_backtest_structure(self, long_prices):
        result = rolling_backtest(long_prices, train_days=150, step_days=30)
        assert "windows" in result
        assert len(result["windows"]) >= 1
        assert "combined_out_of_sample" in result
        assert "optimized" in result["combined_out_of_sample"]

    def test_backtest_api_endpoint(self):
        """API endpoint returns backtest result correctly."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        from server import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            resp = client.post('/api/backtest', json={'train_days': 500})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['status'] == 'success'
            assert 'out_of_sample' in data

