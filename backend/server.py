"""
Flask API server for the Portfolio Optimization Engine.
Serves the frontend and provides REST endpoints for optimization.
"""

import os
import sys
import json
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import fetch_market_data, compute_statistics, DEFAULT_TICKERS
from optimizer import (
    run_full_optimization,
    monte_carlo_gpu,
    optimize_sharpe_gpu,
    compute_efficient_frontier,
    score_asset_combinations,
    get_device_info,
)
from backtest import run_backtest, rolling_backtest

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='')
CORS(app)

# Global cache for computed data
_cache = {}


def _get_stats(tickers=None):
    """Get or compute market statistics (cached)."""
    cache_key = str(sorted(tickers)) if tickers else 'default'
    if cache_key not in _cache:
        prices = fetch_market_data(tickers=tickers)
        stats = compute_statistics(prices)
        _cache[cache_key] = stats
    return _cache[cache_key]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


app.json_encoder = NumpyEncoder


def _jsonify_numpy(data):
    """Convert a dict with numpy arrays to JSON-safe dict."""
    return json.loads(json.dumps(data, cls=NumpyEncoder))


# ==============================================================================
# ROUTES
# ==============================================================================

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/device')
def device_info():
    """Get compute device information."""
    return jsonify(get_device_info())


@app.route('/api/tickers')
def available_tickers():
    """Get list of available tickers."""
    return jsonify({'tickers': DEFAULT_TICKERS, 'count': len(DEFAULT_TICKERS)})


@app.route('/api/optimize', methods=['POST'])
def optimize():
    """
    Run full optimization pipeline.
    Body: { tickers: [...], n_portfolios: 50000, risk_free_rate: 0.04 }
    """
    data = request.json or {}
    tickers = data.get('tickers', None)
    n_portfolios = min(data.get('n_portfolios', 50000), 200000)
    risk_free_rate = data.get('risk_free_rate', 0.04)

    try:
        stats = _get_stats(tickers)
        results = run_full_optimization(
            stats['mean_returns'], stats['cov_matrix'], stats['tickers'],
            n_portfolios=n_portfolios, risk_free_rate=risk_free_rate
        )

        # Subsample scatter data for frontend (max 5000 points)
        n_scatter = min(5000, len(results['monte_carlo']['all_returns']))
        indices = np.random.choice(len(results['monte_carlo']['all_returns']), n_scatter, replace=False)

        response = {
            'status': 'success',
            'scatter': {
                'returns': results['monte_carlo']['all_returns'][indices].tolist(),
                'volatilities': results['monte_carlo']['all_volatilities'][indices].tolist(),
                'sharpe': results['monte_carlo']['all_sharpe'][indices].tolist(),
            },
            'max_sharpe': _jsonify_numpy(results['monte_carlo']['max_sharpe']),
            'min_volatility': _jsonify_numpy(results['monte_carlo']['min_volatility']),
            'optimal': _jsonify_numpy({
                'weights': results['optimization']['weights'],
                'return': results['optimization']['return'],
                'volatility': results['optimization']['volatility'],
                'sharpe': results['optimization']['sharpe'],
            }),
            'frontier': _jsonify_numpy(results['frontier']),
            'asset_scores': _jsonify_numpy(results['asset_scores']),
            'benchmark': _jsonify_numpy(results['benchmark']),
            'device_info': results['device_info'],
            'tickers': stats['tickers'],
            'optimization_history': _jsonify_numpy(results['optimization'].get('history', [])),
        }

        # Add top holdings for the optimal portfolio
        opt_weights = results['optimization']['weights']
        holdings = sorted(
            [{'ticker': t, 'weight': float(w)} for t, w in zip(stats['tickers'], opt_weights)],
            key=lambda x: x['weight'], reverse=True
        )
        response['top_holdings'] = [h for h in holdings if h['weight'] > 0.01]

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/monte-carlo', methods=['POST'])
def run_monte_carlo():
    """Run Monte Carlo simulation only."""
    data = request.json or {}
    n_portfolios = min(data.get('n_portfolios', 50000), 200000)

    stats = _get_stats()
    results = monte_carlo_gpu(stats['mean_returns'], stats['cov_matrix'], n_portfolios)

    n_scatter = min(5000, len(results['all_returns']))
    indices = np.random.choice(len(results['all_returns']), n_scatter, replace=False)

    return jsonify(_jsonify_numpy({
        'status': 'success',
        'scatter': {
            'returns': results['all_returns'][indices],
            'volatilities': results['all_volatilities'][indices],
            'sharpe': results['all_sharpe'][indices],
        },
        'max_sharpe': results['max_sharpe'],
        'min_volatility': results['min_volatility'],
        'computation_time': results['computation_time'],
        'n_portfolios': results['n_portfolios'],
        'device': results['device'],
    }))


@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """Run GPU vs CPU benchmark."""
    from optimizer import benchmark_cpu_baseline
    data = request.json or {}
    n_portfolios = min(data.get('n_portfolios', 50000), 200000)

    stats = _get_stats()

    # GPU/vectorized run
    gpu_results = monte_carlo_gpu(stats['mean_returns'], stats['cov_matrix'], n_portfolios)

    # CPU sequential baseline
    cpu_results = benchmark_cpu_baseline(stats['mean_returns'], stats['cov_matrix'], n_portfolios)

    gpu_throughput = gpu_results['n_portfolios'] / gpu_results['computation_time']
    cpu_throughput = cpu_results['throughput']

    return jsonify({
        'status': 'success',
        'gpu': {
            'time': gpu_results['computation_time'],
            'throughput': gpu_throughput,
            'device': gpu_results['device'],
            'n_portfolios': gpu_results['n_portfolios'],
        },
        'cpu': {
            'time': cpu_results['computation_time'],
            'throughput': cpu_throughput,
            'device': cpu_results['device'],
            'n_portfolios': cpu_results['n_portfolios'],
        },
        'speedup': gpu_throughput / cpu_throughput if cpu_throughput > 0 else 1.0,
    })


@app.route('/api/backtest', methods=['POST'])
def backtest():
    """
    Out-of-sample walk-forward backtest.
    Body: { train_days: 500, tickers: [...], rolling: false }
    train_days: number of days for optimisation window (default 500).
    rolling: if true, run rolling walk-forward instead of single split.
    """
    data = request.json or {}
    tickers = data.get('tickers', None)
    train_days = int(data.get('train_days', 500))
    use_rolling = bool(data.get('rolling', False))

    try:
        prices = fetch_market_data(tickers=tickers)
        if use_rolling:
            result = rolling_backtest(prices, train_days=train_days)
        else:
            result = run_backtest(prices, train_days=train_days)
        return jsonify({'status': 'success', **result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  GPU-Accelerated Portfolio Optimization Engine")
    print(f"  Device: {get_device_info()}")
    print("="*60 + "\n")

    # Pre-load data
    print("Pre-loading market data...")
    _get_stats()
    print("Ready!\n")

    app.run(host='0.0.0.0', port=5001, debug=True)
