#!/usr/bin/env python3
"""
Standalone benchmark: GPU vs CPU portfolio optimization.
Run this to get performance numbers for your resume.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from data_pipeline import fetch_market_data, compute_statistics
from optimizer import (
    monte_carlo_gpu, benchmark_cpu_baseline, optimize_sharpe_gpu,
    score_asset_combinations, get_device_info
)
import time

def main():
    print("=" * 65)
    print("  Portfolio Optimization Benchmark")
    print(f"  Device: {get_device_info()}")
    print("=" * 65)

    # Load data
    print("\n[1] Loading market data...")
    prices = fetch_market_data()
    stats = compute_statistics(prices)
    print(f"    {stats['n_assets']} assets loaded")

    # Benchmark configurations
    configs = [10000, 50000, 100000]

    print(f"\n{'Portfolios':>12} | {'GPU/Vec Time':>12} | {'CPU Time':>12} | {'Speedup':>8} | {'GPU Throughput':>15}")
    print("-" * 72)

    for n in configs:
        # GPU/vectorized
        gpu_result = monte_carlo_gpu(stats['mean_returns'], stats['cov_matrix'], n)
        gpu_time = gpu_result['computation_time']
        gpu_tp = n / gpu_time

        # CPU sequential
        cpu_result = benchmark_cpu_baseline(stats['mean_returns'], stats['cov_matrix'], n)
        cpu_time = cpu_result['computation_time']
        cpu_tp = cpu_result['throughput']

        speedup = gpu_tp / cpu_tp

        print(f"{n:>12,} | {gpu_time:>11.4f}s | {cpu_time:>11.4f}s | {speedup:>7.1f}x | {gpu_tp:>12,.0f}/sec")

    # Gradient optimization benchmark
    print(f"\n{'='*65}")
    print("  Gradient-Based Sharpe Optimization")
    print(f"{'='*65}\n")

    opt = optimize_sharpe_gpu(stats['mean_returns'], stats['cov_matrix'])
    print(f"  Optimal Sharpe: {opt['sharpe']:.4f}")
    print(f"  Expected Return: {opt['return']*100:.1f}%")
    print(f"  Volatility: {opt['volatility']*100:.1f}%")
    print(f"  Time: {opt['computation_time']:.4f}s")
    print(f"  Device: {opt['device']}")

    # Asset scoring benchmark
    print(f"\n{'='*65}")
    print("  Asset Combination Scoring")
    print(f"{'='*65}\n")

    t0 = time.time()
    scores = score_asset_combinations(stats['mean_returns'], stats['cov_matrix'], stats['tickers'])
    t1 = time.time()
    print(f"  Combinations scored: {scores['total_combinations_scored']:,}")
    print(f"  Time: {t1-t0:.4f}s")
    print(f"  Throughput: {scores['total_combinations_scored']/(t1-t0):,.0f} combinations/sec")
    print(f"\n  Top asset: {scores['single_assets'][0]['ticker']} (Sharpe: {scores['single_assets'][0]['sharpe']:.3f})")
    print(f"  Top pair: {scores['asset_pairs'][0]['ticker']} (Sharpe: {scores['asset_pairs'][0]['sharpe']:.3f})")

    print(f"\n{'='*65}")
    print("  Benchmark Complete!")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
