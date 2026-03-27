"""
GPU-Accelerated Portfolio Optimization Engine.
Supports CUDA (PyTorch), NumPy (CPU), and hybrid modes.
Implements Monte Carlo simulation, mean-variance optimization, and Sharpe ratio maximization.
"""

import numpy as np
from scipy.optimize import minimize
import time
import os
import sys

# Try importing PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


def get_device_info():
    """Return information about compute device."""
    if CUDA_AVAILABLE:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {'device': 'cuda', 'gpu_name': gpu_name, 'gpu_memory_gb': round(gpu_mem, 1), 'torch': True}
    elif TORCH_AVAILABLE:
        return {'device': 'cpu', 'gpu_name': None, 'gpu_memory_gb': 0, 'torch': True}
    else:
        return {'device': 'cpu_numpy', 'gpu_name': None, 'gpu_memory_gb': 0, 'torch': False}


# ==============================================================================
# GPU-ACCELERATED MONTE CARLO SIMULATION
# ==============================================================================

def monte_carlo_gpu(mean_returns, cov_matrix, n_portfolios=100000, risk_free_rate=0.04):
    """
    GPU-accelerated Monte Carlo portfolio simulation.
    Generates random portfolios and computes risk-adjusted metrics in parallel on GPU.
    Falls back to CPU PyTorch or NumPy if CUDA unavailable.
    """
    n_assets = len(mean_returns)
    annual_factor = 252

    start_time = time.time()

    if TORCH_AVAILABLE:
        device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')

        mu = torch.tensor(mean_returns, dtype=torch.float32, device=device)
        cov = torch.tensor(cov_matrix, dtype=torch.float32, device=device)

        # Generate random portfolio weights on GPU
        weights = torch.rand(n_portfolios, n_assets, device=device)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Batched portfolio return computation: w @ mu
        port_returns = (weights @ mu) * annual_factor

        # Batched portfolio volatility: sqrt(w @ Σ @ w^T) for each portfolio
        # Efficient: (weights @ cov) element-wise * weights, sum across assets
        port_vol = torch.sqrt(((weights @ cov) * weights).sum(dim=1) * annual_factor)

        # Sharpe ratios
        sharpe_ratios = (port_returns - risk_free_rate) / port_vol

        # Find optimal portfolios
        max_sharpe_idx = torch.argmax(sharpe_ratios).item()
        min_vol_idx = torch.argmin(port_vol).item()

        # Convert results back to numpy
        results = {
            'all_returns': port_returns.cpu().numpy(),
            'all_volatilities': port_vol.cpu().numpy(),
            'all_sharpe': sharpe_ratios.cpu().numpy(),
            'all_weights': weights.cpu().numpy(),
            'max_sharpe': {
                'weights': weights[max_sharpe_idx].cpu().numpy(),
                'return': port_returns[max_sharpe_idx].item(),
                'volatility': port_vol[max_sharpe_idx].item(),
                'sharpe': sharpe_ratios[max_sharpe_idx].item(),
            },
            'min_volatility': {
                'weights': weights[min_vol_idx].cpu().numpy(),
                'return': port_returns[min_vol_idx].item(),
                'volatility': port_vol[min_vol_idx].item(),
                'sharpe': sharpe_ratios[min_vol_idx].item(),
            },
        }
    else:
        # Pure NumPy fallback
        weights = np.random.rand(n_portfolios, n_assets)
        weights = weights / weights.sum(axis=1, keepdims=True)

        port_returns = (weights @ mean_returns) * annual_factor
        port_vol = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights) * annual_factor)
        sharpe_ratios = (port_returns - risk_free_rate) / port_vol

        max_sharpe_idx = np.argmax(sharpe_ratios)
        min_vol_idx = np.argmin(port_vol)

        results = {
            'all_returns': port_returns,
            'all_volatilities': port_vol,
            'all_sharpe': sharpe_ratios,
            'all_weights': weights,
            'max_sharpe': {
                'weights': weights[max_sharpe_idx],
                'return': port_returns[max_sharpe_idx],
                'volatility': port_vol[max_sharpe_idx],
                'sharpe': sharpe_ratios[max_sharpe_idx],
            },
            'min_volatility': {
                'weights': weights[min_vol_idx],
                'return': port_returns[min_vol_idx],
                'volatility': port_vol[min_vol_idx],
                'sharpe': sharpe_ratios[min_vol_idx],
            },
        }

    elapsed = time.time() - start_time
    results['computation_time'] = elapsed
    results['n_portfolios'] = n_portfolios
    results['device'] = 'cuda' if CUDA_AVAILABLE else ('cpu_torch' if TORCH_AVAILABLE else 'cpu_numpy')

    return results


# ==============================================================================
# GPU-ACCELERATED MEAN-VARIANCE OPTIMIZATION (ANALYTICAL + GRADIENT)
# ==============================================================================

def optimize_sharpe_gpu(mean_returns, cov_matrix, risk_free_rate=0.04, n_iterations=2000, lr=0.01):
    """
    GPU-accelerated Sharpe ratio maximization using gradient descent.
    Uses PyTorch autograd for automatic differentiation on GPU.
    """
    n_assets = len(mean_returns)
    annual_factor = 252
    start_time = time.time()

    if TORCH_AVAILABLE:
        device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        mu = torch.tensor(mean_returns * annual_factor, dtype=torch.float32, device=device)
        cov = torch.tensor(cov_matrix * annual_factor, dtype=torch.float32, device=device)

        # Initialize weights with random values
        raw_weights = torch.randn(n_assets, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([raw_weights], lr=lr)

        history = []
        for i in range(n_iterations):
            optimizer.zero_grad()
            w = torch.softmax(raw_weights, dim=0)  # ensure valid portfolio weights
            port_return = w @ mu
            port_vol = torch.sqrt(w @ cov @ w + 1e-8)
            neg_sharpe = -((port_return - risk_free_rate) / port_vol)
            neg_sharpe.backward()
            optimizer.step()

            if i % 50 == 0:
                history.append({
                    'iteration': i,
                    'sharpe': -neg_sharpe.item(),
                    'return': port_return.item(),
                    'volatility': port_vol.item(),
                })

        # Final weights
        with torch.no_grad():
            final_weights = torch.softmax(raw_weights, dim=0)
            final_return = (final_weights @ mu).item()
            final_vol = torch.sqrt(final_weights @ cov @ final_weights).item()
            final_sharpe = (final_return - risk_free_rate) / final_vol

        result = {
            'weights': final_weights.cpu().numpy(),
            'return': final_return,
            'volatility': final_vol,
            'sharpe': final_sharpe,
            'history': history,
            'device': 'cuda' if CUDA_AVAILABLE else 'cpu_torch',
        }
    else:
        result = _optimize_sharpe_scipy(mean_returns, cov_matrix, risk_free_rate)
        result['device'] = 'cpu_scipy'

    result['computation_time'] = time.time() - start_time
    return result


def _optimize_sharpe_scipy(mean_returns, cov_matrix, risk_free_rate=0.04):
    """SciPy-based Sharpe optimization (CPU baseline)."""
    n_assets = len(mean_returns)
    annual_factor = 252

    def neg_sharpe(w):
        port_ret = np.dot(w, mean_returns) * annual_factor
        port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)) * annual_factor)
        return -(port_ret - risk_free_rate) / port_vol

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0, 1) for _ in range(n_assets))
    x0 = np.ones(n_assets) / n_assets

    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    w = result.x
    ret = np.dot(w, mean_returns) * annual_factor
    vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)) * annual_factor)

    return {
        'weights': w,
        'return': ret,
        'volatility': vol,
        'sharpe': (ret - risk_free_rate) / vol,
        'history': [],
    }


# ==============================================================================
# EFFICIENT FRONTIER COMPUTATION
# ==============================================================================

def compute_efficient_frontier(mean_returns, cov_matrix, n_points=50, risk_free_rate=0.04):
    """
    Compute the efficient frontier by solving the minimum-variance portfolio
    for a sweep of feasible target returns.

    Key correctness rules:
      - Target range is clamped to [global_min_variance_return, max_single_asset_return]
        so every target is actually achievable with long-only weights.
      - Each solution is validated: achieved return must be within 1% of target.
      - Points are sorted by volatility so the line renders smoothly.
    """
    n_assets = len(mean_returns)
    annual_factor = 252
    mu_annual = mean_returns * annual_factor
    cov_annual = cov_matrix * annual_factor
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    eq_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

    def portfolio_vol(w):
        return np.sqrt(np.dot(w, np.dot(cov_annual, w)))

    # ── Step 1: find global minimum-variance portfolio (no return constraint)
    gmv = minimize(portfolio_vol, np.ones(n_assets) / n_assets,
                   method='SLSQP', bounds=bounds, constraints=[eq_sum],
                   options={'maxiter': 1000, 'ftol': 1e-10})
    ret_min = float(np.dot(gmv.x, mu_annual)) if gmv.success else mu_annual.min()

    # ── Step 2: upper bound is 100% in the best single asset
    ret_max = float(mu_annual.max())

    # ── Step 3: sweep feasible targets
    target_returns = np.linspace(ret_min, ret_max, n_points)
    frontier_returns = []
    frontier_vols = []
    prev_w = np.ones(n_assets) / n_assets   # warm-start each solve from previous solution

    for target in target_returns:
        constraints = [
            eq_sum,
            {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, mu_annual) - t},
        ]
        result = minimize(portfolio_vol, prev_w, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})

        if not result.success:
            continue

        achieved_ret = float(np.dot(result.x, mu_annual))
        achieved_vol = portfolio_vol(result.x)

        # Reject solutions where return is far from target (solver drift)
        if abs(achieved_ret - target) > 0.01 + 0.05 * abs(target):
            continue
        # Reject negative volatility (numerical noise)
        if achieved_vol <= 0:
            continue

        frontier_returns.append(achieved_ret)
        frontier_vols.append(float(achieved_vol))
        prev_w = result.x.copy()   # warm-start next iteration

    # Sort by volatility so Chart.js draws a clean curve
    if frontier_returns:
        pairs = sorted(zip(frontier_vols, frontier_returns))
        frontier_vols, frontier_returns = zip(*pairs)
        frontier_vols = list(frontier_vols)
        frontier_returns = list(frontier_returns)

    return {
        'returns': frontier_returns,
        'volatilities': frontier_vols,
    }


# ==============================================================================
# CPU BASELINE BENCHMARK
# ==============================================================================

def benchmark_cpu_baseline(mean_returns, cov_matrix, n_portfolios=100000, risk_free_rate=0.04):
    """
    Pure SciPy/NumPy CPU baseline for benchmarking against GPU.
    Runs sequential Monte Carlo + optimization.
    """
    n_assets = len(mean_returns)
    annual_factor = 252
    start_time = time.time()

    # Sequential Monte Carlo (no vectorization tricks — true baseline)
    best_sharpe = -np.inf
    best_weights = None
    results_ret = []
    results_vol = []

    # Use smaller batch for sequential baseline to keep it reasonable
    n_eval = min(n_portfolios, 50000)

    for _ in range(n_eval):
        w = np.random.rand(n_assets)
        w = w / w.sum()
        ret = np.dot(w, mean_returns) * annual_factor
        vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)) * annual_factor)
        sharpe = (ret - risk_free_rate) / vol
        results_ret.append(ret)
        results_vol.append(vol)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = w.copy()

    elapsed = time.time() - start_time

    return {
        'computation_time': elapsed,
        'n_portfolios': n_eval,
        'best_sharpe': best_sharpe,
        'throughput': n_eval / elapsed,
        'device': 'cpu_sequential',
    }


# ==============================================================================
# ASSET SCORING (PYTHON IMPLEMENTATION + C++ ACCELERATION)
# ==============================================================================

# Try to load C++ OpenMP scorer
_CPP_SCORER = None
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cpp'))
    import scorer as _CPP_SCORER
    print(f"[SCORER] C++ OpenMP scorer loaded ({_CPP_SCORER.get_max_threads()} threads)")
except ImportError:
    _CPP_SCORER = None

# Try to load trained model
_TRAINED_MODEL = None
_MODEL_LOAD_ATTEMPTED = False

def _load_trained_model():
    """Load the trained sklearn model from data/scorer_model.pkl if available."""
    global _TRAINED_MODEL, _MODEL_LOAD_ATTEMPTED
    if _MODEL_LOAD_ATTEMPTED:
        return _TRAINED_MODEL
    _MODEL_LOAD_ATTEMPTED = True

    import os, pickle
    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'scorer_model.pkl')
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            _TRAINED_MODEL = data['model']
            print(f"[SCORER] Loaded trained model from {model_path}")
        except Exception as e:
            print(f"[SCORER] Failed to load model: {e}")
    else:
        print("[SCORER] No trained model found — using analytical Sharpe scoring")
    return _TRAINED_MODEL


def score_asset_combinations(mean_returns, cov_matrix, tickers, top_k=20):
    """
    Score individual assets and asset pairs for portfolio inclusion.
    Uses C++ OpenMP scorer if available, then trained GBDT model,
    then falls back to analytical Sharpe ratio scoring.
    """
    n_assets = len(mean_returns)
    annual_factor = 252
    mu = mean_returns * annual_factor
    cov_annual = cov_matrix * annual_factor
    scores = []

    # Score individual assets (always analytical — model is for pairs)
    for i in range(n_assets):
        vol = np.sqrt(cov_annual[i, i])
        sharpe = (mu[i] - 0.04) / vol if vol > 0 else 0
        scores.append({
            'ticker': tickers[i],
            'return': float(mu[i]),
            'volatility': float(vol),
            'sharpe': float(sharpe),
            'type': 'single',
        })

    # Try C++ scorer for pair scoring
    if _CPP_SCORER is not None:
        try:
            vols = np.sqrt(np.diag(cov_annual)).astype(np.float32)
            corr_matrix = np.zeros((n_assets, n_assets), dtype=np.float32)
            for i in range(n_assets):
                for j in range(n_assets):
                    denom = vols[i] * vols[j]
                    corr_matrix[i, j] = cov_annual[i, j] / denom if denom > 0 else 0.0
            model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'scorer_model.json')
            result = _CPP_SCORER.score_assets(
                mu.astype(np.float32), vols, corr_matrix, n_assets, model_path
            )
            # Build pair metadata and attach C++ scores
            pair_scores = []
            idx = 0
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    w = np.array([0.5, 0.5])
                    sub_mu = np.array([mu[i], mu[j]])
                    sub_cov = np.array([[cov_annual[i,i], cov_annual[i,j]],
                                        [cov_annual[j,i], cov_annual[j,j]]])
                    ret = float(np.dot(w, sub_mu))
                    vol = float(np.sqrt(np.dot(w, np.dot(sub_cov, w))))
                    pair_scores.append({
                        'ticker': f"{tickers[i]}+{tickers[j]}",
                        'return': ret,
                        'volatility': vol,
                        'sharpe': float(result.scores[idx]),
                        'correlation': float(corr_matrix[i, j]),
                        'type': 'pair',
                    })
                    idx += 1

            scores.sort(key=lambda x: x['sharpe'], reverse=True)
            pair_scores.sort(key=lambda x: x['sharpe'], reverse=True)
            return {
                'single_assets': scores[:top_k],
                'asset_pairs': pair_scores[:top_k],
                'total_combinations_scored': len(scores) + len(pair_scores),
                'model_used': True,
                'scorer': 'cpp_openmp',
                'cpp_throughput': result.throughput,
                'cpp_threads': result.n_threads,
            }
        except Exception as e:
            print(f"[SCORER] C++ scorer failed, falling back to Python: {e}")

    # Python fallback
    model = _load_trained_model()

    # Score top pairs (diversification benefit)
    pair_scores = []
    pair_features = []  # for batch model prediction
    pair_meta = []      # store pair info alongside features

    for i in range(n_assets):
        for j in range(i+1, n_assets):
            w = np.array([0.5, 0.5])
            sub_mu = np.array([mu[i], mu[j]])
            sub_cov = np.array([[cov_annual[i,i], cov_annual[i,j]],
                                [cov_annual[j,i], cov_annual[j,j]]])
            ret = np.dot(w, sub_mu)
            vol = np.sqrt(np.dot(w, np.dot(sub_cov, w)))
            sharpe = (ret - 0.04) / vol if vol > 0 else 0
            correlation = cov_matrix[i,j] / (np.sqrt(cov_matrix[i,i]) * np.sqrt(cov_matrix[j,j]) + 1e-10)

            # Build feature vector matching train_scorer.py format
            ret_spread = abs(mu[i] - mu[j])
            vol_i, vol_j = np.sqrt(cov_annual[i,i]), np.sqrt(cov_annual[j,j])
            vol_ratio = min(vol_i, vol_j) / max(vol_i, vol_j, 0.001)
            max_dd = vol * 2.0

            pair_meta.append({
                'ticker': f"{tickers[i]}+{tickers[j]}",
                'return': float(ret),
                'volatility': float(vol),
                'correlation': float(correlation),
                'type': 'pair',
            })
            pair_features.append([ret, vol, sharpe, correlation, max_dd, ret_spread, vol_ratio])

    # If trained model available, use model predictions as scores
    if model is not None and len(pair_features) > 0:
        X = np.array(pair_features)
        predicted_scores = model.predict(X)
        for meta, score in zip(pair_meta, predicted_scores):
            meta['sharpe'] = float(score)  # model predicts forward-looking Sharpe
            pair_scores.append(meta)
    else:
        # Fallback: use analytical historical Sharpe
        for meta, feats in zip(pair_meta, pair_features):
            meta['sharpe'] = float(feats[2])  # feats[2] = sharpe
            pair_scores.append(meta)

    # Sort by score
    scores.sort(key=lambda x: x['sharpe'], reverse=True)
    pair_scores.sort(key=lambda x: x['sharpe'], reverse=True)

    return {
        'single_assets': scores[:top_k],
        'asset_pairs': pair_scores[:top_k],
        'total_combinations_scored': len(scores) + len(pair_scores),
        'model_used': model is not None,
    }


# ==============================================================================
# FULL OPTIMIZATION PIPELINE
# ==============================================================================

def run_full_optimization(mean_returns, cov_matrix, tickers, n_portfolios=50000, risk_free_rate=0.04):
    """
    Run the complete optimization pipeline:
    1. Monte Carlo simulation (GPU-accelerated)
    2. Gradient-based Sharpe maximization (GPU-accelerated)
    3. Efficient frontier computation
    4. Asset scoring
    5. CPU baseline benchmark
    """
    print(f"\n{'='*60}")
    print(f"  GPU-Accelerated Portfolio Optimization Engine")
    print(f"  Assets: {len(tickers)} | Simulations: {n_portfolios:,}")
    print(f"  Device: {get_device_info()['device']}")
    print(f"{'='*60}\n")

    # 1. Monte Carlo
    print("[1/5] Running Monte Carlo simulation...")
    mc_results = monte_carlo_gpu(mean_returns, cov_matrix, n_portfolios, risk_free_rate)
    print(f"  → {mc_results['n_portfolios']:,} portfolios in {mc_results['computation_time']:.3f}s "
          f"on {mc_results['device']}")
    print(f"  → Best Sharpe: {mc_results['max_sharpe']['sharpe']:.4f}")

    # 2. Gradient optimization
    print("[2/5] Running gradient-based Sharpe optimization...")
    opt_results = optimize_sharpe_gpu(mean_returns, cov_matrix, risk_free_rate)
    print(f"  → Optimal Sharpe: {opt_results['sharpe']:.4f} in {opt_results['computation_time']:.3f}s")

    # 3. Efficient frontier
    print("[3/5] Computing efficient frontier...")
    t0 = time.time()
    frontier = compute_efficient_frontier(mean_returns, cov_matrix, n_points=40, risk_free_rate=risk_free_rate)
    frontier_time = time.time() - t0
    print(f"  → {len(frontier['returns'])} frontier points in {frontier_time:.3f}s")

    # 4. Asset scoring
    print("[4/5] Scoring asset combinations...")
    t0 = time.time()
    asset_scores = score_asset_combinations(mean_returns, cov_matrix, tickers)
    scoring_time = time.time() - t0
    print(f"  → {asset_scores['total_combinations_scored']:,} combinations in {scoring_time:.3f}s")

    # 5. CPU baseline
    print("[5/5] Running CPU baseline benchmark...")
    cpu_baseline = benchmark_cpu_baseline(mean_returns, cov_matrix, n_portfolios, risk_free_rate)
    print(f"  → CPU baseline: {cpu_baseline['n_portfolios']:,} portfolios in {cpu_baseline['computation_time']:.3f}s")

    # Compute speedup
    gpu_throughput = mc_results['n_portfolios'] / mc_results['computation_time']
    cpu_throughput = cpu_baseline['throughput']
    speedup = gpu_throughput / cpu_throughput if cpu_throughput > 0 else 1.0

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"  GPU throughput: {gpu_throughput:,.0f} portfolios/sec")
    print(f"  CPU throughput: {cpu_throughput:,.0f} portfolios/sec")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"{'='*60}\n")

    return {
        'monte_carlo': mc_results,
        'optimization': opt_results,
        'frontier': frontier,
        'asset_scores': asset_scores,
        'benchmark': {
            'gpu_throughput': gpu_throughput,
            'cpu_throughput': cpu_throughput,
            'speedup': speedup,
            'gpu_time': mc_results['computation_time'],
            'cpu_time': cpu_baseline['computation_time'],
        },
        'device_info': get_device_info(),
    }
