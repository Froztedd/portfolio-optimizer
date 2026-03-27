"""
Train a gradient-boosted decision tree model for asset combination scoring.

This trains an XGBoost model on real market data features and exports the tree
structure as JSON so the C++ OpenMP scorer can load it for fast parallel inference.

Features per asset pair:
  - mean_return: annualized mean return of the pair
  - volatility: annualized portfolio volatility (equal-weight)
  - sharpe: Sharpe ratio of the pair
  - correlation: pairwise correlation between the two assets
  - max_drawdown: simulated max drawdown proxy (2x vol)

Target (label):
  - Forward-looking Sharpe ratio (next 60 trading days) — this is what makes
    the model predictive rather than just ranking by historical Sharpe.

Usage:
    python backend/train_scorer.py

Outputs:
    data/scorer_model.json   — exported tree structure for C++ scorer
    data/scorer_model.pkl    — sklearn model for Python fallback
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import fetch_market_data, compute_returns

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RISK_FREE_RATE = 0.04
ANNUAL_FACTOR = 252
FORWARD_WINDOW = 60  # trading days for forward-looking labels


def build_features_and_labels(prices):
    """
    Build feature matrix and forward-looking labels from price data.
    
    For each pair of assets and each time window, compute:
      - Historical features (lookback)
      - Forward Sharpe ratio (label)
    """
    returns = compute_returns(prices)
    tickers = list(returns.columns)
    n_assets = len(tickers)
    n_days = len(returns)
    
    lookback = 120  # 6 months of history for features
    step = 20       # slide window every 20 days
    
    features_list = []
    labels_list = []
    
    print(f"[TRAIN] Building features from {n_assets} assets, {n_days} trading days")
    print(f"[TRAIN] Lookback={lookback}d, Forward={FORWARD_WINDOW}d, Step={step}d")
    
    for start in range(0, n_days - lookback - FORWARD_WINDOW, step):
        hist_end = start + lookback
        fwd_end = hist_end + FORWARD_WINDOW
        
        hist_returns = returns.iloc[start:hist_end]
        fwd_returns = returns.iloc[hist_end:fwd_end]
        
        hist_mean = hist_returns.mean().values * ANNUAL_FACTOR
        hist_cov = hist_returns.cov().values * ANNUAL_FACTOR
        fwd_mean = fwd_returns.mean().values * ANNUAL_FACTOR
        fwd_cov = fwd_returns.cov().values * ANNUAL_FACTOR
        
        # Sample pairs (all pairs for small universes, random subset for large)
        if n_assets <= 50:
            pairs = [(i, j) for i in range(n_assets) for j in range(i+1, n_assets)]
        else:
            pairs = []
            for _ in range(min(500, n_assets * (n_assets - 1) // 2)):
                i, j = sorted(np.random.choice(n_assets, 2, replace=False))
                pairs.append((i, j))
            pairs = list(set(pairs))
        
        for i, j in pairs:
            # --- Historical features ---
            pair_ret = 0.5 * (hist_mean[i] + hist_mean[j])
            pair_vol = np.sqrt(
                0.25 * hist_cov[i, i] + 0.25 * hist_cov[j, j] +
                0.5 * np.sqrt(hist_cov[i, i]) * np.sqrt(hist_cov[j, j]) *
                (hist_cov[i, j] / (np.sqrt(hist_cov[i, i]) * np.sqrt(hist_cov[j, j]) + 1e-10))
            )
            pair_sharpe = (pair_ret - RISK_FREE_RATE) / max(pair_vol, 0.001)
            correlation = hist_cov[i, j] / (np.sqrt(hist_cov[i, i]) * np.sqrt(hist_cov[j, j]) + 1e-10)
            max_dd = pair_vol * 2.0  # proxy
            
            # Individual asset features
            ret_spread = abs(hist_mean[i] - hist_mean[j])
            vol_ratio = min(np.sqrt(hist_cov[i, i]), np.sqrt(hist_cov[j, j])) / \
                        max(np.sqrt(hist_cov[i, i]), np.sqrt(hist_cov[j, j]), 0.001)
            
            features = [pair_ret, pair_vol, pair_sharpe, correlation, max_dd, ret_spread, vol_ratio]
            
            # --- Forward-looking label (what we predict) ---
            fwd_pair_ret = 0.5 * (fwd_mean[i] + fwd_mean[j])
            fwd_pair_vol = np.sqrt(
                0.25 * fwd_cov[i, i] + 0.25 * fwd_cov[j, j] +
                0.5 * np.sqrt(fwd_cov[i, i] + 1e-10) * np.sqrt(fwd_cov[j, j] + 1e-10) *
                (fwd_cov[i, j] / (np.sqrt(fwd_cov[i, i]) * np.sqrt(fwd_cov[j, j]) + 1e-10))
            )
            fwd_sharpe = (fwd_pair_ret - RISK_FREE_RATE) / max(fwd_pair_vol, 0.001)
            
            if np.isfinite(fwd_sharpe) and np.all(np.isfinite(features)):
                features_list.append(features)
                labels_list.append(fwd_sharpe)
    
    feature_names = ['mean_return', 'volatility', 'sharpe', 'correlation',
                     'max_drawdown', 'return_spread', 'vol_ratio']
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"[TRAIN] Built {len(X)} samples with {len(feature_names)} features")
    return X, y, feature_names


def train_model(X, y, feature_names):
    """
    Train a gradient-boosted decision tree ensemble.
    Uses sklearn's GradientBoostingRegressor (no extra deps needed).
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n[TRAIN] Training GradientBoostingRegressor...")
    print(f"[TRAIN] Train: {len(X_train)}, Test: {len(X_test)}")
    
    model = GradientBoostingRegressor(
        n_estimators=50,      # number of trees
        max_depth=4,          # shallow trees for speed
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    print(f"\n[TRAIN] Results:")
    print(f"  Train RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.4f}")
    print(f"  Test  RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.4f}")
    print(f"  Train R²:   {r2_score(y_train, train_pred):.4f}")
    print(f"  Test  R²:   {r2_score(y_test, test_pred):.4f}")
    
    # Feature importance
    print(f"\n[TRAIN] Feature importance:")
    for name, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
        bar = '█' * int(imp * 40)
        print(f"  {name:20s} {imp:.3f} {bar}")
    
    return model


def export_tree_to_dict(tree, tree_obj):
    """Convert a sklearn DecisionTreeRegressor to a list of node dicts."""
    nodes = []
    t = tree_obj.tree_
    
    for i in range(t.node_count):
        node = {
            'feature_idx': int(t.feature[i]) if t.feature[i] >= 0 else -1,
            'threshold': float(t.threshold[i]) if t.feature[i] >= 0 else 0.0,
            'value': float(t.value[i][0][0]),
            'left': int(t.children_left[i]) if t.children_left[i] >= 0 else -1,
            'right': int(t.children_right[i]) if t.children_right[i] >= 0 else -1,
        }
        nodes.append(node)
    
    return nodes


def export_model(model, feature_names):
    """Export the trained model to JSON for C++ scorer consumption."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    model_data = {
        'n_trees': model.n_estimators,
        'learning_rate': model.learning_rate,
        'base_prediction': float(model.init_.constant_[0][0]),
        'feature_names': feature_names,
        'trees': [],
    }
    
    for i, estimator_array in enumerate(model.estimators_):
        tree_estimator = estimator_array[0]  # single output
        tree_nodes = export_tree_to_dict(tree_estimator, tree_estimator)
        model_data['trees'].append(tree_nodes)
    
    json_path = os.path.join(DATA_DIR, 'scorer_model.json')
    with open(json_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"\n[EXPORT] Saved tree model to {json_path}")
    print(f"[EXPORT] {len(model_data['trees'])} trees, "
          f"avg {np.mean([len(t) for t in model_data['trees']]):.0f} nodes/tree")
    
    # Also save sklearn model for Python fallback
    import pickle
    pkl_path = os.path.join(DATA_DIR, 'scorer_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'model': model, 'feature_names': feature_names}, f)
    print(f"[EXPORT] Saved sklearn model to {pkl_path}")
    
    return json_path


def main():
    print("=" * 60)
    print("  Asset Scorer — Model Training")
    print("=" * 60)
    
    # 1. Fetch real data
    print("\n[1/4] Fetching market data...")
    prices = fetch_market_data()
    
    # 2. Build features
    print("\n[2/4] Building features and labels...")
    X, y, feature_names = build_features_and_labels(prices)
    
    # 3. Train
    print("\n[3/4] Training model...")
    model = train_model(X, y, feature_names)
    
    # 4. Export
    print("\n[4/4] Exporting model...")
    json_path = export_model(model, feature_names)
    
    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print(f"  Model saved to: {json_path}")
    print(f"  To use in C++ scorer: load the JSON and rebuild")
    print(f"  The Python scorer will auto-load from data/scorer_model.pkl")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
