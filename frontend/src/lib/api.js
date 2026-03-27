const BASE = '';

export async function fetchDevice() {
  const r = await fetch(`${BASE}/api/device`);
  return r.json();
}

export async function runOptimize(params = {}) {
  const r = await fetch(`${BASE}/api/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ n_portfolios: 50000, risk_free_rate: 0.04, ...params }),
  });
  return r.json();
}

export async function runBenchmark(n = 50000) {
  const r = await fetch(`${BASE}/api/benchmark`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ n_portfolios: n }),
  });
  return r.json();
}

export async function runBacktest(params = {}) {
  const r = await fetch(`${BASE}/api/backtest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ train_days: 500, ...params }),
  });
  return r.json();
}
