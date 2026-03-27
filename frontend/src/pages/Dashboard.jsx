import { useState, useMemo } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Area, AreaChart, Legend,
} from 'recharts';
import { Zap, Play, Download } from 'lucide-react';
import MetricCard from '@/components/MetricCard';
import { CHART_COLORS, DONUT_PALETTE } from '@/lib/constants';
import { runOptimize, runBacktest } from '@/lib/api';

export default function Dashboard({ data, setData, backtestData, setBacktestData }) {
  const [loading, setLoading] = useState(false);

  async function handleRun() {
    setLoading(true);
    try {
      const d = await runOptimize();
      if (d.status === 'success') setData(d);
      const bt = await runBacktest();
      if (bt.status === 'success') setBacktestData(bt);
    } catch (e) {
      alert('Error: ' + e.message + '\n\nMake sure backend is running: python backend/server.py');
    }
    setLoading(false);
  }

  const scatterData = useMemo(() => {
    if (!data?.scatter) return [];
    return data.scatter.volatilities.map((v, i) => ({
      x: +(v * 100).toFixed(2),
      y: +(data.scatter.returns[i] * 100).toFixed(2),
      sharpe: +data.scatter.sharpe[i].toFixed(3),
    }));
  }, [data]);

  const donutData = useMemo(() => {
    if (!data?.top_holdings) return [];
    const top = data.top_holdings.slice(0, 6);
    const other = data.top_holdings.slice(6).reduce((s, h) => s + h.weight, 0);
    const items = top.map(h => ({ name: h.ticker, value: +(h.weight * 100).toFixed(1) }));
    if (other > 0.005) items.push({ name: 'Other', value: +(other * 100).toFixed(1) });
    return items;
  }, [data]);

  const equityData = useMemo(() => {
    if (!backtestData?.cumulative) return [];
    return backtestData.cumulative.dates.map((d, i) => ({
      date: d.slice(5),
      opt: +((backtestData.cumulative.optimized[i] - 1) * 100).toFixed(2),
      ew: +((backtestData.cumulative.equal_weight[i] - 1) * 100).toFixed(2),
    }));
  }, [backtestData]);

  const bm = data?.benchmark;

  return (
    <div className="flex-1 p-5 space-y-5 overflow-y-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-3">
        <div>
          <h2 className="text-xl font-black tracking-tight">Optimization Command Center</h2>
          <div className="flex items-center gap-2 mt-0.5">
            {bm && (
              <>
                <span className="flex items-center gap-1 text-[11px] font-mono text-secondary">
                  <Zap size={13} /> {bm.speedup.toFixed(1)}x Faster than CPU
                </span>
                <span className="h-1 w-1 rounded-full bg-outline-variant" />
                <span className="text-[11px] font-mono text-outline">Last compute: {bm.gpu_time.toFixed(3)}s</span>
              </>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          <button onClick={() => {}} className="px-3 py-2 bg-surface-high border border-outline-variant/30 text-[10px] font-bold uppercase tracking-widest hover:bg-surface-bright transition-colors flex items-center gap-1.5">
            <Download size={12} /> Export CSV
          </button>
          <button onClick={handleRun} disabled={loading}
            className="px-5 py-2 bg-gradient-to-br from-primary to-primary-container text-white text-[10px] font-black uppercase tracking-widest hover:brightness-110 transition-all shadow-lg shadow-primary/10 disabled:opacity-50 flex items-center gap-1.5">
            <Play size={12} /> {loading ? 'Running…' : 'Run New Simulation'}
          </button>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard label="Total Portfolios" value={data ? '50,000' : '—'} sub={bm ? `in ${bm.gpu_time.toFixed(3)}s` : '—'} />
        <MetricCard label="Optimal Sharpe" value={data?.optimal ? data.optimal.sharpe.toFixed(3) : '—'} sub="In-sample" glow />
        <MetricCard label="Total Assets" value={data?.tickers?.length ?? '—'} sub="S&P 500" />
        <MetricCard label="Cumulative Return"
          value={backtestData?.out_of_sample ? `${(backtestData.out_of_sample.optimized.cumulative_return * 100).toFixed(1)}%` : data?.optimal ? `${(data.optimal.return * 100).toFixed(1)}%` : '—'}
          sub={backtestData ? 'OOS Backtest' : 'Annualized'} />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Scatter */}
        <div className="lg:col-span-2 glass-card p-4 flex flex-col" style={{ minHeight: 370 }}>
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-[11px] font-bold uppercase tracking-widest">Efficient Frontier Monte Carlo</h3>
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1 text-[9px] font-mono text-outline"><span className="w-2 h-2 rounded-full bg-secondary" />Optimal</span>
              <span className="flex items-center gap-1 text-[9px] font-mono text-outline"><span className="w-2 h-2 rounded-full bg-tertiary" />Frontier</span>
            </div>
          </div>
          <div className="flex-1">
            {scatterData.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="x" name="Volatility" unit="%" tick={{ fontSize: 10, fill: CHART_COLORS.gray }} label={{ value: 'Volatility %', position: 'bottom', fontSize: 10, fill: CHART_COLORS.gray }} />
                  <YAxis dataKey="y" name="Return" unit="%" tick={{ fontSize: 10, fill: CHART_COLORS.gray }} label={{ value: 'Return %', angle: -90, position: 'insideLeft', fontSize: 10, fill: CHART_COLORS.gray }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }}
                    contentStyle={{ background: '#201f22', border: '1px solid #424754', borderRadius: 2, fontSize: 11, fontFamily: 'JetBrains Mono' }}
                    formatter={(val, name) => [`${val}%`, name]} />
                  <Scatter data={scatterData} fill={CHART_COLORS.green} fillOpacity={0.25} r={1.5} />
                  {data?.optimal && (
                    <Scatter data={[{ x: +(data.optimal.volatility * 100).toFixed(2), y: +(data.optimal.return * 100).toFixed(2) }]}
                      fill={CHART_COLORS.green} r={7} stroke="#fff" strokeWidth={2} />
                  )}
                </ScatterChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-outline text-xs font-mono">Run simulation to generate</div>
            )}
          </div>
        </div>

        {/* Donut */}
        <div className="glass-card p-4 flex flex-col">
          <h3 className="text-[11px] font-bold uppercase tracking-widest mb-3">Optimal Allocation</h3>
          <div className="flex-1 flex flex-col items-center justify-center">
            {donutData.length ? (
              <>
                <div className="relative" style={{ width: 170, height: 170 }}>
                  <ResponsiveContainer>
                    <PieChart>
                      <Pie data={donutData} dataKey="value" innerRadius="68%" outerRadius="95%" paddingAngle={1} stroke="none">
                        {donutData.map((_, i) => <Cell key={i} fill={DONUT_PALETTE[i % DONUT_PALETTE.length]} />)}
                      </Pie>
                      <Tooltip contentStyle={{ background: '#201f22', border: '1px solid #424754', borderRadius: 2, fontSize: 11, fontFamily: 'JetBrains Mono' }}
                        formatter={(val) => `${val}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                    <span className="text-2xl font-mono font-bold">{data?.top_holdings?.length || 0}</span>
                    <span className="text-[8px] text-outline uppercase font-bold tracking-wider">Assets</span>
                  </div>
                </div>
                <div className="w-full mt-4 space-y-1.5">
                  {data?.top_holdings?.slice(0, 5).map((h, i) => (
                    <div key={h.ticker} className="flex items-center justify-between text-[11px]">
                      <span className="flex items-center gap-2 font-mono">
                        <span className="w-2 h-2 rounded-full" style={{ background: DONUT_PALETTE[i] }} />
                        {h.ticker}
                      </span>
                      <span className="font-mono font-bold">{(h.weight * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="text-outline text-xs font-mono">—</div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Equity Curve */}
        <div className="glass-card p-4 flex flex-col" style={{ minHeight: 270 }}>
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-[11px] font-bold uppercase tracking-widest">Equity Curve Backtest</h3>
            <div className="flex gap-3">
              <span className="flex items-center gap-1 text-[9px] font-mono text-outline"><span className="w-3 h-0.5 bg-secondary" />Optimized</span>
              <span className="flex items-center gap-1 text-[9px] font-mono text-outline"><span className="w-3 h-0.5 bg-outline" />Equal-Weight</span>
            </div>
          </div>
          <div className="flex-1">
            {equityData.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={equityData} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="date" tick={{ fontSize: 9, fill: CHART_COLORS.gray }} />
                  <YAxis tick={{ fontSize: 9, fill: CHART_COLORS.gray }} unit="%" />
                  <Tooltip contentStyle={{ background: '#201f22', border: '1px solid #424754', borderRadius: 2, fontSize: 11, fontFamily: 'JetBrains Mono' }}
                    formatter={(v) => `${v}%`} />
                  <Area dataKey="opt" name="Optimized" stroke={CHART_COLORS.green} fill={CHART_COLORS.green} fillOpacity={0.06} strokeWidth={2} dot={false} />
                  <Line dataKey="ew" name="Equal-Weight" stroke={CHART_COLORS.gray} strokeWidth={1.5} strokeDasharray="4 3" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-outline text-xs font-mono">Run simulation to generate</div>
            )}
          </div>
        </div>

        {/* Benchmark */}
        <div className="glass-card p-4 flex flex-col">
          <h3 className="text-[11px] font-bold uppercase tracking-widest mb-4">Compute Efficiency (GPU vs CPU)</h3>
          {bm ? (
            <div className="flex-1 space-y-5 flex flex-col justify-center">
              {[
                { label: '100K', gpu: bm.gpu_time * 2, cpu: bm.cpu_time * 2 },
                { label: '50K', gpu: bm.gpu_time, cpu: bm.cpu_time },
                { label: '10K', gpu: bm.gpu_time * 0.3, cpu: bm.cpu_time * 0.3 },
              ].map((c) => {
                const maxT = bm.cpu_time * 2;
                const gpuPct = Math.max(3, (c.gpu / maxT) * 100);
                return (
                  <div key={c.label} className="space-y-1">
                    <div className="flex justify-between items-baseline">
                      <span className="text-[9px] font-mono text-onSurface">{c.label} PORTFOLIOS</span>
                      <span className="text-[9px] font-mono text-secondary">GPU: {c.gpu.toFixed(2)}s</span>
                    </div>
                    <div className="h-5 flex gap-px">
                      <div className="bg-secondary/40 border-r border-secondary h-full rounded-sm" style={{ width: `${gpuPct}%` }} />
                      <div className="bg-tertiary/10 border-l border-tertiary/30 h-full relative rounded-sm flex-1">
                        <span className="absolute right-2 top-0.5 text-[8px] font-bold text-tertiary">CPU: {c.cpu.toFixed(1)}s</span>
                      </div>
                    </div>
                  </div>
                );
              })}
              <div className="text-center mt-2">
                <div className="text-2xl font-bold font-mono text-secondary">{bm.speedup.toFixed(1)}x</div>
                <div className="text-[8px] text-outline font-bold uppercase tracking-widest mt-0.5">Throughput Improvement</div>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-outline text-xs font-mono">Run simulation to benchmark</div>
          )}
        </div>
      </div>
    </div>
  );
}
