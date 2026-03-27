import { useState, useMemo } from 'react';
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts';
import { Play } from 'lucide-react';
import { CHART_COLORS } from '@/lib/constants';
import { runBacktest as apiBt } from '@/lib/api';

function StatRow({ label, stats, highlight }) {
  if (!stats) return null;
  const retColor = stats.annual_return > 0 ? 'text-secondary' : 'text-tertiary';
  return (
    <tr className={highlight ? 'bg-surface-low/40' : ''}>
      <td className={`px-4 py-2.5 text-[11px] font-mono ${highlight ? 'text-onSurface font-bold' : 'text-outline'}`}>{label}</td>
      <td className={`px-4 py-2.5 text-right text-[11px] font-mono ${retColor}`}>{(stats.annual_return * 100).toFixed(1)}%</td>
      <td className="px-4 py-2.5 text-right text-[11px] font-mono">{(stats.annual_volatility * 100).toFixed(1)}%</td>
      <td className="px-4 py-2.5 text-right text-[11px] font-mono font-bold">{stats.sharpe_ratio?.toFixed(3)}</td>
      <td className="px-4 py-2.5 text-right text-[11px] font-mono text-tertiary">{(stats.max_drawdown * 100).toFixed(1)}%</td>
      <td className="px-4 py-2.5 text-right text-[11px] font-mono">{(stats.cumulative_return * 100).toFixed(1)}%</td>
    </tr>
  );
}

export default function Backtest({ backtestData, setBacktestData }) {
  const [loading, setLoading] = useState(false);
  const [rollingData, setRollingData] = useState(null);

  async function handleRun(rolling = false) {
    setLoading(true);
    try {
      const d = await apiBt({ train_days: 500, rolling });
      if (d.status === 'success') {
        if (rolling) setRollingData(d);
        else setBacktestData(d);
      }
    } catch (e) {
      alert('Backend error: ' + e.message);
    }
    setLoading(false);
  }

  const equityData = useMemo(() => {
    const src = backtestData?.cumulative;
    if (!src) return [];
    return src.dates.map((d, i) => ({
      date: d.slice(5),
      opt: +((src.optimized[i] - 1) * 100).toFixed(2),
      ew: +((src.equal_weight[i] - 1) * 100).toFixed(2),
    }));
  }, [backtestData]);

  const rollingEquity = useMemo(() => {
    const src = rollingData?.cumulative;
    if (!src) return [];
    return src.dates.map((d, i) => ({
      date: d.slice(5),
      opt: +((src.optimized[i] - 1) * 100).toFixed(2),
      ew: +((src.equal_weight[i] - 1) * 100).toFixed(2),
    }));
  }, [rollingData]);

  const bt = backtestData;
  const sp = bt?.split;

  return (
    <div className="flex-1 p-5 space-y-5 overflow-y-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-3">
        <div>
          <h2 className="text-xl font-black tracking-tight">Backtesting</h2>
          <p className="text-[11px] text-outline font-mono mt-0.5">Out-of-sample walk-forward evaluation</p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => handleRun(false)} disabled={loading}
            className="px-4 py-2 bg-surface-high border border-outline-variant/30 text-[10px] font-bold uppercase tracking-widest hover:bg-surface-bright transition-colors disabled:opacity-50 flex items-center gap-1.5">
            <Play size={12} /> {loading ? 'Running…' : 'Single Split'}
          </button>
          <button onClick={() => handleRun(true)} disabled={loading}
            className="px-4 py-2 bg-gradient-to-br from-primary to-primary-container text-white text-[10px] font-black uppercase tracking-widest hover:brightness-110 transition-all disabled:opacity-50 flex items-center gap-1.5">
            <Play size={12} /> {loading ? 'Running…' : 'Rolling Walk-Forward'}
          </button>
        </div>
      </div>

      {/* Single Split Results */}
      <div className="glass-card rounded-sm overflow-hidden">
        {sp ? (
          <>
            <div className="px-4 py-2.5 border-b border-white/5 flex flex-wrap gap-x-4 text-[10px] font-mono text-outline">
              <span>Train: {sp.train_start} → {sp.train_end} ({sp.train_days}d)</span>
              <span>Test: {sp.test_start} → {sp.test_end} ({sp.test_days}d)</span>
            </div>
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-surface-low text-[9px] font-bold text-outline uppercase tracking-widest">
                  <th className="px-4 py-2">Portfolio</th>
                  <th className="px-4 py-2 text-right">Ann. Return</th>
                  <th className="px-4 py-2 text-right">Ann. Vol</th>
                  <th className="px-4 py-2 text-right">Sharpe</th>
                  <th className="px-4 py-2 text-right">Max DD</th>
                  <th className="px-4 py-2 text-right">Cum. Return</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                <StatRow label="In-sample (overfit)" stats={bt.in_sample} />
                <StatRow label="OOS Optimized" stats={bt.out_of_sample?.optimized} highlight />
                <StatRow label="OOS Equal-Weight" stats={bt.out_of_sample?.equal_weight} />
              </tbody>
            </table>
            <div className="px-4 py-2 border-t border-white/5 text-[10px] font-mono">
              Sharpe lift vs equal-weight:{' '}
              <span className={`font-bold ${bt.out_of_sample?.sharpe_lift > 0 ? 'text-secondary' : 'text-tertiary'}`}>
                {bt.out_of_sample?.sharpe_lift > 0 ? '+' : ''}{bt.out_of_sample?.sharpe_lift?.toFixed(3)}
              </span>
            </div>
          </>
        ) : (
          <div className="px-5 py-10 text-center text-outline text-xs font-mono">
            Click "Single Split" or "Rolling Walk-Forward" to evaluate out-of-sample performance
          </div>
        )}
      </div>

      {/* Equity Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Single split */}
        <div className="glass-card p-4 flex flex-col" style={{ minHeight: 300 }}>
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-[11px] font-bold uppercase tracking-widest">Single Split Equity Curve</h3>
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
              <div className="h-full flex items-center justify-center text-outline text-xs font-mono">Run single split backtest</div>
            )}
          </div>
        </div>

        {/* Rolling */}
        <div className="glass-card p-4 flex flex-col" style={{ minHeight: 300 }}>
          <h3 className="text-[11px] font-bold uppercase tracking-widest mb-2">Rolling Walk-Forward Equity Curve</h3>
          <div className="flex-1">
            {rollingEquity.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={rollingEquity} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                  <XAxis dataKey="date" tick={{ fontSize: 9, fill: CHART_COLORS.gray }} />
                  <YAxis tick={{ fontSize: 9, fill: CHART_COLORS.gray }} unit="%" />
                  <Tooltip contentStyle={{ background: '#201f22', border: '1px solid #424754', borderRadius: 2, fontSize: 11, fontFamily: 'JetBrains Mono' }}
                    formatter={(v) => `${v}%`} />
                  <Area dataKey="opt" name="Optimized" stroke={CHART_COLORS.blue} fill={CHART_COLORS.blue} fillOpacity={0.06} strokeWidth={2} dot={false} />
                  <Line dataKey="ew" name="Equal-Weight" stroke={CHART_COLORS.gray} strokeWidth={1.5} strokeDasharray="4 3" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-outline text-xs font-mono">Run rolling walk-forward backtest</div>
            )}
          </div>
        </div>
      </div>

      {/* Rolling Windows Table */}
      {rollingData?.windows && (
        <div className="glass-card rounded-sm overflow-hidden">
          <div className="px-4 py-2.5 border-b border-white/5">
            <h3 className="text-[11px] font-bold uppercase tracking-widest">Rolling Windows ({rollingData.windows.length} quarters)</h3>
          </div>
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-surface-low text-[9px] font-bold text-outline uppercase tracking-widest">
                <th className="px-4 py-2">Window</th>
                <th className="px-4 py-2">Train Period</th>
                <th className="px-4 py-2">Test Period</th>
                <th className="px-4 py-2 text-right">Train Sharpe</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-[11px] font-mono">
              {rollingData.windows.map((w, i) => (
                <tr key={i} className={i % 2 === 1 ? 'bg-surface-low/30' : ''}>
                  <td className="px-4 py-2 text-primary font-bold">{i + 1}</td>
                  <td className="px-4 py-2 text-onSurfaceVariant">{w.train_start} → {w.train_end}</td>
                  <td className="px-4 py-2 text-onSurfaceVariant">{w.test_start} → {w.test_end}</td>
                  <td className="px-4 py-2 text-right font-bold">{w.opt_sharpe_train?.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rollingData.combined_out_of_sample && (
            <div className="px-4 py-2.5 border-t border-white/5 text-[10px] font-mono flex flex-wrap gap-x-5">
              <span>Optimized: Sharpe <span className="font-bold text-secondary">{rollingData.combined_out_of_sample.optimized.sharpe_ratio?.toFixed(3)}</span></span>
              <span>Equal-Weight: Sharpe <span className="font-bold">{rollingData.combined_out_of_sample.equal_weight.sharpe_ratio?.toFixed(3)}</span></span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
