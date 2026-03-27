import { useMemo } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Area, AreaChart,
} from 'recharts';
import { CHART_COLORS } from '@/lib/constants';

export default function Frontier({ data }) {
  const scatterData = useMemo(() => {
    if (!data?.scatter) return [];
    return data.scatter.volatilities.map((v, i) => ({
      x: +(v * 100).toFixed(2),
      y: +(data.scatter.returns[i] * 100).toFixed(2),
      sharpe: +data.scatter.sharpe[i].toFixed(3),
    }));
  }, [data]);

  const convergence = useMemo(() => {
    if (!data?.optimization_history) return [];
    return data.optimization_history.map(h => ({
      iter: h.iteration,
      sharpe: +h.sharpe.toFixed(3),
    }));
  }, [data]);

  return (
    <div className="flex-1 p-5 space-y-5 overflow-y-auto">
      <h2 className="text-xl font-black tracking-tight">Efficient Frontier Analysis</h2>

      <div className="glass-card p-5" style={{ height: 480 }}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-[11px] font-bold uppercase tracking-widest">Monte Carlo Scatter — 50,000 Portfolios</h3>
          <div className="flex gap-3">
            <span className="flex items-center gap-1 text-[9px] font-mono text-outline"><span className="w-2 h-2 rounded-full bg-secondary" />Max Sharpe</span>
            <span className="flex items-center gap-1 text-[9px] font-mono text-outline"><span className="w-2 h-2 rounded-full bg-tertiary" />Min Vol</span>
          </div>
        </div>
        {scatterData.length ? (
          <ResponsiveContainer width="100%" height="90%">
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
              <XAxis dataKey="x" name="Volatility" unit="%" tick={{ fontSize: 10, fill: CHART_COLORS.gray }}
                label={{ value: 'Volatility (%)', position: 'bottom', fontSize: 11, fill: CHART_COLORS.gray }} />
              <YAxis dataKey="y" name="Return" unit="%" tick={{ fontSize: 10, fill: CHART_COLORS.gray }}
                label={{ value: 'Return (%)', angle: -90, position: 'insideLeft', fontSize: 11, fill: CHART_COLORS.gray }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{ background: '#201f22', border: '1px solid #424754', borderRadius: 2, fontSize: 11, fontFamily: 'JetBrains Mono' }}
                formatter={(val, name) => [`${val}%`, name]} />
              <Scatter data={scatterData} fill={CHART_COLORS.green} fillOpacity={0.2} r={2} />
              {data?.optimal && (
                <Scatter data={[{ x: +(data.optimal.volatility * 100).toFixed(2), y: +(data.optimal.return * 100).toFixed(2) }]}
                  fill={CHART_COLORS.green} r={8} stroke="#fff" strokeWidth={2} name="Max Sharpe" />
              )}
              {data?.min_volatility && (
                <Scatter data={[{ x: +(data.min_volatility.volatility * 100).toFixed(2), y: +(data.min_volatility.return * 100).toFixed(2) }]}
                  fill={CHART_COLORS.red} r={7} stroke="#fff" strokeWidth={2} name="Min Vol" />
              )}
            </ScatterChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-outline text-xs font-mono">Run simulation on Dashboard first</div>
        )}
      </div>

      <div className="glass-card p-5" style={{ height: 280 }}>
        <h3 className="text-[11px] font-bold uppercase tracking-widest mb-3">Convergence — Sharpe Optimization (Gradient Descent)</h3>
        {convergence.length ? (
          <ResponsiveContainer width="100%" height="85%">
            <AreaChart data={convergence} margin={{ top: 5, right: 10, bottom: 5, left: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
              <XAxis dataKey="iter" tick={{ fontSize: 9, fill: CHART_COLORS.gray }} label={{ value: 'Iteration', position: 'bottom', fontSize: 10, fill: CHART_COLORS.gray }} />
              <YAxis tick={{ fontSize: 9, fill: CHART_COLORS.gray }} label={{ value: 'Sharpe', angle: -90, position: 'insideLeft', fontSize: 10, fill: CHART_COLORS.gray }} />
              <Tooltip contentStyle={{ background: '#201f22', border: '1px solid #424754', borderRadius: 2, fontSize: 11, fontFamily: 'JetBrains Mono' }} />
              <Area dataKey="sharpe" stroke={CHART_COLORS.green} fill={CHART_COLORS.green} fillOpacity={0.07} strokeWidth={2} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center text-outline text-xs font-mono">Run simulation on Dashboard first</div>
        )}
      </div>
    </div>
  );
}
