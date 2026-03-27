import { useState, useMemo } from 'react';
import { Search, ChevronLeft, ChevronRight, Download } from 'lucide-react';
import { SECTOR_MAP, getStatus } from '@/lib/constants';
import { cn } from '@/lib/utils';

const PER_PAGE = 12;

export default function Ranking({ data }) {
  const [filter, setFilter] = useState('');
  const [page, setPage] = useState(0);
  const [sortKey, setSortKey] = useState('sharpe');
  const [sortDir, setSortDir] = useState(-1);

  const allAssets = useMemo(() => {
    if (!data?.asset_scores?.single_assets) return [];
    return data.asset_scores.single_assets.map(a => ({
      ticker: a.ticker,
      sector: SECTOR_MAP[a.ticker] || 'Other',
      ret: a.return,
      vol: a.volatility,
      sharpe: a.sharpe,
    }));
  }, [data]);

  const filtered = useMemo(() => {
    const q = filter.toUpperCase();
    let list = allAssets;
    if (q) list = list.filter(a => a.ticker.includes(q) || a.sector.toUpperCase().includes(q));
    list.sort((a, b) => (a[sortKey] - b[sortKey]) * sortDir);
    return list;
  }, [allAssets, filter, sortKey, sortDir]);

  const paged = filtered.slice(page * PER_PAGE, (page + 1) * PER_PAGE);
  const totalPages = Math.ceil(filtered.length / PER_PAGE);

  function handleSort(key) {
    if (sortKey === key) setSortDir(d => d * -1);
    else { setSortKey(key); setSortDir(-1); }
    setPage(0);
  }

  function exportCSV() {
    if (!allAssets.length) return;
    let csv = 'Ticker,Sector,AnnualReturn%,Volatility%,Sharpe\n';
    allAssets.forEach(a => { csv += `${a.ticker},${a.sector},${(a.ret*100).toFixed(2)},${(a.vol*100).toFixed(2)},${a.sharpe.toFixed(3)}\n`; });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a'); link.href = url; link.download = 'vortex_assets.csv'; link.click();
    URL.revokeObjectURL(url);
  }

  const SortHeader = ({ label, field, align }) => (
    <th
      className={cn('px-4 py-2.5 font-semibold cursor-pointer hover:text-onSurface select-none transition-colors', align)}
      onClick={() => handleSort(field)}
    >
      {label} {sortKey === field ? (sortDir === -1 ? '↓' : '↑') : ''}
    </th>
  );

  return (
    <div className="flex-1 p-5 space-y-5 overflow-y-auto">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-3">
        <h2 className="text-xl font-black tracking-tight">Asset Performance ({allAssets.length} Universe)</h2>
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-outline" />
            <input
              value={filter}
              onChange={e => { setFilter(e.target.value); setPage(0); }}
              className="bg-surface-container border border-outline-variant/20 text-[11px] font-mono pl-8 pr-3 py-1.5 w-44 rounded-sm focus:ring-1 focus:ring-primary/30 focus:outline-none text-onSurface placeholder:text-outline"
              placeholder="Filter assets..."
            />
          </div>
          <button onClick={exportCSV} className="px-3 py-1.5 bg-surface-high border border-outline-variant/30 text-[9px] font-bold uppercase tracking-widest hover:bg-surface-bright transition-colors flex items-center gap-1">
            <Download size={11} /> CSV
          </button>
        </div>
      </div>

      <div className="glass-card rounded-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-surface-low text-[9px] font-bold text-outline uppercase tracking-widest">
                <SortHeader label="Symbol" field="ticker" />
                <th className="px-4 py-2.5 font-semibold">Sector</th>
                <SortHeader label="Ann. Return %" field="ret" align="text-right" />
                <SortHeader label="Volatility %" field="vol" align="text-right" />
                <SortHeader label="Sharpe Ratio" field="sharpe" align="text-right" />
                <th className="px-4 py-2.5 font-semibold text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-[11px] font-mono">
              {paged.length ? paged.map((a, i) => {
                const retColor = a.ret > 0 ? 'text-secondary' : 'text-tertiary';
                const retSign = a.ret > 0 ? '+' : '';
                const status = getStatus(a.sharpe);
                return (
                  <tr key={a.ticker} className={cn('hover:bg-surface-container transition-colors', i % 2 === 1 && 'bg-surface-low/30')}>
                    <td className="px-4 py-2.5 font-bold text-primary">{a.ticker}</td>
                    <td className="px-4 py-2.5 font-sans text-onSurfaceVariant text-[10px]">{a.sector}</td>
                    <td className={cn('px-4 py-2.5 text-right', retColor)}>{retSign}{(a.ret * 100).toFixed(2)}</td>
                    <td className="px-4 py-2.5 text-right">{(a.vol * 100).toFixed(2)}</td>
                    <td className="px-4 py-2.5 text-right font-bold">{a.sharpe.toFixed(2)}</td>
                    <td className="px-4 py-2.5 text-center">
                      <span className={cn('px-2 py-0.5 rounded-full text-[8px] font-bold uppercase', status.cls)}>{status.label}</span>
                    </td>
                  </tr>
                );
              }) : (
                <tr><td colSpan={6} className="px-4 py-10 text-center text-outline text-xs">
                  {allAssets.length ? 'No matching assets' : 'Run simulation on Dashboard first'}
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="px-4 py-2.5 border-t border-white/5 flex justify-between items-center text-[9px] text-outline font-bold uppercase tracking-widest">
          <span>
            {filtered.length > 0
              ? `Showing ${page * PER_PAGE + 1}–${Math.min((page + 1) * PER_PAGE, filtered.length)} of ${filtered.length}`
              : 'No assets'}
          </span>
          <div className="flex gap-1">
            <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
              className="p-1 hover:text-onSurface transition-colors disabled:opacity-30">
              <ChevronLeft size={14} />
            </button>
            <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1}
              className="p-1 hover:text-onSurface transition-colors disabled:opacity-30">
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
