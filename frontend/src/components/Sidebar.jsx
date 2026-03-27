import { cn } from '@/lib/utils';
import { LayoutDashboard, TrendingUp, History, BarChart3, Settings } from 'lucide-react';

const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'frontier', label: 'Efficient Frontier', icon: TrendingUp },
  { id: 'backtest', label: 'Backtesting', icon: History },
  { id: 'ranking', label: 'Asset Ranking', icon: BarChart3 },
];

export default function Sidebar({ activePage, setActivePage, device }) {
  return (
    <aside className="fixed left-0 top-0 h-full z-40 flex flex-col bg-surface-low w-[62px] lg:w-56 border-r border-white/5">
      {/* Logo */}
      <div className="p-3 lg:p-4 flex items-center gap-3">
        <div className="w-9 h-9 bg-gradient-to-br from-primary to-primary-container rounded-sm flex items-center justify-center shrink-0 shadow-lg shadow-primary/10">
          <span className="material-symbols-outlined text-white text-lg" style={{ fontVariationSettings: "'FILL' 1" }}>bolt</span>
        </div>
        <div className="hidden lg:block">
          <h1 className="text-onSurface font-black text-sm tracking-tight">Vortex</h1>
          <p className="text-[10px] text-secondary font-mono uppercase tracking-widest truncate">
            {device?.gpu_name ? device.gpu_name.replace('NVIDIA ','').replace('Tesla ','') : 'Detecting…'}
          </p>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-1.5 space-y-0.5 mt-2">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActivePage(id)}
            className={cn(
              'nav-link w-full',
              activePage === id && 'active'
            )}
          >
            <Icon size={18} />
            <span className="hidden lg:block text-[10px] font-semibold uppercase tracking-widest">{label}</span>
          </button>
        ))}
      </nav>

      {/* Bottom */}
      <div className="p-1.5 border-t border-white/5">
        <button className="nav-link w-full">
          <Settings size={18} />
          <span className="hidden lg:block text-[10px] font-semibold uppercase tracking-widest">Settings</span>
        </button>
      </div>
    </aside>
  );
}
