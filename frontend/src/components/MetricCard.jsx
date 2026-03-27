import { cn } from '@/lib/utils';

export default function MetricCard({ label, value, sub, glow, className }) {
  return (
    <div className={cn('metric-card', glow && 'border-l-2 border-secondary glow-green', className)}>
      <span className="text-[9px] font-bold text-outline uppercase tracking-wider">{label}</span>
      <div className="flex items-baseline justify-between">
        <span className={cn('text-xl font-mono font-bold', glow ? 'text-secondary' : 'text-onSurface')}>
          {value}
        </span>
        {sub && <span className="text-[9px] font-mono text-outline">{sub}</span>}
      </div>
    </div>
  );
}
