export default function Header({ device }) {
  const isCuda = device?.device === 'cuda';
  return (
    <header className="sticky top-0 z-50 flex justify-between items-center px-5 h-11 bg-background/90 backdrop-blur-xl border-b border-white/5">
      <div className="flex items-center gap-3">
        <span className="text-sm font-black tracking-tight text-onSurface uppercase">
          Vortex: GPU Portfolio Engine
        </span>
        <span className={`hidden sm:inline-flex items-center px-2 py-0.5 text-[9px] font-mono rounded-sm border ${
          isCuda
            ? 'bg-primary-container/20 text-primary border-primary/20'
            : 'bg-outline-variant/20 text-outline border-outline/20'
        }`}>
          {isCuda ? 'CUDA ACCELERATED' : device ? 'CPU MODE' : 'DETECTING…'}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-bold text-outline uppercase tracking-wider hidden md:block">
          System Ready
        </span>
        <span className="material-symbols-outlined text-outline text-xl">account_circle</span>
      </div>
    </header>
  );
}
