export default function Footer({ device, latency }) {
  return (
    <footer className="h-7 bg-surface-low border-t border-white/5 flex items-center justify-between px-5 text-[8px] font-mono text-outline uppercase tracking-[2px]">
      <div className="flex items-center gap-3">
        <span className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-secondary status-pulse" />
          System Ready
        </span>
        {latency > 0 && <span>Latency: {latency}ms</span>}
      </div>
      <div className="flex items-center gap-3">
        <span>{device?.device === 'cuda' ? `CUDA • ${device.gpu_name || 'GPU'}` : 'CPU • NumPy'}</span>
        <span>Vortex Engine v4.2.0</span>
      </div>
    </footer>
  );
}
