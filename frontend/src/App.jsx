import { useState, useEffect } from 'react';
import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import Dashboard from '@/pages/Dashboard';
import Frontier from '@/pages/Frontier';
import Backtest from '@/pages/Backtest';
import Ranking from '@/pages/Ranking';
import { fetchDevice } from '@/lib/api';

export default function App() {
  const [page, setPage] = useState('dashboard');
  const [device, setDevice] = useState(null);
  const [data, setData] = useState(null);
  const [backtestData, setBacktestData] = useState(null);
  const [latency, setLatency] = useState(0);

  useEffect(() => {
    fetchDevice().then(setDevice).catch(() => {});
  }, []);

  const pageProps = { data, setData, backtestData, setBacktestData, device, setLatency };

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar activePage={page} setActivePage={setPage} device={device} />
      <div className="lg:ml-56 ml-[62px] flex flex-col min-h-screen w-full">
        <Header device={device} />
        <div className="flex-1 flex flex-col">
          {page === 'dashboard' && <Dashboard {...pageProps} />}
          {page === 'frontier' && <Frontier {...pageProps} />}
          {page === 'backtest' && <Backtest {...pageProps} />}
          {page === 'ranking' && <Ranking {...pageProps} />}
        </div>
        <Footer device={device} latency={latency} />
      </div>
    </div>
  );
}
