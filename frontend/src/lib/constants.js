export const SECTOR_MAP = {
  AAPL:'Tech',MSFT:'Tech',GOOGL:'Tech',AMZN:'Tech',NVDA:'Tech',META:'Tech',TSLA:'Tech',
  AVGO:'Tech',CRM:'Tech',CSCO:'Tech',TXN:'Tech',INTC:'Tech',QCOM:'Tech',IBM:'Tech',AMD:'Tech',ACN:'Tech',
  UNH:'Health',JNJ:'Health',MRK:'Health',ABBV:'Health',LLY:'Health',TMO:'Health',ABT:'Health',DHR:'Health',AMGN:'Health',
  JPM:'Finance',V:'Finance',MA:'Finance','BRK-B':'Finance',GS:'Finance',BLK:'Finance',
  PG:'Consumer',PEP:'Consumer',KO:'Consumer',COST:'Consumer',WMT:'Consumer',MCD:'Consumer',NKE:'Consumer',SBUX:'Consumer',HD:'Consumer',LOW:'Consumer',
  XOM:'Industrial',CVX:'Industrial',HON:'Industrial',CAT:'Industrial',BA:'Industrial',RTX:'Industrial',UNP:'Industrial',NEE:'Industrial',PM:'Industrial',
};

export function getStatus(sharpe) {
  if (sharpe > 1.0) return { label: 'High Alpha', cls: 'bg-secondary/10 text-secondary' };
  if (sharpe > 0.5) return { label: 'Growth', cls: 'bg-secondary/10 text-secondary' };
  if (sharpe > 0)   return { label: 'Stable', cls: 'bg-outline-variant/10 text-outline' };
  return { label: 'High Risk', cls: 'bg-tertiary/10 text-tertiary' };
}

export const CHART_COLORS = {
  grid: 'rgba(66,71,84,0.15)',
  green: '#4edea3',
  blue: '#adc6ff',
  blueAlt: '#4d8eff',
  red: '#ffb3ad',
  gray: '#8c909f',
  white: '#e5e1e4',
};

export const DONUT_PALETTE = ['#4edea3','#adc6ff','#4d8eff','#ffb3ad','#6ffbbe','#c2c6d6','#424754'];
