from Algorithm import TradingAlgorithm


tickers = [
    'AAPL',
    'NVDA',
    'KO',
    'SOFI',
    'MSFT',
    'RDDT',
    'META',
]

algos = {}
for ticker in tickers:
    algos[ticker] = TradingAlgorithm(ticker)

for ticker, algo in algos.items():
    algo.run()
