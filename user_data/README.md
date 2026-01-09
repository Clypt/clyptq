# User Data

User-specific data: strategies, results, and logs.

## Directory Structure

```
user_data/
├── strategies/              # User-defined trading strategies
│   └── my_strategy.py
│
├── backtest/                # Backtest mode
│   ├── results/             # {strategy}_{timestamp}.json
│   └── logs/                # {strategy}_{timestamp}.log
│
├── live/                    # Live trading mode
│   ├── results/             # {strategy}_{timestamp}.json
│   └── logs/                # {strategy}_{timestamp}.log
│
└── paper/                   # Paper trading mode
    ├── results/             # {strategy}_{timestamp}.json
    └── logs/                # {strategy}_{timestamp}.log
```

## Result Files

Each result JSON contains:
```json
{
  "strategy": "MyStrategy",
  "mode": "backtest",
  "start": "2024-01-01T00:00:00",
  "end": "2024-12-31T00:00:00",
  "metrics": {
    "total_return": 0.15,
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.08,
    ...
  },
  "trades": [...],
  "equity_curve": [...]
}
```

## Log Files

JSON-formatted logs with timestamps, actions, fills, and errors.

## Data Directories

- `../data/` - Downloaded market data (OHLCV from exchanges)
- `user_data/` - User strategies and trading results
