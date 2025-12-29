# Operations Library

Mathematical operations for alpha factor development.

## Cross-Sectional Operations

Operations that compare values across symbols at a single point in time.

### rank(values)

Rank values across symbols, normalized to [0, 1].

**Parameters:**
- `values: Dict[str, float]` - Symbol-value pairs

**Returns:**
- `Dict[str, float]` - Symbol-rank pairs where rank is in [0, 1]

**Example:**
```python
from clyptq.factors.ops import rank

scores = {"BTC/USDT": 100, "ETH/USDT": 50, "SOL/USDT": 75}
ranked = rank(scores)
# {"BTC/USDT": 1.0, "ETH/USDT": 0.0, "SOL/USDT": 0.5}
```

### normalize(values)

Z-score normalization across symbols.

**Parameters:**
- `values: Dict[str, float]` - Symbol-value pairs

**Returns:**
- `Dict[str, float]` - Normalized values (mean=0, std=1)

**Example:**
```python
from clyptq.factors.ops import normalize

scores = {"BTC/USDT": 100, "ETH/USDT": 50, "SOL/USDT": 75}
normalized = normalize(scores)
```

### winsorize(values, lower=0.05, upper=0.95)

Cap values at percentiles to handle outliers.

**Parameters:**
- `values: Dict[str, float]` - Symbol-value pairs
- `lower: float` - Lower percentile (0-1), default 0.05
- `upper: float` - Upper percentile (0-1), default 0.95

**Returns:**
- `Dict[str, float]` - Winsorized values

**Example:**
```python
from clyptq.factors.ops import winsorize

scores = {"BTC/USDT": 1000, "ETH/USDT": 50, "SOL/USDT": 75}
capped = winsorize(scores, lower=0.1, upper=0.9)
```

### demean(values)

Remove cross-sectional mean from values.

**Parameters:**
- `values: Dict[str, float]` - Symbol-value pairs

**Returns:**
- `Dict[str, float]` - Demeaned values (mean=0)

**Example:**
```python
from clyptq.factors.ops import demean

scores = {"BTC/USDT": 100, "ETH/USDT": 50, "SOL/USDT": 75}
demeaned = demean(scores)
# {"BTC/USDT": 25, "ETH/USDT": -25, "SOL/USDT": 0}
```

## Time-Series Operations

Operations that process sequential data for a single symbol.

### ts_mean(series, period)

Time-series mean over lookback period.

**Parameters:**
- `series: np.ndarray` - Price or value array
- `period: int` - Lookback period

**Returns:**
- `float` - Mean of last period values

**Example:**
```python
from clyptq.factors.ops import ts_mean

prices = np.array([100, 102, 101, 103, 105])
mean = ts_mean(prices, 3)  # Mean of [101, 103, 105] = 103
```

### ts_std(series, period)

Time-series standard deviation over period.

**Parameters:**
- `series: np.ndarray` - Price or value array
- `period: int` - Lookback period

**Returns:**
- `float` - Standard deviation of last period values

**Example:**
```python
from clyptq.factors.ops import ts_std

prices = np.array([100, 102, 101, 103, 105])
std = ts_std(prices, 5)
```

### ts_sum(series, period)

Time-series sum over period.

**Parameters:**
- `series: np.ndarray` - Price or value array
- `period: int` - Lookback period

**Returns:**
- `float` - Sum of last period values

**Example:**
```python
from clyptq.factors.ops import ts_sum

volumes = np.array([1000, 1200, 1100, 1300, 1500])
total_volume = ts_sum(volumes, 3)  # 1100 + 1300 + 1500 = 3900
```

### ts_min(series, period)

Time-series minimum over period.

**Parameters:**
- `series: np.ndarray` - Price or value array
- `period: int` - Lookback period

**Returns:**
- `float` - Minimum of last period values

**Example:**
```python
from clyptq.factors.ops import ts_min

prices = np.array([100, 102, 98, 103, 105])
low = ts_min(prices, 5)  # 98
```

### ts_max(series, period)

Time-series maximum over period.

**Parameters:**
- `series: np.ndarray` - Price or value array
- `period: int` - Lookback period

**Returns:**
- `float` - Maximum of last period values

**Example:**
```python
from clyptq.factors.ops import ts_max

prices = np.array([100, 102, 98, 103, 105])
high = ts_max(prices, 5)  # 105
```

### delay(series, period)

Get value from period bars ago.

**Parameters:**
- `series: np.ndarray` - Price or value array
- `period: int` - Lookback period

**Returns:**
- `float` - Value at series[-(period+1)]

**Example:**
```python
from clyptq.factors.ops import delay

prices = np.array([100, 102, 101, 103, 105])
past_price = delay(prices, 2)  # 101 (2 bars ago)
```

### delta(series, period)

Change over period: current - delay(period).

**Parameters:**
- `series: np.ndarray` - Price or value array
- `period: int` - Lookback period

**Returns:**
- `float` - Difference between current and period-ago value

**Example:**
```python
from clyptq.factors.ops import delta

prices = np.array([100, 102, 101, 103, 105])
change = delta(prices, 4)  # 105 - 100 = 5
```

### ts_rank(series, period)

Percentile rank of current value in period, normalized to [0, 1].

**Parameters:**
- `series: np.ndarray` - Price or value array
- `period: int` - Lookback period

**Returns:**
- `float` - Rank of current value in [0, 1]

**Example:**
```python
from clyptq.factors.ops import ts_rank

prices = np.array([100, 98, 102, 99, 105])
rank = ts_rank(prices, 5)  # 1.0 (105 is highest)
```

### correlation(series_x, series_y, period)

Rolling correlation between two series.

**Parameters:**
- `series_x: np.ndarray` - First series
- `series_y: np.ndarray` - Second series
- `period: int` - Lookback period

**Returns:**
- `float` - Correlation coefficient in [-1, 1]

**Example:**
```python
from clyptq.factors.ops import correlation

btc_prices = np.array([100, 102, 104, 103, 105])
eth_prices = np.array([200, 204, 208, 206, 210])
corr = correlation(btc_prices, eth_prices, 5)  # ~0.98
```

## Usage in Custom Factors

```python
from clyptq.factors.base import Factor
from clyptq.factors.ops import rank, ts_mean, delta
from clyptq.data.store import DataView
from typing import Dict

class CustomAlpha(Factor):
    def __init__(self, lookback: int = 20):
        super().__init__("CustomAlpha")
        self.lookback = lookback

    def compute(self, data: DataView) -> Dict[str, float]:
        scores = {}

        for symbol in data.symbols:
            prices = data.close(symbol, self.lookback + 1)

            mean_price = ts_mean(prices, self.lookback)
            price_change = delta(prices, 5)

            scores[symbol] = price_change / mean_price

        return rank(scores)
```

## Best Practices

**Cross-Sectional Operations:**
- Use `rank()` for signal normalization across universe
- Use `normalize()` for z-score based signals
- Use `demean()` for market-neutral strategies
- Use `winsorize()` before other operations to handle outliers

**Time-Series Operations:**
- Always check series length before applying operations
- Use `ts_rank()` for local extremes detection
- Combine `ts_mean()` and `ts_std()` for volatility-adjusted signals
- Use `correlation()` for pairs trading or sector rotation

**Performance:**
- Cross-sectional operations process all symbols simultaneously
- Time-series operations are per-symbol and can be parallelized
- Pre-compute common operations to avoid redundant calculations
