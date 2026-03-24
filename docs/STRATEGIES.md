# Trading Strategies Documentation

## Overview

This document describes the trading strategies implemented in Phase 1 of the Trading System.

All strategies inherit from `BaseStrategy` and follow a common interface:
- `generate_signals()` - Generate trading signals from price data
- `validate_parameters()` - Validate configuration parameters
- `calculate_metrics()` - Calculate performance metrics

## Phase 1 Strategies

### 1. MA Crossover Strategy (MAIN)

**Description:** Simple trend-following strategy using Moving Average crossovers.

**Entry Signal:** Fast Moving Average crosses above Slow Moving Average
**Exit Signal:** Fast Moving Average crosses below Slow Moving Average

**Parameters:**
- `fast_ma` (int, default=20): Fast MA period
- `slow_ma` (int, default=50): Slow MA period
- `use_volume_filter` (bool, default=False): Filter signals by volume
- `min_volume_percentile` (int, default=50): Min volume percentile for filter

**Example Usage:**

```python
from src.strategies.ma_crossover import create_ma_crossover_strategy
import pandas as pd

# Create strategy
strategy = create_ma_crossover_strategy(fast_ma=20, slow_ma=50)

# Load price data (OHLCV)
data = pd.read_csv('prices.csv', index_col=0, parse_dates=True)

# Generate signals
signals = strategy.generate_signals(data)

# Get performance metrics
metrics = strategy.calculate_metrics()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

**Characteristics:**
- ✅ Trend-following (momentum)
- ✅ Simple and interpretable
- ✅ Good for trending markets
- ❌ May give false signals in ranging markets
- ❌ Can lag at turning points

---

### 2. Momentum Strategy

**Description:** Trend-following strategy using momentum indicators.

**Entry Signal:** 
- Price above EMA
- Positive momentum (rate of change)
- RSI above threshold

**Exit Signal:**
- Price below EMA
- Negative momentum
- RSI below threshold

**Parameters:**
- `momentum_period` (int, default=10): Period for momentum calculation
- `ema_period` (int, default=20): EMA period for trend
- `rsi_period` (int, default=14): RSI period
- `min_rsi_buy` (int, default=40): Min RSI for buy
- `max_rsi_sell` (int, default=60): Max RSI for sell
- `momentum_threshold` (float, default=0.001): Min momentum for signals

**Characteristics:**
- ✅ Uses multiple indicators
- ✅ Momentum confirmation
- ✅ Adjustable sensitivity
- ❌ More complex than MA crossover
- ❌ Lagging signals

---

### 3. Mean Reversion Strategy

**Description:** Contrarian strategy targeting overbought/oversold extremes.

**Entry Signal:** 
- Price touches lower Bollinger Band
- RSI is oversold (<30)

**Exit Signal:**
- Price touches upper Bollinger Band
- RSI is overbought (>70)

**Parameters:**
- `bb_period` (int, default=20): Bollinger Band period
- `bb_std` (float, default=2.0): Number of std deviations
- `rsi_period` (int, default=14): RSI period
- `rsi_overbought` (int, default=70): Overbought threshold
- `rsi_oversold` (int, default=30): Oversold threshold
- `sma_period` (int, default=20): SMA for mean

**Characteristics:**
- ✅ Contrarian approach
- ✅ Good for ranging markets
- ✅ Lower drawdowns
- ❌ Against-the-trend signals
- ❌ Can whipsaw in strong trends

---

## Base Strategy Interface

### Signal Class

```python
@dataclass
class Signal:
    timestamp: datetime
    signal: int  # 1 = BUY, -1 = SELL, 0 = HOLD
    price: float
    confidence: float = 1.0  # 0-1
    reason: str = ""  # Why signal was generated
```

### Position Class

```python
@dataclass
class Position:
    entry_price: float
    entry_time: datetime
    quantity: float
    entry_signal: Signal
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_signal: Optional[Signal] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
```

### StrategyConfig Class

```python
@dataclass
class StrategyConfig:
    name: str
    description: str = ""
    position_size: float = 1.0  # Base position size
    max_position_size: float = 10.0  # Max positions to hold
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    use_dynamic_sizing: bool = False
    parameters: Dict = {}  # Strategy-specific parameters
```

## Technical Indicators

All strategies use indicators from `TechnicalIndicators` class:

### Available Indicators

- **SMA** - Simple Moving Average
- **EMA** - Exponential Moving Average
- **RSI** - Relative Strength Index (14-period)
- **MACD** - Moving Average Convergence Divergence
- **Bollinger Bands** - Price volatility bands
- **ATR** - Average True Range (volatility)
- **Stochastic** - Stochastic Oscillator
- **ADX** - Average Directional Index (trend strength)

### Example: Calculating RSI

```python
from src.strategies.indicators import TechnicalIndicators
import pandas as pd

data = pd.read_csv('prices.csv')
rsi = TechnicalIndicators.rsi(data['Close'], period=14)
print(rsi)
```

## Performance Metrics

All strategies calculate the following metrics:

| Metric | Description | Formula |
|--------|-------------|---------|
| **Total Return** | Total profit/loss from all trades | Π(1 + return_i) - 1 |
| **Sharpe Ratio** | Risk-adjusted return | (mean_return / std_return) × √252 |
| **Win Rate** | % of profitable trades | wins / total_trades |
| **Max Drawdown** | Largest peak-to-trough decline | min((cum_ret - running_max) / running_max) |
| **Avg Trade Return** | Average return per trade | mean(returns) |

## Configuration Files

### YAML Configuration Example

```yaml
strategy:
  name: MA Crossover
  type: ma_crossover
  
parameters:
  fast_ma: 20
  slow_ma: 50
  use_volume_filter: false
  
position_sizing:
  base_size: 1.0
  max_positions: 10
  
risk_management:
  stop_loss_pct: 0.02
  take_profit_pct: 0.05
```

## Feature Engineering

The `FeatureEngineer` class provides tools to transform raw OHLCV data into features:

- **Price Features:** Returns, volatility, ratios
- **Volume Features:** Volume ratio, OBV, PVT
- **Momentum Features:** RSI, MACD, Stochastic
- **Trend Features:** SMA/EMA, ADX, price ratios
- **Volatility Features:** ATR, Bollinger Bands, historical volatility

```python
from src.strategies.feature_engineering import FeatureEngineer

features = FeatureEngineer.create_all_features(data)
normalized = FeatureEngineer.normalize_features(features, method='zscore')
```

## Testing

Run tests with pytest:

```bash
pytest tests/test_strategies_base.py -v
pytest tests/test_strategies_macrossover.py -v
pytest tests/test_strategies_momentum.py -v
pytest tests/test_strategies_meanreversion.py -v
```

## Next Steps (Phase 2/3)

- [ ] Machine Learning Models Integration
  - Random Forest classifier for signal generation
  - LSTM for sequence prediction
  - Ensemble models combining multiple strategies

- [ ] Advanced Position Sizing
  - Kelly Criterion
  - Volatility-based sizing
  - Risk parity

- [ ] Portfolio Management
  - Multi-strategy combination
  - Correlation analysis
  - Optimization framework

- [ ] Live Trading
  - Integration with broker APIs
  - Order execution
  - Real-time monitoring

## Backtest Results (Phase 1)

*To be populated with historical backtest results*

## References

- Moving Average Crossover Strategy - Classic trend-following
- Bollinger Bands & RSI - Mean reversion indicators
- MACD - Momentum confirmation
- Technical Analysis from A to Z (Pring, 2002)
