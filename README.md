# Trading System - Quantitative Strategy Framework

A Python framework for designing, implementing, and backtesting quantitative trading strategies.

## Phase 1 Focus: Moving Average Crossover Strategy

This phase implements a complete foundation for building quantitative trading strategies:

### ✅ Phase 1 Deliverables

- **Base Strategy Interface** (`src/strategies/base_strategy.py`)
  - Abstract `BaseStrategy` class
  - `Signal` class for trading signals
  - `Position` class for position tracking
  - `StrategyConfig` for configuration management
  - Performance metrics calculation (Sharpe, win rate, max drawdown)

- **MA Crossover Strategy** (`src/strategies/ma_crossover.py`) - MAIN IMPLEMENTATION
  - Fast MA (20-day) vs Slow MA (50-day)
  - Buy when fast > slow, sell when fast < slow
  - Volume filtering option
  - Factory function for easy creation

- **Momentum Strategy** (`src/strategies/momentum.py`)
  - Trend-following using momentum indicators
  - EMA for trend direction
  - RSI for confirmation

- **Mean Reversion Strategy** (`src/strategies/mean_reversion.py`)
  - Bollinger Bands for extremes
  - RSI overbought/oversold
  - Contrarian approach

- **Technical Indicators** (`src/strategies/indicators.py`)
  - SMA, EMA (moving averages)
  - RSI, MACD (momentum)
  - Bollinger Bands, ATR (volatility)
  - Stochastic, ADX (trend)

- **Feature Engineering** (`src/strategies/feature_engineering.py`)
  - Price-based features (returns, volatility)
  - Volume-based features (OBV, PVT)
  - Momentum features (RSI, MACD, Stochastic)
  - Trend features (MAs, ADX, ratios)
  - Volatility features (ATR, Bollinger Bands)

- **Comprehensive Tests**
  - `tests/test_strategies_base.py` - Base strategy tests
  - `tests/test_strategies_macrossover.py` - MA crossover tests
  - Signal generation validation
  - Parameter validation
  - Metrics calculation

- **Documentation**
  - `docs/STRATEGIES.md` - Complete strategy documentation
  - `config/ma_crossover.yaml` - Example configuration

## Project Structure

```
trading-system/
├── src/
│   └── strategies/
│       ├── __init__.py
│       ├── base_strategy.py        # Abstract base class
│       ├── ma_crossover.py         # MA crossover (PHASE 1)
│       ├── momentum.py             # Momentum strategy
│       ├── mean_reversion.py       # Mean reversion strategy
│       ├── indicators.py           # Technical indicators
│       └── feature_engineering.py  # Feature pipeline
├── tests/
│   ├── __init__.py
│   ├── test_strategies_base.py
│   ├── test_strategies_macrossover.py
│   └── data/                       # Test data
├── config/
│   └── ma_crossover.yaml          # Strategy configuration
├── docs/
│   └── STRATEGIES.md              # Strategy documentation
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone repository
cd trading-system

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Quick Start

### 1. Create and Run MA Crossover Strategy

```python
import pandas as pd
from src.strategies.ma_crossover import create_ma_crossover_strategy

# Load your OHLCV data
data = pd.read_csv('prices.csv', index_col=0, parse_dates=True)

# Create strategy with default parameters (20-day/50-day MA)
strategy = create_ma_crossover_strategy()

# Generate trading signals
signals = strategy.generate_signals(data)

# Calculate performance metrics
metrics = strategy.calculate_metrics()

# Print results
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

### 2. Customize Parameters

```python
from src.strategies.ma_crossover import create_ma_crossover_strategy

# Create strategy with custom parameters
strategy = create_ma_crossover_strategy(
    fast_ma=15,           # 15-day fast MA
    slow_ma=45,           # 45-day slow MA
    position_size=2.0,    # 2x position
    stop_loss_pct=0.03,   # 3% stop loss
    take_profit_pct=0.08  # 8% take profit
)

signals = strategy.generate_signals(data)
```

### 3. Access Moving Averages

```python
# Get the calculated MA values
ma_values = strategy.get_ma_values()

fast_ma = ma_values['fast_ma']
slow_ma = ma_values['slow_ma']

# Plot if desired
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price')
plt.plot(data.index, fast_ma, label=f'Fast MA ({strategy.fast_ma})')
plt.plot(data.index, slow_ma, label=f'Slow MA ({strategy.slow_ma})')
plt.legend()
plt.show()
```

### 4. Extract Trading Signals

```python
# Get all signals
signals = strategy.signals

for signal in signals:
    if signal.signal == 1:
        print(f"BUY @ {signal.timestamp}: ${signal.price:.2f} (Confidence: {signal.confidence:.0%})")
    elif signal.signal == -1:
        print(f"SELL @ {signal.timestamp}: ${signal.price:.2f} (Confidence: {signal.confidence:.0%})")
```

## Data Format

All strategies require OHLCV (Open, High, Low, Close, Volume) data:

```python
import pandas as pd

# Required columns
data = pd.DataFrame({
    'Open': [...],
    'High': [...],
    'Low': [...],
    'Close': [...],
    'Volume': [...]
}, index=pd.DatetimeIndex([...]))  # Index must be DatetimeIndex
```

## Strategy Comparison

| Strategy | Type | Best For | Complexity |
|----------|------|----------|-----------|
| **MA Crossover** | Trend-Following | Trending Markets | ⭐ Simple |
| **Momentum** | Trend-Following | Strong Trends | ⭐⭐ Medium |
| **Mean Reversion** | Contrarian | Ranging Markets | ⭐⭐ Medium |

## Performance Metrics

All strategies calculate:

- **Total Return** - Overall profit/loss percentage
- **Sharpe Ratio** - Risk-adjusted return (annualized)
- **Win Rate** - Percentage of profitable trades
- **Max Drawdown** - Largest peak-to-trough decline
- **Avg Trade Return** - Average return per trade
- **Trade Count** - Total number of trades

## Feature Engineering

Transform raw price data into ML-ready features:

```python
from src.strategies.feature_engineering import FeatureEngineer

# Create comprehensive feature set
features = FeatureEngineer.create_all_features(data)

# Or create specific feature types
price_features = FeatureEngineer.create_price_features(data)
volume_features = FeatureEngineer.create_volume_features(data)
momentum_features = FeatureEngineer.create_momentum_features(data)

# Normalize for ML
normalized = FeatureEngineer.normalize_features(features, method='zscore')
```

## Testing

Run test suite:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_strategies_base.py -v

# Specific test
pytest tests/test_strategies_macrossover.py::test_signal_generation -v

# With coverage
pytest tests/ --cov=src/strategies --cov-report=html
```

## Configuration

Strategies can be configured via YAML files:

```yaml
strategy:
  name: "MA Crossover"
  type: "ma_crossover"

parameters:
  fast_ma: 20
  slow_ma: 50

risk_management:
  stop_loss_pct: 0.02
  take_profit_pct: 0.05
```

Load and use configuration:

```python
import yaml
from src.strategies.ma_crossover import MAcrossoverStrategy
from src.strategies.base_strategy import StrategyConfig

with open('config/ma_crossover.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

config = StrategyConfig(
    name=config_data['strategy']['name'],
    parameters=config_data['parameters'],
    stop_loss_pct=config_data['risk_management']['stop_loss_pct'],
    take_profit_pct=config_data['risk_management']['take_profit_pct']
)

strategy = MAcrossoverStrategy(config)
```

## Phase 2/3 Roadmap

### Phase 2: ML Integration
- [ ] Random Forest for signal generation
- [ ] LSTM for sequence prediction
- [ ] Ensemble models

### Phase 3: Advanced Features
- [ ] Multi-strategy portfolio
- [ ] Advanced position sizing (Kelly Criterion)
- [ ] Correlation analysis
- [ ] Live trading integration

## Requirements

See `requirements.txt` for full list:

- pandas >= 1.3.0
- numpy >= 1.20.0
- pytest >= 6.0.0
- PyYAML >= 5.0

## Development

### Adding a New Strategy

1. Create new file in `src/strategies/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signals()` and `validate_parameters()`
4. Add unit tests in `tests/`
5. Update documentation

Example:

```python
from src.strategies.base_strategy import BaseStrategy, Signal, StrategyConfig
import pandas as pd

class MyStrategy(BaseStrategy):
    def validate_parameters(self) -> bool:
        # Validate parameters
        return True
    
    def generate_signals(self, data: pd.DataFrame):
        # Generate signals
        signals = []
        # ... implementation
        return signals
```

## License

MIT License

## Contributing

Contributions welcome! Please:
1. Follow the code structure
2. Add unit tests
3. Update documentation
4. Submit pull request

## Support

For issues or questions, please create a GitHub issue.

## Acknowledgments

- Technical analysis principles from "Technical Analysis from A to Z" (Pring)
- Strategy frameworks inspired by industry best practices
- Built with pandas, NumPy, and pytest

---

**Phase 1 Status:** ✅ Complete - MA Crossover strategy fully implemented with tests and documentation

**Next:** Phase 2 - Machine Learning Integration
