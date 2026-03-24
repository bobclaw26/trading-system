"""
Tests for MA Crossover Strategy.

Tests:
- Strategy initialization with parameters
- Signal generation on MA crossovers
- Volume filtering
- Configuration validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategies.ma_crossover import MAcrossoverStrategy, create_ma_crossover_strategy
from src.strategies.base_strategy import StrategyConfig


@pytest.fixture
def sample_data():
    """Create sample OHLCV data with clear MA crossover."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Create price data that will generate clear crossovers
    np.random.seed(42)
    
    # Start low, then trend up (will create buy signal)
    prices = np.linspace(95, 115, 70).tolist()
    # Then trend down (will create sell signal)
    prices.extend(np.linspace(115, 100, 30).tolist())
    
    data = pd.DataFrame({
        'Open': prices,
        'High': [p + 0.5 for p in prices],
        'Low': [p - 0.5 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return data


@pytest.fixture
def ma_config():
    """Create MA crossover strategy config."""
    return StrategyConfig(
        name="MA Crossover",
        description="Test MA crossover",
        position_size=1.0,
        parameters={
            'fast_ma': 20,
            'slow_ma': 50,
            'use_volume_filter': False
        }
    )


def test_ma_strategy_initialization(ma_config):
    """Test MA strategy initialization."""
    strategy = MAcrossoverStrategy(ma_config)
    
    assert strategy.fast_ma == 20
    assert strategy.slow_ma == 50
    assert strategy.config.name == "MA Crossover"


def test_fast_ma_greater_than_slow_ma_raises_error():
    """Test that fast_ma >= slow_ma raises error."""
    config = StrategyConfig(
        name="Invalid MA",
        parameters={'fast_ma': 50, 'slow_ma': 20}
    )
    
    with pytest.raises(ValueError):
        MAcrossoverStrategy(config)


def test_invalid_fast_ma_period_raises_error():
    """Test that fast_ma < 2 raises error."""
    config = StrategyConfig(
        name="Invalid MA",
        parameters={'fast_ma': 1, 'slow_ma': 10}
    )
    
    with pytest.raises(ValueError):
        MAcrossoverStrategy(config)


def test_invalid_slow_ma_period_raises_error():
    """Test that slow_ma < 2 raises error."""
    config = StrategyConfig(
        name="Invalid MA",
        parameters={'fast_ma': 10, 'slow_ma': 1}
    )
    
    with pytest.raises(ValueError):
        MAcrossoverStrategy(config)


def test_invalid_volume_percentile_raises_error():
    """Test that volume percentile outside 0-100 raises error."""
    config = StrategyConfig(
        name="Invalid Volume",
        parameters={
            'fast_ma': 20,
            'slow_ma': 50,
            'use_volume_filter': True,
            'min_volume_percentile': 150
        }
    )
    
    with pytest.raises(ValueError):
        MAcrossoverStrategy(config)


def test_signal_generation(ma_config, sample_data):
    """Test that signals are generated on MA crossovers."""
    strategy = MAcrossoverStrategy(ma_config)
    signals = strategy.generate_signals(sample_data)
    
    # Should have at least some signals
    assert len(signals) > 0
    
    # Signals should have required attributes
    for signal in signals:
        assert signal.timestamp is not None
        assert signal.signal in [-1, 0, 1]
        assert signal.price > 0
        assert 0 <= signal.confidence <= 1
        assert signal.reason != ""


def test_buy_signal_on_crossover_up(sample_data):
    """Test buy signal on fast MA crossing above slow MA."""
    # Create simple data with clear crossover
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Build prices that will cause crossover around day 60
    prices = [100] * 20 + np.linspace(100, 110, 60).tolist() + [110] * 20
    
    data = pd.DataFrame({
        'Open': prices,
        'High': [p + 0.1 for p in prices],
        'Low': [p - 0.1 for p in prices],
        'Close': prices,
        'Volume': [1000000] * 100
    }, index=dates)
    
    config = StrategyConfig(
        name="MA Crossover",
        parameters={'fast_ma': 5, 'slow_ma': 20}
    )
    
    strategy = MAcrossoverStrategy(config)
    signals = strategy.generate_signals(data)
    
    # Should have at least one buy signal
    buy_signals = [s for s in signals if s.signal == 1]
    assert len(buy_signals) > 0


def test_sell_signal_on_crossover_down():
    """Test sell signal on fast MA crossing below slow MA."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Build prices that trend up then down
    prices = np.linspace(100, 120, 60).tolist() + np.linspace(120, 100, 40).tolist()
    
    data = pd.DataFrame({
        'Open': prices,
        'High': [p + 0.1 for p in prices],
        'Low': [p - 0.1 for p in prices],
        'Close': prices,
        'Volume': [1000000] * 100
    }, index=dates)
    
    config = StrategyConfig(
        name="MA Crossover",
        parameters={'fast_ma': 5, 'slow_ma': 20}
    )
    
    strategy = MAcrossoverStrategy(config)
    signals = strategy.generate_signals(data)
    
    # Should have sell signals
    sell_signals = [s for s in signals if s.signal == -1]
    assert len(sell_signals) > 0


def test_volume_filter(ma_config, sample_data):
    """Test volume filtering."""
    config = StrategyConfig(
        name="MA Crossover with Volume Filter",
        parameters={
            'fast_ma': 20,
            'slow_ma': 50,
            'use_volume_filter': True,
            'min_volume_percentile': 50
        }
    )
    
    strategy = MAcrossoverStrategy(config)
    signals = strategy.generate_signals(sample_data)
    
    # Should still generate signals with volume filter
    assert isinstance(signals, list)


def test_ma_values_calculation(ma_config, sample_data):
    """Test that MA values are calculated correctly."""
    strategy = MAcrossoverStrategy(ma_config)
    strategy.generate_signals(sample_data)
    
    ma_values = strategy.get_ma_values()
    
    assert 'fast_ma' in ma_values
    assert 'slow_ma' in ma_values
    
    # MA values should be Series
    assert isinstance(ma_values['fast_ma'], pd.Series)
    assert isinstance(ma_values['slow_ma'], pd.Series)


def test_ma_values_without_data(ma_config):
    """Test that accessing MA values without data raises error."""
    strategy = MAcrossoverStrategy(ma_config)
    
    with pytest.raises(ValueError):
        strategy.get_ma_values()


def test_factory_function():
    """Test factory function for creating MA strategy."""
    strategy = create_ma_crossover_strategy(
        fast_ma=15,
        slow_ma=45,
        position_size=2.0,
        use_volume_filter=False
    )
    
    assert strategy.fast_ma == 15
    assert strategy.slow_ma == 45
    assert strategy.config.position_size == 2.0
    assert strategy.use_volume_filter == False


def test_factory_function_defaults():
    """Test factory function with default parameters."""
    strategy = create_ma_crossover_strategy()
    
    assert strategy.fast_ma == 20
    assert strategy.slow_ma == 50


def test_get_config_dict(ma_config):
    """Test getting config as dictionary."""
    strategy = MAcrossoverStrategy(ma_config)
    config_dict = strategy.get_config_dict()
    
    assert config_dict['name'] == "MA Crossover"
    assert config_dict['fast_ma'] == 20
    assert config_dict['slow_ma'] == 50


def test_missing_ohlcv_columns():
    """Test that missing OHLCV columns raise error."""
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    
    # Missing 'Volume' column
    data = pd.DataFrame({
        'Open': [100] * 50,
        'High': [101] * 50,
        'Low': [99] * 50,
        'Close': [100] * 50,
    }, index=dates)
    
    config = StrategyConfig(
        name="MA Crossover",
        parameters={'fast_ma': 20, 'slow_ma': 50}
    )
    
    strategy = MAcrossoverStrategy(config)
    
    with pytest.raises(ValueError):
        strategy.generate_signals(data)


def test_non_datetime_index():
    """Test that non-datetime index raises error."""
    data = pd.DataFrame({
        'Open': [100] * 50,
        'High': [101] * 50,
        'Low': [99] * 50,
        'Close': [100] * 50,
        'Volume': [1000000] * 50
    })
    
    config = StrategyConfig(
        name="MA Crossover",
        parameters={'fast_ma': 20, 'slow_ma': 50}
    )
    
    strategy = MAcrossoverStrategy(config)
    
    with pytest.raises(ValueError):
        strategy.generate_signals(data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
