"""
Tests for base strategy interface.

Tests:
- Strategy initialization and validation
- Configuration management
- Performance metrics calculation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.base_strategy import BaseStrategy, StrategyConfig, Signal, Position


# Create a simple test strategy
class SimpleTestStrategy(BaseStrategy):
    """Simple strategy for testing base functionality."""
    
    def validate_parameters(self) -> bool:
        return True
    
    def generate_signals(self, data: pd.DataFrame):
        # Just buy on first day, sell on last day
        signals = []
        if len(data) > 0:
            signals.append(Signal(
                timestamp=data.index[0],
                signal=1,
                price=data['Close'].iloc[0],
                confidence=1.0,
                reason="Test buy"
            ))
            signals.append(Signal(
                timestamp=data.index[-1],
                signal=-1,
                price=data['Close'].iloc[-1],
                confidence=1.0,
                reason="Test sell"
            ))
        return signals


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    np.random.seed(42)
    prices = 100 + np.random.randn(100).cumsum()
    
    data = pd.DataFrame({
        'Open': prices,
        'High': prices + np.abs(np.random.randn(100)),
        'Low': prices - np.abs(np.random.randn(100)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return data


@pytest.fixture
def strategy_config():
    """Create sample strategy config."""
    return StrategyConfig(
        name="Test Strategy",
        description="Test strategy for unit tests",
        position_size=1.0,
        parameters={'test_param': 42}
    )


def test_strategy_initialization(strategy_config):
    """Test strategy initialization."""
    strategy = SimpleTestStrategy(strategy_config)
    
    assert strategy.config.name == "Test Strategy"
    assert strategy.config.position_size == 1.0
    assert len(strategy.signals) == 0
    assert len(strategy.positions) == 0


def test_signal_creation():
    """Test Signal creation and attributes."""
    now = datetime.now()
    signal = Signal(
        timestamp=now,
        signal=1,
        price=100.0,
        confidence=0.9,
        reason="Test signal"
    )
    
    assert signal.signal == 1
    assert signal.price == 100.0
    assert signal.confidence == 0.9
    assert signal.reason == "Test signal"


def test_position_creation():
    """Test Position creation and tracking."""
    now = datetime.now()
    signal = Signal(timestamp=now, signal=1, price=100.0)
    
    position = Position(
        entry_price=100.0,
        entry_time=now,
        quantity=10.0,
        entry_signal=signal
    )
    
    assert position.entry_price == 100.0
    assert position.quantity == 10.0
    assert position.exit_price is None


def test_add_signal(strategy_config, sample_data):
    """Test adding signals to strategy."""
    strategy = SimpleTestStrategy(strategy_config)
    
    signal = Signal(
        timestamp=sample_data.index[0],
        signal=1,
        price=100.0
    )
    
    strategy.add_signal(signal)
    
    assert len(strategy.signals) == 1
    assert strategy.signals[0].signal == 1


def test_generate_signals(strategy_config, sample_data):
    """Test signal generation."""
    strategy = SimpleTestStrategy(strategy_config)
    signals = strategy.generate_signals(sample_data)
    
    assert len(signals) == 2
    assert signals[0].signal == 1  # Buy
    assert signals[1].signal == -1  # Sell


def test_position_close(strategy_config):
    """Test closing a position."""
    now = datetime.now()
    entry_signal = Signal(timestamp=now, signal=1, price=100.0)
    
    position = Position(
        entry_price=100.0,
        entry_time=now,
        quantity=10.0,
        entry_signal=entry_signal
    )
    
    exit_signal = Signal(timestamp=now + timedelta(days=1), signal=-1, price=110.0)
    
    # Calculate expected values
    expected_pnl = (110.0 - 100.0) * 10.0
    expected_pnl_pct = (110.0 - 100.0) / 100.0
    
    position.exit_price = exit_signal.price
    position.exit_time = exit_signal.timestamp
    position.exit_signal = exit_signal
    position.pnl = (position.exit_price - position.entry_price) * position.quantity
    position.pnl_pct = (position.exit_price - position.entry_price) / position.entry_price
    
    assert position.exit_price == 110.0
    assert position.pnl == pytest.approx(expected_pnl)
    assert position.pnl_pct == pytest.approx(expected_pnl_pct)


def test_metrics_empty(strategy_config):
    """Test metrics with no positions."""
    strategy = SimpleTestStrategy(strategy_config)
    metrics = strategy.calculate_metrics()
    
    assert metrics['total_return'] == 0.0
    assert metrics['sharpe_ratio'] == 0.0
    assert metrics['win_rate'] == 0.0


def test_metrics_with_trades(strategy_config):
    """Test metrics calculation with trades."""
    strategy = SimpleTestStrategy(strategy_config)
    
    # Create some positions with PnL
    now = datetime.now()
    entry_signal = Signal(timestamp=now, signal=1, price=100.0)
    
    # Winning trade
    pos1 = Position(
        entry_price=100.0,
        entry_time=now,
        quantity=1.0,
        entry_signal=entry_signal,
        exit_price=110.0,
        exit_time=now + timedelta(days=1),
        pnl_pct=0.10  # 10% gain
    )
    
    # Losing trade
    pos2 = Position(
        entry_price=100.0,
        entry_time=now + timedelta(days=2),
        quantity=1.0,
        entry_signal=entry_signal,
        exit_price=95.0,
        exit_time=now + timedelta(days=3),
        pnl_pct=-0.05  # 5% loss
    )
    
    strategy.positions = [pos1, pos2]
    metrics = strategy.calculate_metrics()
    
    assert metrics['num_trades'] == 2
    assert metrics['win_rate'] == 0.5  # 1 win, 1 loss


def test_get_config_dict(strategy_config):
    """Test getting config as dictionary."""
    strategy = SimpleTestStrategy(strategy_config)
    config_dict = strategy.get_config_dict()
    
    assert config_dict['name'] == "Test Strategy"
    assert config_dict['position_size'] == 1.0
    assert config_dict['parameters']['test_param'] == 42


def test_strategy_summary(strategy_config, sample_data):
    """Test strategy summary output."""
    strategy = SimpleTestStrategy(strategy_config)
    strategy.generate_signals(sample_data)
    
    summary = strategy.summary()
    
    assert "Test Strategy" in summary
    assert "Total Return" in summary
    assert "Sharpe Ratio" in summary
    assert "Win Rate" in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
