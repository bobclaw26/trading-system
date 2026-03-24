"""
Base Strategy Interface

Abstract base class for all trading strategies. All strategies inherit from this class.
Provides a common interface for generating trading signals, managing positions, and evaluating performance.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class Signal:
    """Represents a trading signal"""
    timestamp: datetime
    signal: int  # 1 = BUY, -1 = SELL, 0 = HOLD
    price: float
    confidence: float = 1.0  # Confidence level 0-1
    reason: str = ""  # Why the signal was generated
    

@dataclass
class Position:
    """Represents an open position"""
    entry_price: float
    entry_time: datetime
    quantity: float
    entry_signal: Signal
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_signal: Optional[Signal] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    
@dataclass
class StrategyConfig:
    """Configuration for a strategy"""
    name: str
    description: str = ""
    position_size: float = 1.0  # Base position size
    max_position_size: float = 10.0  # Max positions to hold
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    use_dynamic_sizing: bool = False
    parameters: Dict = field(default_factory=dict)  # Strategy-specific parameters


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    All strategies must inherit from this class and implement:
    - generate_signals()
    - validate_parameters()
    
    Features:
    - Signal generation interface
    - Position tracking
    - Performance metrics (Sharpe, win rate, etc.)
    - Configuration management
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy with configuration.
        
        Args:
            config: StrategyConfig object with strategy parameters
        """
        self.config = config
        self.signals: List[Signal] = []
        self.positions: List[Position] = []
        self.price_data: Optional[pd.DataFrame] = None
        self.portfolio_value: float = 100000.0  # Default starting capital
        
        # Validate configuration
        self.validate_parameters()
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on price data.
        
        Args:
            data: DataFrame with OHLCV data (Open, High, Low, Close, Volume)
                  Must have index as datetime
        
        Returns:
            List of Signal objects
        """
        pass
    
    @abstractmethod
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid
            
        Raises:
            ValueError if parameters are invalid
        """
        pass
    
    def add_signal(self, signal: Signal):
        """Record a trading signal."""
        self.signals.append(signal)
    
    def add_position(self, position: Position):
        """Record an open position."""
        self.positions.append(position)
    
    def close_position(self, position: Position, exit_signal: Signal):
        """Close an open position."""
        position.exit_price = exit_signal.price
        position.exit_time = exit_signal.timestamp
        position.exit_signal = exit_signal
        position.pnl = (position.exit_price - position.entry_price) * position.quantity
        position.pnl_pct = (position.exit_price - position.entry_price) / position.entry_price
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for backtesting.
        
        Returns:
            Dict with metrics: Sharpe ratio, win rate, max drawdown, etc.
        """
        if not self.positions:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
            }
        
        # Closed positions
        closed_positions = [p for p in self.positions if p.exit_price is not None]
        if not closed_positions:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'num_trades': 0,
            }
        
        # Calculate PnL
        pnls = [p.pnl_pct for p in closed_positions]
        returns = np.array(pnls)
        
        # Win rate
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        win_rate = winning_trades / len(pnls) if pnls else 0.0
        
        # Sharpe ratio (assuming 252 trading days)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
        
        # Max drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Total return
        total_return = np.prod(1 + returns) - 1
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_drawdown),
            'num_trades': len(pnls),
            'avg_trade_return': float(mean_return),
            'std_trade_return': float(std_return),
        }
    
    def get_config_dict(self) -> Dict:
        """Return strategy configuration as dictionary."""
        return {
            'name': self.config.name,
            'description': self.config.description,
            'position_size': self.config.position_size,
            'max_position_size': self.config.max_position_size,
            'stop_loss_pct': self.config.stop_loss_pct,
            'take_profit_pct': self.config.take_profit_pct,
            'parameters': self.config.parameters,
        }
    
    def summary(self) -> str:
        """Return summary of strategy and its performance."""
        metrics = self.calculate_metrics()
        avg_return = metrics.get('avg_trade_return', 0.0)
        return f"""
Strategy: {self.config.name}
Description: {self.config.description}

Metrics:
- Total Return: {metrics['total_return']:.2%}
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
- Win Rate: {metrics['win_rate']:.2%}
- Max Drawdown: {metrics['max_drawdown']:.2%}
- Number of Trades: {metrics['num_trades']}
- Avg Trade Return: {avg_return:.2%}

Configuration: {self.get_config_dict()}
"""
