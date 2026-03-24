"""
Moving Average Crossover Strategy

Phase 1 Implementation: Simple MA Crossover Strategy
- Fast MA (20-day) crosses Slow MA (50-day)
- Buy when fast MA > slow MA
- Sell when fast MA < slow MA
- Clear entry/exit logic

This is a trend-following strategy that generates signals based on moving average crossovers.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, StrategyConfig
from .indicators import TechnicalIndicators, validate_data


class MAcrossoverStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Configuration parameters:
    - fast_ma: Fast MA period (default 20)
    - slow_ma: Slow MA period (default 50)
    - use_volume_filter: Apply volume filter (default False)
    - min_volume_percentile: Min volume percentile (default 50)
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize MA Crossover strategy."""
        # Set parameters before calling super().__init__ because validate_parameters() is called there
        self.fast_ma = config.parameters.get('fast_ma', 20)
        self.slow_ma = config.parameters.get('slow_ma', 50)
        self.use_volume_filter = config.parameters.get('use_volume_filter', False)
        self.min_volume_percentile = config.parameters.get('min_volume_percentile', 50)
        
        super().__init__(config)
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.fast_ma >= self.slow_ma:
            raise ValueError(f"fast_ma ({self.fast_ma}) must be < slow_ma ({self.slow_ma})")
        
        if self.fast_ma < 2:
            raise ValueError(f"fast_ma must be >= 2, got {self.fast_ma}")
        
        if self.slow_ma < 2:
            raise ValueError(f"slow_ma must be >= 2, got {self.slow_ma}")
        
        if self.use_volume_filter:
            if not (0 <= self.min_volume_percentile <= 100):
                raise ValueError(f"min_volume_percentile must be 0-100, got {self.min_volume_percentile}")
        
        return True
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals using MA crossover.
        
        Args:
            data: DataFrame with OHLCV data
                  Required columns: Open, High, Low, Close, Volume
                  Index must be datetime
        
        Returns:
            List of Signal objects with BUY/SELL signals
        """
        # Validate input data
        validate_data(data)
        
        self.price_data = data.copy()
        signals = []
        
        # Calculate moving averages
        fast_ma = TechnicalIndicators.sma(data['Close'], self.fast_ma)
        slow_ma = TechnicalIndicators.sma(data['Close'], self.slow_ma)
        
        # Calculate volume filter if enabled
        volume_filter = None
        if self.use_volume_filter:
            volume_threshold = data['Volume'].rolling(window=20).quantile(self.min_volume_percentile / 100)
            volume_filter = data['Volume'] >= volume_threshold
        
        # Generate signals
        for i in range(1, len(data)):
            timestamp = data.index[i]
            close_price = data['Close'].iloc[i]
            
            # Skip if we don't have enough data for MAs yet
            if pd.isna(fast_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]):
                continue
            
            # Check volume filter
            if volume_filter is not None and not volume_filter.iloc[i]:
                continue
            
            # Previous values
            fast_ma_prev = fast_ma.iloc[i - 1]
            slow_ma_prev = slow_ma.iloc[i - 1]
            fast_ma_curr = fast_ma.iloc[i]
            slow_ma_curr = slow_ma.iloc[i]
            
            # Buy signal: fast MA crosses above slow MA
            if fast_ma_prev <= slow_ma_prev and fast_ma_curr > slow_ma_curr:
                signal = Signal(
                    timestamp=timestamp,
                    signal=1,  # BUY
                    price=close_price,
                    confidence=0.8,  # 80% confidence
                    reason=f"Fast MA ({self.fast_ma}) crossed above Slow MA ({self.slow_ma})"
                )
                signals.append(signal)
                self.add_signal(signal)
            
            # Sell signal: fast MA crosses below slow MA
            elif fast_ma_prev >= slow_ma_prev and fast_ma_curr < slow_ma_curr:
                signal = Signal(
                    timestamp=timestamp,
                    signal=-1,  # SELL
                    price=close_price,
                    confidence=0.8,
                    reason=f"Fast MA ({self.fast_ma}) crossed below Slow MA ({self.slow_ma})"
                )
                signals.append(signal)
                self.add_signal(signal)
        
        return signals
    
    def get_ma_values(self) -> Dict[str, pd.Series]:
        """Return the calculated MA values."""
        if self.price_data is None:
            raise ValueError("No price data available. Call generate_signals first.")
        
        return {
            'fast_ma': TechnicalIndicators.sma(self.price_data['Close'], self.fast_ma),
            'slow_ma': TechnicalIndicators.sma(self.price_data['Close'], self.slow_ma),
        }
    
    def get_config_dict(self) -> Dict:
        """Return strategy configuration."""
        config_dict = super().get_config_dict()
        config_dict['fast_ma'] = self.fast_ma
        config_dict['slow_ma'] = self.slow_ma
        config_dict['use_volume_filter'] = self.use_volume_filter
        return config_dict


def create_ma_crossover_strategy(
    fast_ma: int = 20,
    slow_ma: int = 50,
    position_size: float = 1.0,
    use_volume_filter: bool = False,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.05,
) -> MAcrossoverStrategy:
    """
    Factory function to create a MA Crossover strategy with custom parameters.
    
    Args:
        fast_ma: Fast MA period (default 20)
        slow_ma: Slow MA period (default 50)
        position_size: Base position size (default 1.0)
        use_volume_filter: Whether to filter by volume (default False)
        stop_loss_pct: Stop loss percentage (default 2%)
        take_profit_pct: Take profit percentage (default 5%)
    
    Returns:
        MAcrossoverStrategy instance
    """
    config = StrategyConfig(
        name="MA Crossover",
        description=f"Moving Average Crossover: {fast_ma}-day fast MA vs {slow_ma}-day slow MA",
        position_size=position_size,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        parameters={
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'use_volume_filter': use_volume_filter,
        }
    )
    
    return MAcrossoverStrategy(config)
