"""
Momentum Strategy (Trend-Following)

Identifies strong uptrends and downtrends using:
- Price momentum (rate of change)
- EMA for trend direction
- RSI for confirmation
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, StrategyConfig
from .indicators import TechnicalIndicators, validate_data


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy - Trend Following.
    
    Configuration parameters:
    - momentum_period: Period for momentum calculation (default 10)
    - ema_period: EMA period for trend (default 20)
    - rsi_period: RSI period (default 14)
    - min_rsi_buy: Min RSI for buy signal (default 40)
    - max_rsi_sell: Max RSI for sell signal (default 60)
    - momentum_threshold: Min momentum to trigger signals (default 0.001)
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize Momentum strategy."""
        # Set parameters before calling super().__init__ because validate_parameters() is called there
        self.momentum_period = config.parameters.get('momentum_period', 10)
        self.ema_period = config.parameters.get('ema_period', 20)
        self.rsi_period = config.parameters.get('rsi_period', 14)
        self.min_rsi_buy = config.parameters.get('min_rsi_buy', 40)
        self.max_rsi_sell = config.parameters.get('max_rsi_sell', 60)
        self.momentum_threshold = config.parameters.get('momentum_threshold', 0.001)
        
        super().__init__(config)
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.momentum_period < 2:
            raise ValueError(f"momentum_period must be >= 2, got {self.momentum_period}")
        
        if self.ema_period < 2:
            raise ValueError(f"ema_period must be >= 2, got {self.ema_period}")
        
        if self.rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2, got {self.rsi_period}")
        
        if not (0 <= self.min_rsi_buy <= 100):
            raise ValueError(f"min_rsi_buy must be 0-100, got {self.min_rsi_buy}")
        
        if not (0 <= self.max_rsi_sell <= 100):
            raise ValueError(f"max_rsi_sell must be 0-100, got {self.max_rsi_sell}")
        
        return True
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals using momentum indicators.
        
        Buy: Price above EMA + positive momentum + RSI > threshold
        Sell: Price below EMA + negative momentum + RSI < threshold
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            List of Signal objects
        """
        validate_data(data)
        
        self.price_data = data.copy()
        signals = []
        
        # Calculate indicators
        close = data['Close']
        
        # Momentum (rate of change)
        momentum = close.pct_change(self.momentum_period)
        
        # EMA for trend
        ema = TechnicalIndicators.ema(close, self.ema_period)
        
        # RSI for confirmation
        rsi = TechnicalIndicators.rsi(close, self.rsi_period)
        
        for i in range(self.momentum_period, len(data)):
            timestamp = data.index[i]
            price = close.iloc[i]
            
            if pd.isna(momentum.iloc[i]) or pd.isna(ema.iloc[i]) or pd.isna(rsi.iloc[i]):
                continue
            
            mom_value = momentum.iloc[i]
            ema_value = ema.iloc[i]
            rsi_value = rsi.iloc[i]
            
            # Buy signal: Strong uptrend
            if (price > ema_value and 
                mom_value > self.momentum_threshold and
                rsi_value > self.min_rsi_buy):
                
                signal = Signal(
                    timestamp=timestamp,
                    signal=1,  # BUY
                    price=price,
                    confidence=min(abs(mom_value) / 0.01, 1.0),  # Confidence based on momentum
                    reason=f"Momentum uptrend: Price > EMA, Momentum = {mom_value:.2%}, RSI = {rsi_value:.1f}"
                )
                signals.append(signal)
                self.add_signal(signal)
            
            # Sell signal: Strong downtrend
            elif (price < ema_value and
                  mom_value < -self.momentum_threshold and
                  rsi_value < self.max_rsi_sell):
                
                signal = Signal(
                    timestamp=timestamp,
                    signal=-1,  # SELL
                    price=price,
                    confidence=min(abs(mom_value) / 0.01, 1.0),
                    reason=f"Momentum downtrend: Price < EMA, Momentum = {mom_value:.2%}, RSI = {rsi_value:.1f}"
                )
                signals.append(signal)
                self.add_signal(signal)
        
        return signals


def create_momentum_strategy(
    momentum_period: int = 10,
    ema_period: int = 20,
    rsi_period: int = 14,
    position_size: float = 1.0,
) -> MomentumStrategy:
    """Factory function to create a Momentum strategy."""
    config = StrategyConfig(
        name="Momentum",
        description=f"Trend-following momentum strategy ({momentum_period}-period)",
        position_size=position_size,
        parameters={
            'momentum_period': momentum_period,
            'ema_period': ema_period,
            'rsi_period': rsi_period,
        }
    )
    
    return MomentumStrategy(config)
