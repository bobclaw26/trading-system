"""
Mean Reversion Strategy (Contrarian)

Identifies overbought/oversold conditions and reverts to mean using:
- Bollinger Bands
- RSI
- Price deviation from moving average
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime

from .base_strategy import BaseStrategy, Signal, StrategyConfig
from .indicators import TechnicalIndicators, validate_data


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy - Contrarian approach.
    
    Trades price extreme moves back towards the mean.
    
    Configuration parameters:
    - bb_period: Bollinger Band period (default 20)
    - bb_std: Number of std deviations for BB (default 2.0)
    - rsi_period: RSI period (default 14)
    - rsi_overbought: RSI overbought threshold (default 70)
    - rsi_oversold: RSI oversold threshold (default 30)
    - sma_period: SMA for mean (default 20)
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize Mean Reversion strategy."""
        super().__init__(config)
        
        self.bb_period = config.parameters.get('bb_period', 20)
        self.bb_std = config.parameters.get('bb_std', 2.0)
        self.rsi_period = config.parameters.get('rsi_period', 14)
        self.rsi_overbought = config.parameters.get('rsi_overbought', 70)
        self.rsi_oversold = config.parameters.get('rsi_oversold', 30)
        self.sma_period = config.parameters.get('sma_period', 20)
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.bb_period < 2:
            raise ValueError(f"bb_period must be >= 2, got {self.bb_period}")
        
        if self.bb_std <= 0:
            raise ValueError(f"bb_std must be > 0, got {self.bb_std}")
        
        if self.rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2, got {self.rsi_period}")
        
        if not (0 <= self.rsi_overbought <= 100):
            raise ValueError(f"rsi_overbought must be 0-100, got {self.rsi_overbought}")
        
        if not (0 <= self.rsi_oversold <= 100):
            raise ValueError(f"rsi_oversold must be 0-100, got {self.rsi_oversold}")
        
        return True
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals using mean reversion indicators.
        
        Buy: Price touches lower BB + RSI oversold
        Sell: Price touches upper BB + RSI overbought
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            List of Signal objects
        """
        validate_data(data)
        
        self.price_data = data.copy()
        signals = []
        
        close = data['Close']
        
        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = TechnicalIndicators.bollinger_bands(
            close, self.bb_period, self.bb_std
        )
        
        # RSI
        rsi = TechnicalIndicators.rsi(close, self.rsi_period)
        
        for i in range(self.bb_period, len(data)):
            timestamp = data.index[i]
            price = close.iloc[i]
            
            if pd.isna(upper_bb.iloc[i]) or pd.isna(rsi.iloc[i]):
                continue
            
            upper = upper_bb.iloc[i]
            lower = lower_bb.iloc[i]
            rsi_value = rsi.iloc[i]
            middle = middle_bb.iloc[i]
            
            # Calculate how far price is from BB (normalized)
            bb_width = upper - lower
            price_bb_position = (price - lower) / bb_width if bb_width > 0 else 0.5
            
            # Buy signal: Price touches lower BB + oversold
            if (price <= lower and rsi_value < self.rsi_oversold):
                confidence = (self.rsi_oversold - rsi_value) / self.rsi_oversold
                
                signal = Signal(
                    timestamp=timestamp,
                    signal=1,  # BUY
                    price=price,
                    confidence=min(confidence, 1.0),
                    reason=f"Mean reversion: Price at lower BB, RSI = {rsi_value:.1f} (oversold)"
                )
                signals.append(signal)
                self.add_signal(signal)
            
            # Sell signal: Price touches upper BB + overbought
            elif (price >= upper and rsi_value > self.rsi_overbought):
                confidence = (rsi_value - self.rsi_overbought) / (100 - self.rsi_overbought)
                
                signal = Signal(
                    timestamp=timestamp,
                    signal=-1,  # SELL
                    price=price,
                    confidence=min(confidence, 1.0),
                    reason=f"Mean reversion: Price at upper BB, RSI = {rsi_value:.1f} (overbought)"
                )
                signals.append(signal)
                self.add_signal(signal)
        
        return signals


def create_mean_reversion_strategy(
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 14,
    position_size: float = 1.0,
) -> MeanReversionStrategy:
    """Factory function to create a Mean Reversion strategy."""
    config = StrategyConfig(
        name="Mean Reversion",
        description=f"Contrarian mean reversion strategy (BB {bb_period}-period)",
        position_size=position_size,
        parameters={
            'bb_period': bb_period,
            'bb_std': bb_std,
            'rsi_period': rsi_period,
        }
    )
    
    return MeanReversionStrategy(config)
