"""
Technical Indicators Library

Implements common technical indicators for strategy signals:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
"""

import pandas as pd
import numpy as np
from typing import Tuple


class TechnicalIndicators:
    """Static methods for calculating technical indicators"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            data: Price series (typically Close prices)
            period: Number of periods for MA
            
        Returns:
            Series with SMA values
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int, adjust: bool = False) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            data: Price series
            period: Number of periods for EMA
            adjust: Use pandas adjustment method
            
        Returns:
            Series with EMA values
        """
        return data.ewm(span=period, adjust=adjust).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        RSI measures momentum on a scale of 0-100.
        >70 indicates overbought, <30 indicates oversold.
        
        Args:
            data: Price series (typically Close prices)
            period: Number of periods (default 14)
            
        Returns:
            Series with RSI values (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Returns three series:
        1. MACD line (fast EMA - slow EMA)
        2. Signal line (EMA of MACD)
        3. Histogram (MACD - Signal)
        
        Args:
            data: Price series (typically Close prices)
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
            
        Returns:
            Tuple of (macd, signal_line, histogram)
        """
        ema_fast = TechnicalIndicators.ema(data, period=fast)
        ema_slow = TechnicalIndicators.ema(data, period=slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, period=signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Returns three series:
        1. Upper band (SMA + num_std * std_dev)
        2. Middle band (SMA)
        3. Lower band (SMA - num_std * std_dev)
        
        Args:
            data: Price series (typically Close prices)
            period: Period for SMA (default 20)
            num_std: Number of standard deviations (default 2)
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return upper, middle, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (volatility indicator).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for ATR (default 14)
            
        Returns:
            Series with ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Returns two series:
        1. %K (fast stochastic line)
        2. %D (slow stochastic line)
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for stochastic (default 14)
            smooth_k: K smoothing period
            smooth_d: D smoothing period
            
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent_smooth = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent_smooth.rolling(window=smooth_d).mean()
        
        return k_percent_smooth, d_percent
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (trend strength indicator).
        
        ADX > 25 indicates strong trend, < 20 indicates weak trend.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for ADX (default 14)
            
        Returns:
            Series with ADX values
        """
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        
        dx = 100 * (di_diff / di_sum)
        adx = dx.rolling(window=period).mean()
        
        return adx


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required OHLCV columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    return True
