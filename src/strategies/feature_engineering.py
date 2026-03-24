"""
Feature Engineering Pipeline

Transforms raw OHLCV data into engineered features for ML models.
Includes:
- Price-based features (returns, volatility, etc.)
- Technical indicator features
- Volume-based features
- Statistical features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from .indicators import TechnicalIndicators, validate_data


class FeatureEngineer:
    """
    Feature engineering pipeline for strategy features.
    
    Transforms raw OHLCV data into features suitable for:
    - Technical strategy signals
    - Machine learning models
    - Factor analysis
    """
    
    @staticmethod
    def create_price_features(data: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create price-based features.
        
        Features:
        - Returns (log returns, simple returns)
        - High-Low ratio
        - Close-Open ratio
        - VWAP
        
        Args:
            data: DataFrame with OHLCV
            periods: Periods for rolling statistics
        
        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=data.index)
        
        # Simple returns
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # High-Low ratio (volatility proxy)
        features['hl_ratio'] = (data['High'] - data['Low']) / data['Close']
        
        # Close-Open ratio (intraday momentum)
        features['co_ratio'] = (data['Close'] - data['Open']) / data['Open']
        
        # Rolling returns
        for period in periods:
            features[f'return_{period}d'] = data['Close'].pct_change(period)
            features[f'return_volatile_{period}d'] = data['Close'].pct_change(period).rolling(period).std()
        
        return features
    
    @staticmethod
    def create_volume_features(data: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create volume-based features.
        
        Features:
        - Volume ratio
        - On-Balance Volume (OBV)
        - Price-Volume trend
        
        Args:
            data: DataFrame with OHLCV
            periods: Periods for rolling statistics
        
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=data.index)
        
        # Volume metrics
        features['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        # On-Balance Volume
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ema'] = TechnicalIndicators.ema(obv, 20)
        
        # Price-Volume Trend
        pvt = (data['Close'].pct_change() * data['Volume']).fillna(0).cumsum()
        features['pvt'] = pvt
        
        # Volume trend
        for period in periods:
            features[f'volume_trend_{period}d'] = data['Volume'].rolling(period).mean()
        
        return features
    
    @staticmethod
    def create_momentum_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum indicator features.
        
        Features:
        - RSI
        - MACD
        - Stochastic
        
        Args:
            data: DataFrame with OHLCV
        
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=data.index)
        
        close = data['Close']
        
        # RSI
        features['rsi_14'] = TechnicalIndicators.rsi(close, 14)
        features['rsi_21'] = TechnicalIndicators.rsi(close, 21)
        
        # MACD
        macd, signal, histogram = TechnicalIndicators.macd(close)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        
        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(data['High'], data['Low'], close)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        
        return features
    
    @staticmethod
    def create_trend_features(data: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """
        Create trend features.
        
        Features:
        - SMA
        - EMA
        - ADX
        - Price position relative to MAs
        
        Args:
            data: DataFrame with OHLCV
            periods: MA periods
        
        Returns:
            DataFrame with trend features
        """
        features = pd.DataFrame(index=data.index)
        
        close = data['Close']
        
        # Moving averages
        for period in periods:
            sma = TechnicalIndicators.sma(close, period)
            ema = TechnicalIndicators.ema(close, period)
            
            features[f'sma_{period}'] = sma
            features[f'ema_{period}'] = ema
            
            # Price relative to MA
            features[f'price_sma_ratio_{period}'] = close / sma
            features[f'price_ema_ratio_{period}'] = close / ema
        
        # ADX (trend strength)
        features['adx'] = TechnicalIndicators.adx(data['High'], data['Low'], close)
        
        return features
    
    @staticmethod
    def create_volatility_features(data: pd.DataFrame, periods: List[int] = [14, 20]) -> pd.DataFrame:
        """
        Create volatility features.
        
        Features:
        - ATR
        - Bollinger Bands
        - Historical volatility
        
        Args:
            data: DataFrame with OHLCV
            periods: Periods for ATR and volatility
        
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)
        
        close = data['Close']
        
        # ATR
        for period in periods:
            atr = TechnicalIndicators.atr(data['High'], data['Low'], close, period)
            features[f'atr_{period}'] = atr
            features[f'atr_pct_{period}'] = atr / close
        
        # Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(close, 20, 2.0)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = upper - lower
        features['bb_position'] = (close - lower) / (upper - lower)
        
        # Historical volatility
        for period in periods:
            hv = close.pct_change().rolling(period).std()
            features[f'hist_vol_{period}d'] = hv
        
        return features
    
    @staticmethod
    def create_all_features(data: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """
        Create comprehensive feature set.
        
        Args:
            data: DataFrame with OHLCV
            dropna: Drop rows with NaN values
        
        Returns:
            DataFrame with all engineered features
        """
        validate_data(data)
        
        # Combine all feature types
        features = pd.DataFrame(index=data.index)
        
        features = features.join(FeatureEngineer.create_price_features(data))
        features = features.join(FeatureEngineer.create_volume_features(data))
        features = features.join(FeatureEngineer.create_momentum_features(data))
        features = features.join(FeatureEngineer.create_trend_features(data))
        features = features.join(FeatureEngineer.create_volatility_features(data))
        
        if dropna:
            features = features.dropna()
        
        return features
    
    @staticmethod
    def normalize_features(features: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize features for ML models.
        
        Args:
            features: Feature DataFrame
            method: 'zscore' or 'minmax'
        
        Returns:
            Normalized feature DataFrame
        """
        if method == 'zscore':
            # Z-score normalization
            return (features - features.mean()) / features.std()
        elif method == 'minmax':
            # Min-Max normalization
            return (features - features.min()) / (features.max() - features.min())
        else:
            raise ValueError(f"Unknown normalization method: {method}")
