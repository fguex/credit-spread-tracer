"""
Feature engineering utilities for credit spread analysis.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_equilibrium_spread(spread, method='rolling', window=252, min_periods=None):
    """
    Compute equilibrium spread level.
    
    Parameters
    ----------
    spread : pd.Series
        Credit spread time series (in basis points)
    method : str
        Method for equilibrium estimation: 'rolling' or 'expanding'
    window : int
        Window size for rolling mean (default: 252 trading days = 1 year)
    min_periods : int, optional
        Minimum observations required. If None, uses window//2
        
    Returns
    -------
    pd.Series
        Equilibrium spread series
    """
    if min_periods is None:
        min_periods = max(window // 2, 20)
    
    if method == 'rolling':
        return spread.rolling(window=window, min_periods=min_periods).mean()
    elif method == 'expanding':
        return spread.expanding(min_periods=min_periods).mean()
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_z_score(spread, equilibrium_spread, window=21, min_periods=None):
    """
    Compute z-score of spread deviation from equilibrium.
    
    Parameters
    ----------
    spread : pd.Series
        Credit spread time series
    equilibrium_spread : pd.Series
        Equilibrium spread level
    window : int
        Window for computing rolling standard deviation (default: 21 = 1 month)
    min_periods : int, optional
        Minimum observations required
        
    Returns
    -------
    pd.Series
        Z-score of deviation
    """
    if min_periods is None:
        min_periods = max(window // 2, 10)
    
    deviation = spread - equilibrium_spread
    rolling_std = deviation.rolling(window=window, min_periods=min_periods).std()
    z_score = deviation / rolling_std
    
    return z_score


def compute_momentum(spread, horizon=5):
    """
    Compute spread momentum over specified horizon.
    
    Parameters
    ----------
    spread : pd.Series
        Credit spread time series
    horizon : int
        Lookback period in days (default: 5 = 1 week)
        
    Returns
    -------
    pd.Series
        Momentum (change over horizon)
    """
    return spread - spread.shift(horizon)


def compute_realized_volatility(spread, window=21, annualize=False):
    """
    Compute realized volatility of spread changes.
    
    Parameters
    ----------
    spread : pd.Series
        Credit spread time series
    window : int
        Window for computing realized volatility (default: 21 = 1 month)
    annualize : bool
        Whether to annualize the volatility (assumes 252 trading days)
        
    Returns
    -------
    pd.Series
        Realized volatility
    """
    returns = spread.diff()
    realized_vol = returns.rolling(window=window).std()
    
    if annualize:
        realized_vol = realized_vol * np.sqrt(252)
    
    return realized_vol


def prepare_microstructure_features(spread, equilibrium_window=252, 
                                    z_score_window=21, momentum_horizon=5,
                                    volatility_window=21):
    """
    Prepare all microstructure features for analysis.
    
    Parameters
    ----------
    spread : pd.Series
        Credit spread time series
    equilibrium_window : int
        Window for equilibrium computation
    z_score_window : int
        Window for z-score computation
    momentum_horizon : int
        Horizon for momentum computation
    volatility_window : int
        Window for realized volatility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all features
    """
    df = pd.DataFrame(index=spread.index)
    df['spread'] = spread
    
    # Equilibrium and deviation
    df['equilibrium'] = compute_equilibrium_spread(spread, window=equilibrium_window)
    df['deviation'] = spread - df['equilibrium']
    
    # Z-score
    df['z_score'] = compute_z_score(spread, df['equilibrium'], window=z_score_window)
    
    # Momentum
    df['momentum'] = compute_momentum(spread, horizon=momentum_horizon)
    
    # Realized volatility
    df['realized_vol'] = compute_realized_volatility(spread, window=volatility_window)
    
    return df


def normalize_series(series, method='standardize'):
    """
    Normalize a time series.
    
    Parameters
    ----------
    series : pd.Series
        Input series
    method : str
        Normalization method: 'standardize' (z-score) or 'minmax'
        
    Returns
    -------
    pd.Series
        Normalized series
    """
    if method == 'standardize':
        return (series - series.mean()) / series.std()
    elif method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    else:
        raise ValueError(f"Unknown normalization method: {method}")
