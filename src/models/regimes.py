"""
Regime assignment based on latent state.

Single responsibility: classify observations into regimes.
No Kalman filtering, no regressions, no plotting.

Regime definition:
    - Low stress (high liquidity): x_t < 33rd percentile
    - Normal: 33rd ≤ x_t ≤ 67th percentile  
    - High stress (low liquidity): x_t > 67th percentile
"""

import pandas as pd
import numpy as np
from typing import Dict

from ..config import REGIME_QUANTILES, REGIME_NAMES, MIN_REGIME_OBS


def assign_regimes(latent_state: pd.Series) -> pd.Series:
    """
    Assign regime labels based on latent state quantiles.
    
    Deterministic assignment using configured quantile thresholds.
    
    Parameters
    ----------
    latent_state : pd.Series
        Smoothed latent state from Kalman filter
        
    Returns
    -------
    pd.Series
        Regime labels (0, 1, 2)
        
    Notes
    -----
    Regime interpretation:
        0 = Low stress (high dealer liquidity capacity)
        1 = Normal conditions
        2 = High stress (low dealer liquidity capacity)
    """
    # Compute quantile thresholds on full sample
    q_low = latent_state.quantile(REGIME_QUANTILES["low_stress"])
    q_high = latent_state.quantile(REGIME_QUANTILES["high_stress"])
    
    print(f"Regime thresholds:")
    print(f"  Low/Normal boundary:  {q_low:.3f} ({REGIME_QUANTILES['low_stress']:.0%} quantile)")
    print(f"  Normal/High boundary: {q_high:.3f} ({REGIME_QUANTILES['high_stress']:.0%} quantile)")
    
    # Assign regimes
    regimes = pd.Series(1, index=latent_state.index, name="regime")  # Default = Normal
    regimes[latent_state < q_low] = 0   # Low stress
    regimes[latent_state > q_high] = 2  # High stress
    
    return regimes


def summarize_regimes(regimes: pd.Series) -> pd.DataFrame:
    """
    Summarize regime distribution.
    
    Parameters
    ----------
    regimes : pd.Series
        Regime labels
        
    Returns
    -------
    pd.DataFrame
        Summary table with counts, frequencies, names
    """
    counts = regimes.value_counts().sort_index()
    freqs = regimes.value_counts(normalize=True).sort_index()
    
    summary = pd.DataFrame({
        "Regime": [REGIME_NAMES[r] for r in counts.index],
        "Count": counts.values,
        "Frequency": freqs.values,
    }, index=counts.index)
    
    return summary


def validate_regime_counts(regimes: pd.Series) -> None:
    """
    Validate that each regime has sufficient observations.
    
    Parameters
    ----------
    regimes : pd.Series
        Regime labels
        
    Raises
    ------
    ValueError
        If any regime has too few observations
    """
    counts = regimes.value_counts()
    
    insufficient = counts[counts < MIN_REGIME_OBS]
    
    if len(insufficient) > 0:
        raise ValueError(
            f"Some regimes have fewer than {MIN_REGIME_OBS} observations:\n{insufficient}\n"
            "Consider adjusting REGIME_QUANTILES or extending sample period."
        )
    
    print(f"All regimes have sufficient observations (min: {counts.min()})")


def get_regime_indices(regimes: pd.Series) -> Dict[int, pd.DatetimeIndex]:
    """
    Get date indices for each regime.
    
    Useful for splitting data by regime.
    
    Parameters
    ----------
    regimes : pd.Series
        Regime labels
        
    Returns
    -------
    dict
        Mapping from regime label to DatetimeIndex
    """
    regime_indices = {}
    
    for regime_label in sorted(regimes.unique()):
        regime_indices[regime_label] = regimes[regimes == regime_label].index
    
    return regime_indices


def compute_regime_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """
    Compute empirical transition probabilities between regimes.
    
    P[i,j] = probability of transitioning from regime i to regime j.
    
    Parameters
    ----------
    regimes : pd.Series
        Regime labels
        
    Returns
    -------
    pd.DataFrame
        Transition matrix (3 x 3)
    """
    # Get transitions
    regime_t = regimes.iloc[:-1].values
    regime_t1 = regimes.iloc[1:].values
    
    # Count transitions
    n_regimes = len(REGIME_NAMES)
    transitions = np.zeros((n_regimes, n_regimes))
    
    for i in range(len(regime_t)):
        from_regime = int(regime_t[i])
        to_regime = int(regime_t1[i])
        transitions[from_regime, to_regime] += 1
    
    # Normalize to probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_probs = np.divide(
        transitions, 
        row_sums, 
        out=np.zeros_like(transitions), 
        where=row_sums != 0
    )
    
    # Format as DataFrame
    df = pd.DataFrame(
        transition_probs,
        index=[f"Regime {i}" for i in range(n_regimes)],
        columns=[f"Regime {i}" for i in range(n_regimes)]
    )
    
    return df


def compute_regime_persistence(regimes: pd.Series) -> Dict[int, float]:
    """
    Compute persistence (autocorrelation) of each regime.
    
    Parameters
    ----------
    regimes : pd.Series
        Regime labels
        
    Returns
    -------
    dict
        Mapping from regime label to persistence probability
    """
    transition_matrix = compute_regime_transition_matrix(regimes)
    
    # Diagonal = P(stay in same regime)
    persistence = {}
    for i in range(len(REGIME_NAMES)):
        persistence[i] = transition_matrix.iloc[i, i]
    
    return persistence


if __name__ == "__main__":
    # Example usage
    from ..data.fetch_fred import load_raw_data
    from ..data.clean import clean_data
    from ..data.transforms import prepare_observables
    from .state_space import estimate_latent_state
    
    print("Loading data and estimating latent state...")
    raw = load_raw_data()
    clean = clean_data(raw)
    obs = prepare_observables(clean)
    latent_state, _ = estimate_latent_state(obs)
    
    print("\nAssigning regimes...")
    regimes = assign_regimes(latent_state)
    
    print("\nRegime summary:")
    print(summarize_regimes(regimes))
    
    print("\nTransition matrix:")
    print(compute_regime_transition_matrix(regimes))
    
    print("\nPersistence:")
    print(compute_regime_persistence(regimes))
