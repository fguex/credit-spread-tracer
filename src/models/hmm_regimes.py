"""
Hidden Markov Model for market regime identification.

This module uses a Gaussian HMM to identify discrete market regimes
(low stress, normal, high stress) based on observable stress indicators.

Single responsibility: HMM estimation and regime inference.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings

from src.config import HMM_CONFIG


def prepare_hmm_features(
    data: pd.DataFrame,
    feature_cols: list[str],
    standardize: bool = True
) -> tuple[np.ndarray, StandardScaler | None]:
    """
    Prepare features for HMM estimation.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with feature columns
    feature_cols : list[str]
        Column names to use as HMM features
    standardize : bool, default=True
        Whether to standardize features (recommended for HMM)
    
    Returns
    -------
    features : np.ndarray
        Prepared feature matrix (n_samples, n_features)
    scaler : StandardScaler or None
        Fitted scaler (if standardize=True), else None
    
    Notes
    -----
    - Removes any rows with NaN values
    - Standardization helps HMM convergence and interpretation
    """
    # Extract features
    features = data[feature_cols].copy()
    
    # Drop NaN
    features = features.dropna()
    
    if len(features) == 0:
        raise ValueError("No valid observations after removing NaN")
    
    # Convert to numpy
    X = features.values
    
    # Standardize
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, scaler


def fit_gaussian_hmm(
    features: np.ndarray,
    n_regimes: int = 3,
    n_iter: int = 100,
    random_state: int = 42,
    covariance_type: str = "full",
    verbose: bool = False
) -> hmm.GaussianHMM:
    """
    Fit Gaussian Hidden Markov Model to features.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    n_regimes : int, default=3
        Number of hidden states (regimes)
    n_iter : int, default=100
        Maximum EM iterations
    random_state : int, default=42
        Random seed for reproducibility
    covariance_type : str, default="full"
        Covariance structure: "spherical", "diag", "full", "tied"
    verbose : bool, default=False
        Print convergence information
    
    Returns
    -------
    model : GaussianHMM
        Fitted HMM model
    
    Notes
    -----
    - Uses EM algorithm to estimate transition matrix and emission parameters
    - "full" covariance allows different correlations per regime
    - Convergence not guaranteed; check model.monitor_.converged
    """
    # Initialize model
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=verbose
    )
    
    # Fit model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model.fit(features)
    
    # Fix degenerate transition matrix (rows summing to 0)
    transmat = model.transmat_.copy()
    for i in range(len(transmat)):
        if transmat[i].sum() == 0:
            # State never observed - set uniform transition
            transmat[i] = 1.0 / len(transmat)
    model.transmat_ = transmat
    
    if verbose:
        print(f"Converged: {model.monitor_.converged}")
        # Skip scoring if model is degenerate
        try:
            print(f"Final log-likelihood: {model.score(features):.2f}")
        except:
            print("Warning: Could not compute log-likelihood (possibly degenerate model)")
    
    return model


def decode_regimes(
    model: hmm.GaussianHMM,
    features: np.ndarray,
    algorithm: str = "viterbi"
) -> np.ndarray:
    """
    Decode most likely sequence of regimes using fitted HMM.
    
    Parameters
    ----------
    model : GaussianHMM
        Fitted HMM model
    features : np.ndarray
        Feature matrix (n_samples, n_features)
    algorithm : str, default="viterbi"
        Decoding algorithm: "viterbi" (most likely path) or "map" (posterior mode)
    
    Returns
    -------
    regimes : np.ndarray
        Regime labels (0, 1, 2, ..., n_regimes-1)
    
    Notes
    -----
    - Viterbi finds globally optimal state sequence
    - MAP (posterior mode) finds locally optimal state at each time
    - Viterbi generally preferred for regime identification
    """
    if algorithm == "viterbi":
        regimes = model.predict(features)
    elif algorithm == "map":
        posteriors = model.predict_proba(features)
        regimes = posteriors.argmax(axis=1)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return regimes


def sort_regimes_by_stress(
    regimes: np.ndarray,
    features: np.ndarray,
    stress_col_idx: int = 0
) -> tuple[np.ndarray, dict]:
    """
    Relabel regimes by ascending stress level.
    
    Parameters
    ----------
    regimes : np.ndarray
        Original regime labels from HMM
    features : np.ndarray
        Feature matrix
    stress_col_idx : int, default=0
        Column index of primary stress indicator (e.g., VIX)
    
    Returns
    -------
    sorted_regimes : np.ndarray
        Relabeled regimes: 0=Low Stress, 1=Normal, 2=High Stress
    mapping : dict
        Mapping from old to new labels
    
    Notes
    -----
    - HMM labels are arbitrary; we sort by mean stress level
    - Assumes higher stress indicator values = higher stress
    """
    n_regimes = len(np.unique(regimes))
    
    # Compute mean stress per regime
    regime_means = []
    for r in range(n_regimes):
        mask = regimes == r
        mean_stress = features[mask, stress_col_idx].mean()
        regime_means.append((r, mean_stress))
    
    # Sort by mean stress
    regime_means.sort(key=lambda x: x[1])
    
    # Create mapping: old label -> new label
    mapping = {old: new for new, (old, _) in enumerate(regime_means)}
    
    # Relabel
    sorted_regimes = np.array([mapping[r] for r in regimes])
    
    return sorted_regimes, mapping


def estimate_regimes_hmm(
    data: pd.DataFrame,
    feature_cols: list[str],
    n_regimes: int = None,
    n_iter: int = None,
    random_state: int = None,
    covariance_type: str = None,
    algorithm: str = None,
    stress_col: str = None,
    verbose: bool = False
) -> tuple[pd.Series, hmm.GaussianHMM, StandardScaler]:
    """
    End-to-end HMM regime estimation.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with feature columns
    feature_cols : list[str]
        Column names to use as HMM features
    n_regimes : int, optional
        Number of regimes (default from config)
    n_iter : int, optional
        EM iterations (default from config)
    random_state : int, optional
        Random seed (default from config)
    covariance_type : str, optional
        Covariance structure (default from config)
    algorithm : str, optional
        Decoding algorithm (default from config)
    stress_col : str, optional
        Primary stress indicator for sorting regimes (default: first feature)
    verbose : bool, default=False
        Print convergence info
    
    Returns
    -------
    regimes : pd.Series
        Regime labels aligned with original data index
    model : GaussianHMM
        Fitted HMM model
    scaler : StandardScaler
        Fitted feature scaler
    
    Example
    -------
    >>> features = ['vix', 'stlfsi', 'realized_vol']
    >>> regimes, model, scaler = estimate_regimes_hmm(data, features)
    >>> print(regimes.value_counts())
    """
    # Load defaults from config
    n_regimes = n_regimes or HMM_CONFIG["n_regimes"]
    n_iter = n_iter or HMM_CONFIG["n_iter"]
    random_state = random_state or HMM_CONFIG["random_state"]
    covariance_type = covariance_type or HMM_CONFIG["covariance_type"]
    algorithm = algorithm or HMM_CONFIG["algorithm"]
    
    # Prepare features
    X, scaler = prepare_hmm_features(data, feature_cols, standardize=True)
    
    # Fit HMM
    model = fit_gaussian_hmm(
        X, 
        n_regimes=n_regimes,
        n_iter=n_iter,
        random_state=random_state,
        covariance_type=covariance_type,
        verbose=verbose
    )
    
    # Decode regimes
    regimes = decode_regimes(model, X, algorithm=algorithm)
    
    # Sort regimes by stress level
    stress_col_idx = 0 if stress_col is None else feature_cols.index(stress_col)
    regimes_sorted, mapping = sort_regimes_by_stress(regimes, X, stress_col_idx)
    
    # Align with original data (account for dropped NaN rows)
    valid_idx = data[feature_cols].dropna().index
    regimes_series = pd.Series(regimes_sorted, index=valid_idx, name="regime")
    
    if verbose:
        print(f"\nRegime distribution:")
        print(regimes_series.value_counts().sort_index())
        print(f"\nRegime relabeling mapping: {mapping}")
    
    return regimes_series, model, scaler


def compute_regime_statistics(
    regimes: pd.Series,
    features: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute descriptive statistics by regime.
    
    Parameters
    ----------
    regimes : pd.Series
        Regime labels
    features : pd.DataFrame
        Feature dataframe
    
    Returns
    -------
    stats : pd.DataFrame
        Mean and std of each feature by regime
    """
    stats_list = []
    
    for regime in sorted(regimes.unique()):
        mask = regimes == regime
        regime_features = features.loc[mask]
        
        stats_dict = {
            "regime": regime,
            "count": len(regime_features),
            "frequency": len(regime_features) / len(regimes)
        }
        
        # Mean and std for each feature
        for col in features.columns:
            stats_dict[f"{col}_mean"] = regime_features[col].mean()
            stats_dict[f"{col}_std"] = regime_features[col].std()
        
        stats_list.append(stats_dict)
    
    return pd.DataFrame(stats_list)


if __name__ == "__main__":
    # Example usage
    print("HMM Regime Identification Module")
    print("=" * 60)
    
    # Simulate data
    np.random.seed(42)
    n = 500
    
    # Three regimes with different means
    true_regimes = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
    
    # Features with regime-dependent distributions
    vix = np.where(true_regimes == 0, np.random.normal(15, 3, n),
           np.where(true_regimes == 1, np.random.normal(20, 4, n),
                    np.random.normal(30, 5, n)))
    
    stress = np.where(true_regimes == 0, np.random.normal(-1, 0.5, n),
             np.where(true_regimes == 1, np.random.normal(0, 0.8, n),
                      np.random.normal(2, 1, n)))
    
    data = pd.DataFrame({
        "vix": vix,
        "stress": stress,
        "true_regime": true_regimes
    })
    
    # Estimate regimes
    features = ["vix", "stress"]
    regimes, model, scaler = estimate_regimes_hmm(
        data, features, verbose=True
    )
    
    # Accuracy
    accuracy = (regimes == data.loc[regimes.index, "true_regime"]).mean()
    print(f"\nRecovery accuracy: {accuracy:.1%}")
    
    # Transition matrix
    print(f"\nEstimated transition matrix:")
    print(model.transmat_)
