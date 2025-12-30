"""
State-space model for latent liquidity/balance-sheet capacity.

Single responsibility: estimate latent state via Kalman filter.
No regime assignment, no regressions, no plotting.

Model specification:
    x_t = φ x_{t-1} + η_t,  η_t ~ N(0, Q)    [state equation]
    y_t = H x_t + ε_t,      ε_t ~ N(0, R)    [observation equation]

where:
    x_t = latent liquidity/balance-sheet capacity
    y_t = [VIX, STLFSI, |ΔS_t|]' = observable proxies
"""

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from typing import Tuple, Optional

from ..config import KALMAN_CONFIG


def build_kalman_filter(n_obs: int) -> KalmanFilter:
    """
    Construct Kalman filter object with configured parameters.
    
    Parameters
    ----------
    n_obs : int
        Number of observable series
        
    Returns
    -------
    KalmanFilter
        Configured filter ready for estimation
    """
    # Scalar latent state: x_t ∈ ℝ
    n_dim_state = KALMAN_CONFIG["n_dim_state"]
    
    # AR(1) transition: x_t = φ x_{t-1} + η_t
    transition_matrix = np.array([[KALMAN_CONFIG["transition_matrix"]]])
    transition_cov = np.array([[KALMAN_CONFIG["transition_cov"]]])
    
    # Observation equation: y_t = H x_t + ε_t
    # Each observable loads on the single latent factor
    observation_matrix = np.ones((n_obs, n_dim_state))
    observation_cov = KALMAN_CONFIG["observation_cov"] * np.eye(n_obs)
    
    # Initial distribution: x_0 ~ N(μ_0, Σ_0)
    initial_state_mean = np.array([KALMAN_CONFIG["initial_state_mean"]])
    initial_state_cov = np.array([[KALMAN_CONFIG["initial_state_cov"]]])
    
    kf = KalmanFilter(
        n_dim_state=n_dim_state,
        n_dim_obs=n_obs,
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_cov,
        observation_covariance=observation_cov,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_cov,
    )
    
    return kf


def estimate_latent_state(
    observables: pd.DataFrame,
    return_covariance: bool = False
) -> Tuple[pd.Series, Optional[np.ndarray]]:
    """
    Estimate latent liquidity state via Kalman smoother.
    
    Uses fixed parameters from config (no EM estimation).
    
    Parameters
    ----------
    observables : pd.DataFrame
        Standardized observable matrix (T x K)
        Each column is one observable series
    return_covariance : bool
        Whether to return state covariance matrix
        
    Returns
    -------
    latent_state : pd.Series
        Smoothed latent state x_t (T x 1)
    state_cov : np.ndarray, optional
        Smoothed state covariance Σ_t (T x 1 x 1)
        
    Notes
    -----
    - Uses Rauch-Tung-Striebel smoother (forward-backward pass)
    - Smoothed estimates use full sample information
    - For out-of-sample forecasting, use filter() instead of smooth()
    """
    # Validate input
    if observables.isna().any().any():
        raise ValueError("Observables contain NaN values. Clean data first.")
    
    n_obs = observables.shape[1]
    
    # Build filter
    kf = build_kalman_filter(n_obs)
    
    # Run smoother
    state_means, state_covs = kf.smooth(observables.values)
    
    # Extract scalar state (first dimension)
    latent_state = pd.Series(
        state_means[:, 0],
        index=observables.index,
        name="latent_state"
    )
    
    print(f"Estimated latent state for {len(latent_state)} periods")
    print(f"State range: [{latent_state.min():.3f}, {latent_state.max():.3f}]")
    print(f"State mean: {latent_state.mean():.3f}, std: {latent_state.std():.3f}")
    
    if return_covariance:
        return latent_state, state_covs
    else:
        return latent_state, None


def filter_latent_state(observables: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
    """
    Filter (not smooth) latent state.
    
    Uses only information available up to time t.
    Use for out-of-sample analysis or real-time applications.
    
    Parameters
    ----------
    observables : pd.DataFrame
        Observable matrix
        
    Returns
    -------
    latent_state : pd.Series
        Filtered state estimates
    state_cov : np.ndarray
        Filtered state covariances
    """
    n_obs = observables.shape[1]
    kf = build_kalman_filter(n_obs)
    
    state_means, state_covs = kf.filter(observables.values)
    
    latent_state = pd.Series(
        state_means[:, 0],
        index=observables.index,
        name="latent_state_filtered"
    )
    
    return latent_state, state_covs


def compute_log_likelihood(observables: pd.DataFrame) -> float:
    """
    Compute log-likelihood of data under model.
    
    Useful for model comparison or parameter tuning.
    
    Parameters
    ----------
    observables : pd.DataFrame
        Observable matrix
        
    Returns
    -------
    float
        Log-likelihood value
    """
    n_obs = observables.shape[1]
    kf = build_kalman_filter(n_obs)
    
    loglik = kf.loglikelihood(observables.values)
    
    print(f"Log-likelihood: {loglik:.2f}")
    return loglik


def estimate_kalman_parameters_em(
    observables: pd.DataFrame,
    n_iter: int = 50,
    em_vars: str = 'transition_covariance,observation_covariance',
    tolerance: float = 1e-4,
    verbose: bool = True
) -> Tuple[KalmanFilter, pd.Series, dict]:
    """
    Estimate Kalman filter parameters via Expectation-Maximization (EM).
    
    The EM algorithm iteratively:
    1. E-step: Compute expected sufficient statistics given current parameters
    2. M-step: Update parameters to maximize expected log-likelihood
    
    Parameters
    ----------
    observables : pd.DataFrame
        Observable matrix (T x K)
    n_iter : int, default=50
        Maximum number of EM iterations
    em_vars : str, default='transition_covariance,observation_covariance'
        Which parameters to estimate. Options:
        - 'transition_covariance': Estimate Q (state noise)
        - 'observation_covariance': Estimate R (observation noise)
        - 'transition_matrices': Estimate φ (AR coefficient)
        - 'observation_matrices': Estimate H (factor loadings)
        - 'initial_state_mean': Estimate μ_0
        - 'initial_state_covariance': Estimate Σ_0
        Separate multiple with commas
    tolerance : float, default=1e-4
        Convergence threshold (change in log-likelihood)
    verbose : bool, default=True
        Print iteration details
        
    Returns
    -------
    kf_fitted : KalmanFilter
        Fitted Kalman filter with estimated parameters
    latent_state : pd.Series
        Smoothed latent state using estimated parameters
    params : dict
        Dictionary of estimated parameters
        
    Notes
    -----
    - Uses pykalman's built-in EM algorithm
    - Fixes observation matrix H = [1, 1, ..., 1]' (all observables load on factor)
    - Can estimate subset of parameters while keeping others fixed
    - Convergence is not guaranteed; check log-likelihood progression
    
    References
    ----------
    - Shumway & Stoffer (2017): Time Series Analysis and Its Applications
    - Durbin & Koopman (2012): Time Series Analysis by State Space Methods
    
    Examples
    --------
    >>> # Estimate only noise covariances (most common)
    >>> kf, state, params = estimate_kalman_parameters_em(
    ...     observables, 
    ...     em_vars='transition_covariance,observation_covariance'
    ... )
    >>> 
    >>> # Estimate all parameters
    >>> kf, state, params = estimate_kalman_parameters_em(
    ...     observables,
    ...     em_vars='all'
    ... )
    """
    if observables.isna().any().any():
        raise ValueError("Observables contain NaN values. Clean data first.")
    
    n_obs = observables.shape[1]
    n_dim_state = KALMAN_CONFIG["n_dim_state"]
    
    # Initialize with config parameters
    transition_matrix = np.array([[KALMAN_CONFIG["transition_matrix"]]])
    transition_cov = np.array([[KALMAN_CONFIG["transition_cov"]]])
    observation_matrix = np.ones((n_obs, n_dim_state))
    observation_cov = KALMAN_CONFIG["observation_cov"] * np.eye(n_obs)
    initial_state_mean = np.array([KALMAN_CONFIG["initial_state_mean"]])
    initial_state_cov = np.array([[KALMAN_CONFIG["initial_state_cov"]]])
    
    # Create initial Kalman filter
    kf = KalmanFilter(
        n_dim_state=n_dim_state,
        n_dim_obs=n_obs,
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=transition_cov,
        observation_covariance=observation_cov,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_cov,
    )
    
    if verbose:
        print("=" * 80)
        print("KALMAN FILTER EM PARAMETER ESTIMATION")
        print("=" * 80)
        print(f"Observations: {len(observables)} x {n_obs}")
        print(f"State dimension: {n_dim_state}")
        print(f"Max iterations: {n_iter}")
        print(f"Estimating: {em_vars}")
        print(f"Convergence tolerance: {tolerance}")
        
        # Initial log-likelihood
        initial_loglik = kf.loglikelihood(observables.values)
        print(f"\nInitial log-likelihood: {initial_loglik:.4f}")
        print("\nIteration | Log-Likelihood | Improvement")
        print("-" * 50)
    
    # Run EM algorithm
    kf_fitted = kf.em(
        X=observables.values,
        n_iter=n_iter,
        em_vars=em_vars.split(',') if em_vars != 'all' else None,
    )
    
    # Final log-likelihood
    final_loglik = kf_fitted.loglikelihood(observables.values)
    
    if verbose:
        print(f"\nFinal log-likelihood: {final_loglik:.4f}")
        print(f"Total improvement: {final_loglik - initial_loglik:.4f}")
        print("\n" + "=" * 80)
        print("ESTIMATED PARAMETERS")
        print("=" * 80)
    
    # Extract estimated parameters
    params = {}
    
    if kf_fitted.transition_matrices is not None:
        phi = kf_fitted.transition_matrices[0, 0]
        params['transition_matrix'] = phi
        if verbose:
            print(f"φ (AR coefficient): {phi:.4f}")
            if abs(phi) >= 1:
                print("  ⚠ WARNING: |φ| >= 1, state may be non-stationary!")
    
    if kf_fitted.transition_covariance is not None:
        Q = kf_fitted.transition_covariance[0, 0]
        params['transition_covariance'] = Q
        if verbose:
            print(f"Q (state noise variance): {Q:.6f}")
    
    if kf_fitted.observation_matrices is not None:
        H = kf_fitted.observation_matrices
        params['observation_matrices'] = H
        if verbose:
            print(f"H (factor loadings): {H.flatten()}")
    
    if kf_fitted.observation_covariance is not None:
        R = kf_fitted.observation_covariance
        params['observation_covariance'] = R
        if verbose:
            print(f"R (observation noise, diagonal):")
            for i in range(n_obs):
                print(f"  Observable {i+1}: {R[i,i]:.6f}")
    
    if kf_fitted.initial_state_mean is not None:
        mu0 = kf_fitted.initial_state_mean[0]
        params['initial_state_mean'] = mu0
        if verbose:
            print(f"μ₀ (initial state mean): {mu0:.4f}")
    
    if kf_fitted.initial_state_covariance is not None:
        Sigma0 = kf_fitted.initial_state_covariance[0, 0]
        params['initial_state_covariance'] = Sigma0
        if verbose:
            print(f"Σ₀ (initial state variance): {Sigma0:.6f}")
    
    # Compute smoothed state with estimated parameters
    state_means, _ = kf_fitted.smooth(observables.values)
    latent_state = pd.Series(
        state_means[:, 0],
        index=observables.index,
        name="latent_state_em"
    )
    
    if verbose:
        print("\n" + "=" * 80)
        print("LATENT STATE STATISTICS (EM-estimated)")
        print("=" * 80)
        print(f"Mean: {latent_state.mean():.4f}")
        print(f"Std:  {latent_state.std():.4f}")
        print(f"Min:  {latent_state.min():.4f}")
        print(f"Max:  {latent_state.max():.4f}")
        
        # Compare with fixed-parameter state
        kf_fixed = build_kalman_filter(n_obs)
        state_fixed, _ = kf_fixed.smooth(observables.values)
        state_fixed = pd.Series(state_fixed[:, 0], index=observables.index)
        correlation = latent_state.corr(state_fixed)
        print(f"\nCorrelation with fixed-parameter state: {correlation:.4f}")
    
    return kf_fitted, latent_state, params


def compare_kalman_specifications(
    observables: pd.DataFrame,
    specs: list = None
) -> pd.DataFrame:
    """
    Compare different Kalman filter specifications.
    
    Parameters
    ----------
    observables : pd.DataFrame
        Observable matrix
    specs : list of dict, optional
        List of specifications to compare. Each dict contains em_vars.
        If None, uses default comparison set.
        
    Returns
    -------
    pd.DataFrame
        Comparison table with log-likelihoods and parameters
        
    Examples
    --------
    >>> comparison = compare_kalman_specifications(observables)
    >>> print(comparison.sort_values('log_likelihood', ascending=False))
    """
    if specs is None:
        specs = [
            {'name': 'Fixed (Config)', 'em_vars': None},
            {'name': 'Estimate Q only', 'em_vars': 'transition_covariance'},
            {'name': 'Estimate R only', 'em_vars': 'observation_covariance'},
            {'name': 'Estimate Q and R', 'em_vars': 'transition_covariance,observation_covariance'},
            {'name': 'Estimate φ, Q, R', 'em_vars': 'transition_matrices,transition_covariance,observation_covariance'},
        ]
    
    results = []
    n_obs = observables.shape[1]
    
    print("=" * 80)
    print("COMPARING KALMAN FILTER SPECIFICATIONS")
    print("=" * 80)
    
    for spec in specs:
        name = spec['name']
        em_vars = spec['em_vars']
        
        print(f"\n{name}...")
        
        if em_vars is None:
            # Fixed parameters
            kf = build_kalman_filter(n_obs)
            loglik = kf.loglikelihood(observables.values)
            params = {
                'transition_matrix': KALMAN_CONFIG["transition_matrix"],
                'transition_covariance': KALMAN_CONFIG["transition_cov"],
                'observation_covariance': KALMAN_CONFIG["observation_cov"],
            }
        else:
            # EM estimation
            kf, _, params = estimate_kalman_parameters_em(
                observables, 
                em_vars=em_vars,
                verbose=False,
                n_iter=30
            )
            loglik = kf.loglikelihood(observables.values)
        
        result = {
            'specification': name,
            'log_likelihood': loglik,
            **params
        }
        results.append(result)
        
        print(f"  Log-likelihood: {loglik:.2f}")
    
    comparison = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(comparison.to_string(index=False))
    
    return comparison


if __name__ == "__main__":
    # Example usage
    from ..data.fetch_fred import load_raw_data
    from ..data.clean import clean_data
    from ..data.transforms import prepare_observables
    
    print("Loading data...")
    raw = load_raw_data()
    clean = clean_data(raw)
    obs = prepare_observables(clean)
    
    print("\nEstimating latent state...")
    latent_state, _ = estimate_latent_state(obs)
    
    print("\nLatent state summary:")
    print(latent_state.describe())
