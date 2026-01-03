"""
Configuration constants for credit spread mean-reversion analysis.

All numeric parameters, dates, and series specifications live here.
No magic numbers elsewhere in the codebase.
"""

# =============================================================================
# DATA SOURCES
# =============================================================================

# Primary target: Investment-grade credit spread
TARGET_SERIES = "BAMLC0A0CM"  # ICE BofA US Corporate Master OAS

# Observable proxies for market stress / liquidity conditions (HMM features)
OBSERVABLES = {
    "VIX": "VIXCLS",           # Implied volatility (fear gauge)
    "ANFCI": "ANFCI",          # Adjusted National Financial Conditions Index (replaces STLFSI)
    "MOVE": "MOVE",            # Bond market volatility (Merrill Lynch Option Volatility Estimate)
    "TED": None,               # TED spread (3m LIBOR - 3m T-bill) - computed from components
    # Realized volatility of spreads (computed)
    # Bid-ask spread proxy: |ΔS_t| (computed)
}

# Additional macro variables (optional, for robustness checks)
ADDITIONAL = {
    "DGS10": "DGS10",          # 10Y Treasury yield
    "DGS2": "DGS2",            # 2Y Treasury yield (for term spread)
    "DGS3MO": "DGS3MO",        # 3-month T-bill rate (for TED spread)
    "DTWEXBGS": "DTWEXBGS",    # Trade-weighted dollar index (funding conditions)
    # SP500 removed - not used in HMM, incomplete historical data
}

# =============================================================================
# SAMPLE PERIOD
# =============================================================================

START_DATE = "2015-01-01"  # Post-taper period (excludes GFC structural break)
END_DATE = "2025-12-31"    # Use data through end of 2025

# =============================================================================
# HIDDEN MARKOV MODEL PARAMETERS
# =============================================================================

# HMM specification for regime identification
HMM_CONFIG = {
    "n_regimes": 2,                    # Normal vs High Stress (2-regime model for trading)
    "n_iter": 100,                     # EM algorithm iterations
    "random_state": 42,                # Reproducibility
    "covariance_type": "full",         # Full covariance per regime
    "init_params": "stmc",             # Initialize: startprob, transmat, means, covars
    "params": "stmc",                  # Parameters to update
    "algorithm": "viterbi",            # Decoding algorithm (viterbi or map)
}

# Feature engineering for HMM
HMM_FEATURES = {
    "realized_vol_window": 21,         # 21-day realized volatility of spreads
    "abs_change_window": 5,            # 5-day rolling mean of |ΔS_t|
    "stress_interaction": True,        # Include VIX × STLFSI interaction
    "standardize": False,              # Z-score normalize features (DISABLED: raw values work better)
}

# =============================================================================
# LEGACY: KALMAN FILTER PARAMETERS (deprecated, use HMM instead)
# =============================================================================

# Kalman filter specification (kept for backward compatibility)
KALMAN_CONFIG = {
    "n_dim_state": 1,               # Scalar latent liquidity state
    "transition_matrix": 0.95,      # AR(1) persistence
    "transition_cov": 0.1,          # State innovation variance
    "observation_cov": 0.5,         # Observation noise variance
    "initial_state_mean": 0.0,
    "initial_state_cov": 1.0,
}

# =============================================================================
# REGIME DEFINITION
# =============================================================================

# Regime assignment based on smoothed latent state quantiles
REGIME_QUANTILES = {
    "low_stress": 0.33,    # x_t < 33rd percentile = low stress (high liquidity)
    "high_stress": 0.67,   # x_t > 67th percentile = high stress (low liquidity)
}

# Regime labels (2-regime model)
REGIME_NAMES = {
    0: "Normal (Low/Medium Stress)",
    1: "High Stress (Crisis)",
}

# =============================================================================
# MEAN-REVERSION TEST SPECIFICATIONS
# =============================================================================

# Forecast horizons (in trading days)
HORIZONS = [1, 5, 10, 21]  # 1d, 1w, 2w, 1m

# OLS specifications
REGRESSION_CONFIG = {
    "robust_se": True,
    "cov_type": "HAC",        # Newey-West HAC standard errors
    "maxlags": 5,             # For HAC covariance
}

# =============================================================================
# DATA PROCESSING
# =============================================================================

# Standardization (all observables scaled before Kalman filter)
STANDARDIZE = True

# Handle missing data
FILL_METHOD = None  # None = drop NaN rows; alternatives: 'ffill', 'interpolate'

# Minimum observations required for regime
MIN_REGIME_OBS = 30

# =============================================================================
# OUTPUT PATHS
# =============================================================================

DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
RESULTS_TABLES = "results/tables"
RESULTS_FIGURES = "results/figures"

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

RANDOM_SEED = 42
