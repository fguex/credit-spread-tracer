"""
Mean-reversion regression tests.

Single responsibility: test conditional mean reversion by regime.
No data fetching, no state-space estimation, no regime assignment.

Model specification:
    ΔS_{t+h} = α + β S_t + ε_{t+h}

Test H0: β = 0 (no mean reversion) vs H1: β < 0 (mean reversion)

Runs separately for each regime and horizon.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from typing import Dict, List, Tuple

from ..config import HORIZONS, REGRESSION_CONFIG, MIN_REGIME_OBS


def run_mean_reversion_test(
    spread: pd.Series,
    horizon: int,
    robust_se: bool = True,
    cov_type: str = "HAC",
    maxlags: int = 5
) -> Dict:
    """
    Test mean reversion at specified horizon.
    
    Model: ΔS_{t+h} = α + β S_t + ε_{t+h}
    
    Parameters
    ----------
    spread : pd.Series
        Credit spread level (in bps)
    horizon : int
        Forecast horizon in days
    robust_se : bool
        Whether to use robust standard errors
    cov_type : str
        Covariance estimator: 'HAC' (Newey-West) or 'HC3'
    maxlags : int
        Maximum lags for HAC covariance
        
    Returns
    -------
    dict
        Regression results with keys:
        - 'alpha': intercept
        - 'beta': mean-reversion coefficient
        - 'se_alpha', 'se_beta': standard errors
        - 't_alpha', 't_beta': t-statistics
        - 'p_alpha', 'p_beta': p-values
        - 'r_squared': R²
        - 'n_obs': number of observations
        - 'horizon': forecast horizon
    """
    # Construct target variable: ΔS_{t+h} = S_{t+h} - S_t
    delta_S = spread.shift(-horizon) - spread
    
    # Construct regressor: S_t
    X = pd.DataFrame({"spread": spread})
    y = delta_S
    
    # Align and drop NaN
    data = pd.concat([y.rename("delta_S"), X], axis=1).dropna()
    
    if len(data) < 10:
        raise ValueError(f"Insufficient observations after alignment: {len(data)}")
    
    # Add constant
    X_reg = sm.add_constant(data["spread"])
    y_reg = data["delta_S"]
    
    # Run OLS
    model = OLS(y_reg, X_reg)
    
    if robust_se:
        if cov_type == "HAC":
            results = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        else:
            results = model.fit(cov_type=cov_type)
    else:
        results = model.fit()
    
    # Extract results
    params = results.params
    se = results.bse
    tvalues = results.tvalues
    pvalues = results.pvalues
    
    output = {
        "horizon": horizon,
        "n_obs": int(results.nobs),
        "alpha": params["const"],
        "beta": params["spread"],
        "se_alpha": se["const"],
        "se_beta": se["spread"],
        "t_alpha": tvalues["const"],
        "t_beta": tvalues["spread"],
        "p_alpha": pvalues["const"],
        "p_beta": pvalues["spread"],
        "r_squared": results.rsquared,
        "r_squared_adj": results.rsquared_adj,
    }
    
    return output


def run_regime_conditional_tests(
    spread: pd.Series,
    regimes: pd.Series,
    horizons: List[int]
) -> pd.DataFrame:
    """
    Run mean-reversion tests conditional on regime.
    
    For each regime and horizon, tests:
        ΔS_{t+h} = α + β S_t + ε_{t+h}
    
    Parameters
    ----------
    spread : pd.Series
        Credit spread level
    regimes : pd.Series
        Regime labels (0, 1, 2)
    horizons : list
        Forecast horizons to test
        
    Returns
    -------
    pd.DataFrame
        Results table with columns:
        - regime, horizon, n_obs, alpha, beta, se_beta, t_beta, p_beta, r_squared
    """
    results_list = []
    
    for regime_label in sorted(regimes.unique()):
        # Filter to regime
        regime_mask = regimes == regime_label
        spread_regime = spread[regime_mask]
        
        # Check sufficient observations
        if len(spread_regime) < MIN_REGIME_OBS:
            print(f"Warning: Regime {regime_label} has only {len(spread_regime)} obs < {MIN_REGIME_OBS}, skipping")
            continue
        
        for h in horizons:
            try:
                result = run_mean_reversion_test(
                    spread_regime,
                    horizon=h,
                    robust_se=REGRESSION_CONFIG["robust_se"],
                    cov_type=REGRESSION_CONFIG["cov_type"],
                    maxlags=REGRESSION_CONFIG["maxlags"]
                )
                result["regime"] = int(regime_label)
                results_list.append(result)
                
            except Exception as e:
                print(f"Error in regime {regime_label}, horizon {h}: {e}")
                continue
    
    # Combine into DataFrame
    df = pd.DataFrame(results_list)
    
    # Reorder columns
    col_order = [
        "regime", "horizon", "n_obs",
        "alpha", "beta", "se_beta", "t_beta", "p_beta",
        "r_squared"
    ]
    df = df[col_order]
    
    return df


def compute_half_life(beta: float) -> float:
    """
    Compute half-life of mean reversion.
    
    From AR(1) process: half-life = -ln(2) / ln(1 + β)
    
    Parameters
    ----------
    beta : float
        Mean-reversion coefficient (should be < 0)
        
    Returns
    -------
    float
        Half-life in days. Returns inf if β ≥ 0.
    """
    if beta >= 0:
        return np.inf
    
    return -np.log(2) / np.log(1 + beta)


def interpret_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interpretation columns to results table.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_regime_conditional_tests()
        
    Returns
    -------
    pd.DataFrame
        Results with added columns:
        - mean_reverting: bool (β < 0 and p < 0.05)
        - half_life: float (days)
        - significance: str ('***', '**', '*', 'ns')
    """
    df = results_df.copy()
    
    # Mean reversion detected?
    df["mean_reverting"] = (df["beta"] < 0) & (df["p_beta"] < 0.05)
    
    # Half-life
    df["half_life"] = df["beta"].apply(compute_half_life)
    
    # Significance stars
    def significance_stars(p):
        if p < 0.01:
            return "***"
        elif p < 0.05:
            return "**"
        elif p < 0.10:
            return "*"
        else:
            return "ns"
    
    df["significance"] = df["p_beta"].apply(significance_stars)
    
    return df


def run_unconditional_test(
    spread: pd.Series,
    horizons: List[int]
) -> pd.DataFrame:
    """
    Run unconditional mean-reversion tests (baseline).
    
    Same model as conditional tests but using full sample.
    
    Parameters
    ----------
    spread : pd.Series
        Credit spread level
    horizons : list
        Forecast horizons
        
    Returns
    -------
    pd.DataFrame
        Results table (same format as conditional tests)
    """
    results_list = []
    
    for h in horizons:
        try:
            result = run_mean_reversion_test(
                spread,
                horizon=h,
                robust_se=REGRESSION_CONFIG["robust_se"],
                cov_type=REGRESSION_CONFIG["cov_type"],
                maxlags=REGRESSION_CONFIG["maxlags"]
            )
            result["regime"] = "Unconditional"
            results_list.append(result)
            
        except Exception as e:
            print(f"Error in unconditional test, horizon {h}: {e}")
            continue
    
    df = pd.DataFrame(results_list)
    return df


if __name__ == "__main__":
    # Example usage
    from ..data.fetch_fred import load_raw_data
    from ..data.clean import clean_data
    from ..data.transforms import prepare_observables
    from .state_space import estimate_latent_state
    from .regimes import assign_regimes
    
    print("Loading data and running full pipeline...")
    raw = load_raw_data()
    clean = clean_data(raw)
    
    # Get spread
    spread = clean["spread"]
    
    # Estimate latent state and assign regimes
    obs = prepare_observables(clean)
    latent_state, _ = estimate_latent_state(obs)
    regimes = assign_regimes(latent_state)
    
    # Align spread and regimes
    data = pd.concat([spread, regimes], axis=1).dropna()
    
    print("\nRunning unconditional tests...")
    uncond_results = run_unconditional_test(data["spread"], HORIZONS)
    print(interpret_results(uncond_results))
    
    print("\nRunning conditional tests...")
    cond_results = run_regime_conditional_tests(
        data["spread"],
        data["regime"],
        HORIZONS
    )
    print(interpret_results(cond_results))
