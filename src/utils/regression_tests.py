"""
Regression and statistical testing utilities for credit spread analysis.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy import stats


def run_conditional_regression(y, X, robust_se=True, cov_type='HC3'):
    """
    Run OLS regression with optional robust standard errors.
    
    Parameters
    ----------
    y : pd.Series or np.ndarray
        Dependent variable
    X : pd.DataFrame or np.ndarray
        Independent variables (without constant)
    robust_se : bool
        Whether to compute robust standard errors
    cov_type : str
        Covariance estimator type for robust SE (HC0, HC1, HC2, HC3)
        
    Returns
    -------
    dict
        Dictionary containing regression results:
        - 'model': fitted OLS model
        - 'summary': regression summary
        - 'coef': coefficient estimates
        - 'se': standard errors
        - 'pvalues': p-values
        - 'rsquared': R-squared
        - 'nobs': number of observations
    """
    # Prepare data
    if isinstance(X, pd.Series):
        X = X.to_frame()
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    # Align indices if pandas objects
    if isinstance(y, pd.Series) and isinstance(X_with_const, pd.DataFrame):
        data = pd.concat([y, X_with_const], axis=1).dropna()
        y_clean = data.iloc[:, 0]
        X_clean = data.iloc[:, 1:]
    else:
        y_clean = y
        X_clean = X_with_const
    
    # Fit model
    model = sm.OLS(y_clean, X_clean)
    
    if robust_se:
        results = model.fit(cov_type=cov_type)
    else:
        results = model.fit()
    
    # Extract key results
    output = {
        'model': results,
        'summary': results.summary(),
        'coef': results.params,
        'se': results.bse,
        'tvalues': results.tvalues,
        'pvalues': results.pvalues,
        'rsquared': results.rsquared,
        'rsquared_adj': results.rsquared_adj,
        'nobs': int(results.nobs),
        'aic': results.aic,
        'bic': results.bic
    }
    
    return output


def run_adf_test(series, maxlag=None, regression='c'):
    """
    Run Augmented Dickey-Fuller test for unit root.
    
    Parameters
    ----------
    series : pd.Series
        Time series to test
    maxlag : int, optional
        Maximum number of lags to use
    regression : str
        Deterministic terms: 'c' (constant), 'ct' (constant+trend), 'n' (none)
        
    Returns
    -------
    dict
        Dictionary containing ADF test results
    """
    series_clean = series.dropna()
    
    if len(series_clean) < 10:
        return {
            'test_statistic': np.nan,
            'pvalue': np.nan,
            'usedlag': np.nan,
            'nobs': len(series_clean),
            'critical_values': {},
            'stationary': False
        }
    
    result = adfuller(series_clean, maxlag=maxlag, regression=regression)
    
    return {
        'test_statistic': result[0],
        'pvalue': result[1],
        'usedlag': result[2],
        'nobs': result[3],
        'critical_values': result[4],
        'stationary': result[1] < 0.05
    }


def compute_half_life(beta, method='ar1'):
    """
    Compute mean-reversion half-life.
    
    Parameters
    ----------
    beta : float
        Mean-reversion coefficient
        For AR(1): ΔS_t = α + β·S_{t-1} + ε, use beta directly
        For deviation model: ΔS_t = α + β·d_{t-1} + ε, use beta directly
    method : str
        Computation method: 'ar1' or 'ou' (Ornstein-Uhlenbeck)
        
    Returns
    -------
    float
        Half-life in periods (days if daily data)
    """
    if beta >= 0:
        return np.inf
    
    if method == 'ar1':
        # From AR(1) process: half-life = -ln(2) / ln(1 + β)
        # where β < 0 for mean reversion
        return -np.log(2) / np.log(1 + beta)
    elif method == 'ou':
        # From continuous-time OU process: half-life = ln(2) / κ
        # where κ = -β (speed of mean reversion)
        kappa = -beta
        return np.log(2) / kappa
    else:
        raise ValueError(f"Unknown method: {method}")


def bootstrap_confidence_interval(data, stat_func, n_bootstrap=1000, 
                                   confidence_level=0.95, random_state=None):
    """
    Compute bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : pd.Series or np.ndarray
        Input data
    stat_func : callable
        Function to compute statistic (takes data as input)
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary with 'lower', 'upper', 'mean', 'std' of bootstrap distribution
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample_indices = np.random.choice(n, size=n, replace=True)
        
        if isinstance(data, pd.Series):
            sample = data.iloc[sample_indices]
        else:
            sample = data[sample_indices]
        
        try:
            stat = stat_func(sample)
            bootstrap_stats.append(stat)
        except:
            continue
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentiles
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return {
        'lower': np.percentile(bootstrap_stats, lower_percentile),
        'upper': np.percentile(bootstrap_stats, upper_percentile),
        'mean': np.mean(bootstrap_stats),
        'std': np.std(bootstrap_stats),
        'distribution': bootstrap_stats
    }


def test_regime_differences(group1_residuals, group2_residuals):
    """
    Test for significant differences between two regimes.
    
    Parameters
    ----------
    group1_residuals : array-like
        Regression residuals from regime 1
    group2_residuals : array-like
        Regression residuals from regime 2
        
    Returns
    -------
    dict
        Dictionary with test statistics and p-values
    """
    # Variance ratio test (F-test)
    var1 = np.var(group1_residuals, ddof=1)
    var2 = np.var(group2_residuals, ddof=1)
    f_stat = var1 / var2 if var1 > var2 else var2 / var1
    df1 = len(group1_residuals) - 1
    df2 = len(group2_residuals) - 1
    f_pvalue = 2 * min(stats.f.cdf(f_stat, df1, df2), 
                       1 - stats.f.cdf(f_stat, df1, df2))
    
    # Mean difference test (t-test assuming unequal variances)
    t_stat, t_pvalue = stats.ttest_ind(group1_residuals, group2_residuals, 
                                       equal_var=False)
    
    return {
        'variance_ratio': f_stat,
        'f_pvalue': f_pvalue,
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'mean_diff': np.mean(group1_residuals) - np.mean(group2_residuals)
    }
