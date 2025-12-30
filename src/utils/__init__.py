"""
Utility modules for credit spread analysis.
"""
from .feature_engineering import (
    compute_z_score,
    compute_momentum,
    compute_realized_volatility,
    compute_equilibrium_spread
)
from .regression_tests import (
    run_conditional_regression,
    run_adf_test,
    compute_half_life,
    bootstrap_confidence_interval
)
from .plotting import (
    plot_spread_decomposition,
    plot_regression_diagnostics,
    plot_regime_comparison,
    save_figure
)

__all__ = [
    'compute_z_score',
    'compute_momentum',
    'compute_realized_volatility',
    'compute_equilibrium_spread',
    'run_conditional_regression',
    'run_adf_test',
    'compute_half_life',
    'bootstrap_confidence_interval',
    'plot_spread_decomposition',
    'plot_regression_diagnostics',
    'plot_regime_comparison',
    'save_figure'
]
