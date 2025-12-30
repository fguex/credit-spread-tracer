"""
Plotting utilities for credit spread analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path


# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def save_figure(fig, filename, output_dir, dpi=300, bbox_inches='tight'):
    """
    Save figure to output directory.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    filename : str
        Output filename (without path)
    output_dir : str or Path
        Output directory path
    dpi : int
        Resolution in dots per inch
    bbox_inches : str
        Bounding box setting
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure: {filepath}")


def plot_spread_decomposition(spread, equilibrium, deviation, 
                              figsize=(14, 8), output_dir=None):
    """
    Plot spread decomposition into equilibrium and deviation components.
    
    Parameters
    ----------
    spread : pd.Series
        Credit spread time series
    equilibrium : pd.Series
        Equilibrium spread component
    deviation : pd.Series
        Deviation from equilibrium
    figsize : tuple
        Figure size
    output_dir : str or Path, optional
        Directory to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Spread and equilibrium
    axes[0].plot(spread.index, spread, label='Observed Spread', 
                linewidth=1.5, alpha=0.8)
    axes[0].plot(equilibrium.index, equilibrium, label='Equilibrium', 
                linewidth=2, linestyle='--')
    axes[0].set_ylabel('Spread (bps)')
    axes[0].set_title('Credit Spread Decomposition')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Deviation
    axes[1].plot(deviation.index, deviation, label='Deviation', 
                color='C2', linewidth=1)
    axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8)
    axes[1].fill_between(deviation.index, 0, deviation, alpha=0.3)
    axes[1].set_ylabel('Deviation (bps)')
    axes[1].set_title('Deviation from Equilibrium')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Distribution of deviation
    axes[2].hist(deviation.dropna(), bins=50, alpha=0.7, edgecolor='black')
    axes[2].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Deviation (bps)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Deviations')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(fig, 'spread_decomposition.png', output_dir)
    
    return fig


def plot_regression_diagnostics(results, residuals, fitted_values, 
                                figsize=(12, 10), output_dir=None):
    """
    Plot regression diagnostic plots.
    
    Parameters
    ----------
    results : statsmodels results object
        Fitted regression model
    residuals : pd.Series
        Regression residuals
    fitted_values : pd.Series
        Fitted values
    figsize : tuple
        Figure size
    output_dir : str or Path, optional
        Directory to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Residuals vs Fitted
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.5, s=20)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Q-Q plot
    sm.graphics.qqplot(residuals, line='45', ax=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scale-Location
    standardized_residuals = residuals / residuals.std()
    axes[1, 0].scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)), 
                      alpha=0.5, s=20)
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('√|Standardized Residuals|')
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals histogram
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residuals Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(fig, 'regression_diagnostics.png', output_dir)
    
    return fig


def plot_regime_comparison(data, regime_column, value_column, 
                          figsize=(12, 6), output_dir=None):
    """
    Compare variable distributions across regimes.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data containing regime labels and values
    regime_column : str
        Column name for regime labels
    value_column : str
        Column name for values to compare
    figsize : tuple
        Figure size
    output_dir : str or Path, optional
        Directory to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Box plot by regime
    data.boxplot(column=value_column, by=regime_column, ax=axes[0])
    axes[0].set_title(f'{value_column} by Regime')
    axes[0].set_xlabel('Regime')
    axes[0].set_ylabel(value_column)
    plt.suptitle('')
    
    # Plot 2: Distribution by regime
    regimes = data[regime_column].unique()
    for regime in sorted(regimes):
        regime_data = data[data[regime_column] == regime][value_column]
        axes[1].hist(regime_data, alpha=0.5, label=f'Regime {regime}', bins=30)
    
    axes[1].set_xlabel(value_column)
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{value_column} Distribution by Regime')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(fig, f'{value_column}_by_regime.png', output_dir)
    
    return fig


def plot_time_series_with_regimes(series, regime_labels, 
                                  figsize=(14, 6), output_dir=None):
    """
    Plot time series colored by regime.
    
    Parameters
    ----------
    series : pd.Series
        Time series to plot
    regime_labels : pd.Series
        Regime labels aligned with series
    figsize : tuple
        Figure size
    output_dir : str or Path, optional
        Directory to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique regimes and assign colors
    regimes = sorted(regime_labels.unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(regimes)))
    
    # Plot each regime segment
    for i, regime in enumerate(regimes):
        mask = regime_labels == regime
        ax.plot(series.index[mask], series[mask], 'o-', 
               label=f'Regime {regime}', color=colors[i], 
               markersize=3, linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(series.name if series.name else 'Value')
    ax.set_title(f'{series.name} by Regime' if series.name else 'Time Series by Regime')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(fig, 'time_series_regimes.png', output_dir)
    
    return fig


def plot_coefficient_comparison(coef_dict, figsize=(10, 6), output_dir=None):
    """
    Compare regression coefficients across different models or regimes.
    
    Parameters
    ----------
    coef_dict : dict
        Dictionary mapping model/regime names to coefficient DataFrames
        Each DataFrame should have columns: ['coef', 'se', 'pvalue']
    figsize : tuple
        Figure size
    output_dir : str or Path, optional
        Directory to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(coef_dict))
    width = 0.8 / len(next(iter(coef_dict.values())))
    
    for i, (model_name, coefs) in enumerate(coef_dict.items()):
        positions = x_pos[i] + np.arange(len(coefs)) * width
        ax.bar(positions, coefs['coef'], width, 
              yerr=1.96 * coefs['se'], label=model_name,
              capsize=5, alpha=0.8)
    
    ax.set_xlabel('Variable')
    ax.set_ylabel('Coefficient')
    ax.set_title('Coefficient Comparison Across Models')
    ax.set_xticks(x_pos + width * (len(next(iter(coef_dict.values()))) - 1) / 2)
    ax.set_xticklabels(coef_dict.keys())
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        save_figure(fig, 'coefficient_comparison.png', output_dir)
    
    return fig
