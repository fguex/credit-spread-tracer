# Runner for Normal-regime 10d bootstrap (minimal results)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.regression.quantile_regression import QuantReg
import gc

DATA_DIR = Path('data/processed')
RESULTS_DIR = Path('results/tables')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
data = pd.read_csv(DATA_DIR / 'full_processed_data_hmm.csv', index_col=0, parse_dates=True)
print('Data loaded:', data.shape)
print('Regime counts:')
print(data['regime'].value_counts().sort_index())

# Targeted bootstrap
B = 1000
horizon = 10
regime_id = 1

d = data.copy()
d['spread_lag1'] = d['spread'].shift(1)
d[f'spread_change_{horizon}d'] = d['spread'].shift(-horizon) - d['spread']
regime_data = d[d['regime'] == regime_id][['spread_lag1', f'spread_change_{horizon}d']].dropna()
n_obs = len(regime_data)
print(f'Normal regime (id=1) – Horizon={horizon}d – N={n_obs}')

if n_obs < 20:
    print('Insufficient data for reliable bootstrap (need >=20). Skipping.')
else:
    X = regime_data['spread_lag1'].values.flatten()
    y = regime_data[f'spread_change_{horizon}d'].values
    betas = np.full(B, np.nan)
    half_lives = np.full(B, np.nan)
    rng = np.random.default_rng(12345)
    for i in range(B):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        Xb = X[idx]
        yb = y[idx]
        Xb_c = np.column_stack([np.ones(len(Xb)), Xb])
        try:
            qr = QuantReg(yb, Xb_c).fit(q=0.5, max_iter=1000)
            b = qr.params[1]
            betas[i] = b
            half_lives[i] = np.log(2) / abs(b) if b < 0 else np.inf
        except Exception:
            pass
    valid_b = betas[~np.isnan(betas)]
    valid_hl = half_lives[np.isfinite(half_lives)]
    median_b = np.median(valid_b) if len(valid_b)>0 else np.nan
    b_ci = np.percentile(valid_b, [2.5, 97.5]) if len(valid_b)>0 else (np.nan, np.nan)
    median_hl = np.median(valid_hl) if len(valid_hl)>0 else np.inf
    hl_ci = np.percentile(valid_hl, [2.5, 97.5]) if len(valid_hl)>0 else (np.inf, np.inf)
    print('--- Bootstrap summary ---')
    print(f'Median β = {median_b:.4f}')
    print(f'95% CI β = [{b_ci[0]:.4f}, {b_ci[1]:.4f}]')
    print(f'Median half-life = {median_hl:.1f} days')
    print(f'95% CI half-life = [{hl_ci[0]:.1f}, {hl_ci[1]:.1f}]')

    # Save numeric results
    out = {
        'median_beta': float(median_b),
        'beta_ci_lower': float(b_ci[0]),
        'beta_ci_upper': float(b_ci[1]),
        'median_half_life': float(median_hl),
        'hl_ci_lower': float(hl_ci[0]),
        'hl_ci_upper': float(hl_ci[1]),
        'n_obs': int(n_obs),
        'n_valid_bootstrap': int(len(valid_b))
    }
    pd.Series(out).to_json(RESULTS_DIR / 'bootstrap_normal_10d_summary.json')

    # Plots
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    if len(valid_b)>0:
        axes[0].hist(valid_b, bins=40, color='tab:blue', alpha=0.7, edgecolor='black')
        axes[0].axvline(median_b, color='darkblue', linewidth=2, label=f'Median {median_b:.4f}')
        axes[0].axvline(b_ci[0], color='red', linestyle='--', linewidth=1.5)
        axes[0].axvline(b_ci[1], color='red', linestyle='--', linewidth=1.5)
        axes[0].axvline(0, color='gray', linewidth=1)
        axes[0].set_title('Bootstrap β distribution (Normal, 10d)')
        axes[0].legend()
    hl_display = np.clip(valid_hl, 0, 200) if len(valid_hl)>0 else np.array([])
    if len(hl_display)>0:
        axes[1].boxplot(hl_display, vert=True, patch_artist=True, boxprops=dict(facecolor='tab:green', alpha=0.6))
    axes[1].set_title('Bootstrap half-life (Normal, 10d) — capped at 200d')
    axes[1].set_ylabel('Half-life (days)')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'bootstrap_normal_10d.png', dpi=150)
    print(f'Plots saved to {RESULTS_DIR / "bootstrap_normal_10d.png"}')

    # Cleanup
    del betas, half_lives, valid_b, valid_hl, hl_display
    gc.collect()

print('Done')
