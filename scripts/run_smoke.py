# Small smoke runner: build daily dataset and run backtest
# Usage: python scripts/run_smoke.py

from src.features.build_features import build_dataset
from src.backtest.run_backtest import run_backtest, save_results
import pandas as pd
import sys

try:
    print('Building daily dataset (this may fetch from FRED)...')
    ds = build_dataset(start='1995-01-01', out_path='data/processed/dataset.csv', mode='daily', h=20)
    print('Dataset built: rows=%d, cols=%s' % (len(ds), ds.columns.tolist()))
except Exception as e:
    print('Failed to build dataset:', type(e).__name__, e)
    sys.exit(1)

# Quick run of the backtest using a 3-year initial window
start_year = ds.index.min().year
train_start = start_year
train_end = start_year + 3
print(f'Running backtest with initial window {train_start}-{train_end}...')
try:
    res = run_backtest(ds, train_start_year=train_start, train_end_year=train_end, alpha=1.0)
    save_results(res, out_dir='data/processed')
    print('Backtest finished')
    print('predictions non-null:', res.predictions.dropna().shape[0])
    print('pnl non-null:', res.pnl.dropna().shape[0])
    print('hit_ratio:', res.hit_ratio)
except Exception as e:
    print('Backtest failed:', type(e).__name__, e)
    sys.exit(2)

print('Smoke run complete. Inspect data/processed for outputs: predictions.csv, pnl.csv, cum_pnl.csv, coef_history.csv')
