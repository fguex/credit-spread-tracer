"""Feature engineering module (stable path).

This module provides `build_features_dataset` which is identical to the previously
planned `build_features.build_features_dataset` but lives in a clean file to avoid
editor/merge issues.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple

from src.data.load_data import build_processed_dataset
from src.data.data_macro import build_macro_dataset


def zscore(series: pd.Series, window: int = 60) -> pd.Series:
    roll_mean = series.rolling(window, min_periods=max(5, int(window / 6))).mean()
    roll_std = series.rolling(window, min_periods=max(5, int(window / 6))).std().replace(0, 1e-8)
    return (series - roll_mean) / roll_std


def build_features_dataset(start: str = "2016-01-01", end: str | None = None, *,
                           h: int = 20, save: bool = True, out_path: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    # Load core spread dataset (daily) and macro dataset
    df_spread = build_processed_dataset(start=start, end=end, resample_weekly=False, save=False)
    df_macro, _ = build_macro_dataset(start=start, end=end, include_bonus=True, save=False)

    # Create prediction target at horizon h (y_{t+1} style)
    df_spread = df_spread.copy()
    df_spread[f'y_{{t+1}}'] = df_spread['S_t'].shift(-h) - df_spread['S_t']

    # Align indices
    idx = df_spread.index.intersection(df_macro.index)
    df_spread = df_spread.reindex(idx)
    df_macro = df_macro.reindex(idx)

    X = pd.DataFrame(index=idx)

    # Credit core
    X['S_t'] = df_spread['S_t']
    X['ΔS_1d'] = df_spread['S_t'].diff(1)
    X['ΔS_5d'] = df_spread['S_t'].diff(5)
    X['ΔS_20d'] = df_spread['S_t'].diff(20)
    X['z_S_20d'] = zscore(df_spread['S_t'], window=20)
    X['z_S_60d'] = zscore(df_spread['S_t'], window=60)

    # Volatility
    X['vol_S_20d'] = df_spread['S_t'].rolling(20).std()
    X['vol_S_60d'] = df_spread['S_t'].rolling(60).std()
    X['Δvol_S_5d'] = X['vol_S_20d'].diff(5)

    # Risk appetite / VIX
    if 'VIXCLS' in df_macro.columns:
        X['VIX_t'] = df_macro['VIXCLS']
        X['ΔVIX_1d'] = df_macro['VIXCLS'].diff(1)
        X['ΔVIX_5d'] = df_macro['VIXCLS'].diff(5)
        X['z_VIX_60d'] = zscore(df_macro['VIXCLS'], window=60)
        p90 = df_macro['VIXCLS'].quantile(0.9)
        X['VIX_regime'] = (df_macro['VIXCLS'] > p90).astype(int)

    # Equities
    if 'SP500' in df_macro.columns:
        X['SP500_ret_1d'] = np.log(df_macro['SP500']).diff(1)
        X['SP500_ret_5d'] = np.log(df_macro['SP500']).diff(5)
        X['SP500_ret_20d'] = np.log(df_macro['SP500']).diff(20)
        X['corr_ΔS_SP500_60d'] = X['ΔS_1d'].rolling(60).corr(X['SP500_ret_1d'])

    # Yield curve
    if 'T10Y2Y' in df_macro.columns:
        X['YC'] = df_macro['T10Y2Y']
        X['ΔYC_5d'] = df_macro['T10Y2Y'].diff(5)
        X['z_YC_60d'] = zscore(df_macro['T10Y2Y'], window=60)

    # Liquidity
    if 'NFCI' in df_macro.columns:
        X['NFCI_t'] = df_macro['NFCI']
        X['ΔNFCI_1d'] = df_macro['NFCI'].diff(1)
        X['NFCI_stress'] = (df_macro['NFCI'] > 0).astype(int)

    if 'RRPONTSYD' in df_macro.columns:
        X['RRP_t'] = df_macro['RRPONTSYD']
        X['ΔRRP_5d'] = df_macro['RRPONTSYD'].diff(5)
        X['z_RRP_60d'] = zscore(df_macro['RRPONTSYD'], window=60)

    if 'TEDRATE' in df_macro.columns:
        X['TED_t'] = df_macro['TEDRATE']
        X['ΔTED_1d'] = df_macro['TEDRATE'].diff(1)

    # Macro cycle
    if 'CPIAUCSL' in df_macro.columns:
        X['CPI_yoy'] = df_macro['CPIAUCSL'].pct_change(12)
        X['ΔCPI_yoy'] = X['CPI_yoy'].diff(1)
        X['inflation_shock'] = (X['ΔCPI_yoy'] > X['ΔCPI_yoy'].quantile(0.9)).astype(int)

    if 'UNRATE' in df_macro.columns:
        X['UNRATE'] = df_macro['UNRATE']
        X['ΔUNRATE_3m'] = df_macro['UNRATE'].diff(63)

    # Cross-credit
    if 'BAMLH0A0HYM2' in df_macro.columns and 'BAMLC0A0CM' in df_spread.columns:
        X['HY_OAS'] = df_macro['BAMLH0A0HYM2']
        X['IG_OAS'] = df_spread['S_t']
        X['HY_IG_ratio'] = X['HY_OAS'] / X['IG_OAS']
        X['ΔHY_IG_ratio'] = X['HY_IG_ratio'].diff(1)

    # Regimes and interactions
    stress = pd.Series(False, index=idx)
    if 'VIXCLS' in df_macro.columns:
        stress = stress | (df_macro['VIXCLS'] > df_macro['VIXCLS'].quantile(0.9))
    if 'NFCI' in df_macro.columns:
        stress = stress | (df_macro['NFCI'] > 0.5)
    X['stress'] = stress.astype(int)
    X['liquidity_tight'] = (X.get('ΔRRP_5d', pd.Series(0, index=idx)) > 0).astype(int)
    X['curve_inverted'] = (X.get('YC', pd.Series(0, index=idx)) < 0).astype(int)
    X['ΔS_x_stress'] = X['ΔS_1d'] * X['stress']
    X['ΔS_x_liquidity'] = X['ΔS_1d'] * X['liquidity_tight']

    # Avoid lookahead
    X = X.shift(1).dropna(how='any')
    y = df_spread.reindex(X.index)['y_{t+1}']

    if out_path is None:
        out_path = 'data/processed/features_dataset.parquet'
    if save:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        X.to_parquet(p)
        y.to_csv(p.with_suffix('.target.csv'), index=True)

    return X, y
