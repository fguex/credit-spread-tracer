"""Feature engineering pipeline for credit-spread-tracer.

Produces a DataFrame of predictors aligned to the target (no look-ahead).
Computes the credit-core signals, volatility, market/risk appetite features,
yield-curve, liquidity/funding, macro-cycle variables, cross-credit features,
and regime/interaction dummies.

Functions:
 - build_features_dataset(start, end, save, out_path): returns features DF and target y
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
                           save: bool = True, out_path: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Orchestrate features creation.

    Returns (X, y) where y is the target `y_{t+1}` present in the processed spread dataset.
    Features are shifted by 1 day to avoid lookahead.
    """
    # Load core spread dataset (daily) and macro dataset
    df_spread = build_processed_dataset(start=start, end=end, resample_weekly=False, save=False)
    df_macro, _ = build_macro_dataset(start=start, end=end, include_bonus=True, save=False)

    # Align indices
    idx = df_spread.index.intersection(df_macro.index)
    df_spread = df_spread.reindex(idx)
    df_macro = df_macro.reindex(idx)

    X = pd.DataFrame(index=idx)

    # 1) Credit-specific core signals
    X['S_t'] = df_spread['S_t']
    X['ΔS_1d'] = df_spread['S_t'].diff(1)
    X['ΔS_5d'] = df_spread['S_t'].diff(5)
    X['ΔS_20d'] = df_spread['S_t'].diff(20)
    X['z_S_20d'] = zscore(df_spread['S_t'], window=20)
    X['z_S_60d'] = zscore(df_spread['S_t'], window=60)

    # Volatility of spread
    X['vol_S_20d'] = df_spread['S_t'].rolling(20).std()
    X['vol_S_60d'] = df_spread['S_t'].rolling(60).std()
    X['Δvol_S_5d'] = X['vol_S_20d'].diff(5)

    # 2) Risk appetite / equities
    if 'VIXCLS' in df_macro.columns:
        X['VIX_t'] = df_macro['VIXCLS']
        X['ΔVIX_1d'] = df_macro['VIXCLS'].diff(1)
        X['ΔVIX_5d'] = df_macro['VIXCLS'].diff(5)
        X['z_VIX_60d'] = zscore(df_macro['VIXCLS'], window=60)
        # VIX regime dummy
        p90 = df_macro['VIXCLS'].quantile(0.9)
        X['VIX_regime'] = (df_macro['VIXCLS'] > p90).astype(int)

    if 'SP500' in df_macro.columns:
        X['SP500_ret_1d'] = np.log(df_macro['SP500']).diff(1)
        X['SP500_ret_5d'] = np.log(df_macro['SP500']).diff(5)
        X['SP500_ret_20d'] = np.log(df_macro['SP500']).diff(20)
        # rolling corr between ΔS and SP500 returns
        X['corr_ΔS_SP500_60d'] = X['ΔS_1d'].rolling(60).corr(X['SP500_ret_1d'])

    # 3) Yield curve
    if 'T10Y2Y' in df_macro.columns:
        X['YC'] = df_macro['T10Y2Y']
        X['ΔYC_5d'] = df_macro['T10Y2Y'].diff(5)
        X['z_YC_60d'] = zscore(df_macro['T10Y2Y'], window=60)
    elif 'T10Y3M' in df_macro.columns:
        X['YC'] = df_macro['T10Y3M']
        X['ΔYC_5d'] = df_macro['T10Y3M'].diff(5)
        X['z_YC_60d'] = zscore(df_macro['T10Y3M'], window=60)

    # 4) Funding & liquidity
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

    # 5) Macro cycle
    if 'CPIAUCSL' in df_macro.columns:
        # Year-over-year CPI
        X['CPI_yoy'] = df_macro['CPIAUCSL'].pct_change(12)
        X['ΔCPI_yoy'] = X['CPI_yoy'].diff(1)
        X['inflation_shock'] = (X['ΔCPI_yoy'] > X['ΔCPI_yoy'].quantile(0.9)).astype(int)

    if 'UNRATE' in df_macro.columns:
        X['UNRATE'] = df_macro['UNRATE']
        X['ΔUNRATE_3m'] = df_macro['UNRATE'].diff(63)  # approx 3 months trading days

    # 6) Cross-credit
    if 'BAMLH0A0HYM2' in df_macro.columns and 'BAMLC0A0CM' in df_spread.columns:
        X['HY_OAS'] = df_macro['BAMLH0A0HYM2']
        X['IG_OAS'] = df_spread['S_t']
        X['HY_IG_ratio'] = X['HY_OAS'] / X['IG_OAS']
        X['ΔHY_IG_ratio'] = X['HY_IG_ratio'].diff(1)

    # 7) Regime dummies and interactions
    # stress = VIX > P90 or NFCI > 0.5
    stress = pd.Series(False, index=idx)
    if 'VIXCLS' in df_macro.columns:
        stress = stress | (df_macro['VIXCLS'] > df_macro['VIXCLS'].quantile(0.9))
    if 'NFCI' in df_macro.columns:
        stress = stress | (df_macro['NFCI'] > 0.5)
    X['stress'] = stress.astype(int)

    X['liquidity_tight'] = (X.get('ΔRRP_5d', pd.Series(0, index=idx)) > 0).astype(int)
    X['curve_inverted'] = (X.get('YC', pd.Series(0, index=idx)) < 0).astype(int)

    # Interactions
    X['ΔS_x_stress'] = X['ΔS_1d'] * X['stress']
    X['ΔS_x_liquidity'] = X['ΔS_1d'] * X['liquidity_tight']

    # Avoid lookahead: shift all predictors by 1 day so they are known at time t
    X = X.shift(1).dropna(how='any')

    # Target (aligned to X index)
    y = df_spread.reindex(X.index)['y_{t+1}']

    if out_path is None:
        out_path = 'data/processed/features_dataset.parquet'
    if save:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        X.to_parquet(p)
        y.to_csv(p.with_suffix('.target.csv'), index=True)

    return X, y


if __name__ == '__main__':
    print('Building features dataset (this may fetch data)')
    X, y = build_features_dataset(start='2016-01-01', save=False)
    print('Features shape:', X.shape)
"""Feature engineering pipeline for credit-spread-tracer.

Produces a DataFrame of predictors aligned to the target (no look-ahead).
Computes the credit-core signals, volatility, market/risk appetite features,
yield-curve, liquidity/funding, macro-cycle variables, cross-credit features,
and regime/interaction dummies.

Functions:
 - build_features_dataset(start, end, save, out_path): returns features DF and target y
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
                           save: bool = True, out_path: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Orchestrate features creation.

    Returns (X, y) where y is the target `y_{t+1}` present in the processed spread dataset.
    Features are shifted by 1 day to avoid lookahead.
    """
    # Load core spread dataset (daily) and macro dataset
    df_spread = build_processed_dataset(start=start, end=end, resample_weekly=False, save=False)
    df_macro, _ = build_macro_dataset(start=start, end=end, include_bonus=True, save=False)

    # Align indices
    idx = df_spread.index.intersection(df_macro.index)
    df_spread = df_spread.reindex(idx)
    df_macro = df_macro.reindex(idx)

    X = pd.DataFrame(index=idx)

    # 1) Credit-specific core signals
    X['S_t'] = df_spread['S_t']
    X['ΔS_1d'] = df_spread['S_t'].diff(1)
    X['ΔS_5d'] = df_spread['S_t'].diff(5)
    X['ΔS_20d'] = df_spread['S_t'].diff(20)
    X['z_S_20d'] = zscore(df_spread['S_t'], window=20)
    X['z_S_60d'] = zscore(df_spread['S_t'], window=60)

    # Volatility of spread
    X['vol_S_20d'] = df_spread['S_t'].rolling(20).std()
    X['vol_S_60d'] = df_spread['S_t'].rolling(60).std()
    X['Δvol_S_5d'] = X['vol_S_20d'].diff(5)

    # 2) Risk appetite / equities
    if 'VIXCLS' in df_macro.columns:
        X['VIX_t'] = df_macro['VIXCLS']
        X['ΔVIX_1d'] = df_macro['VIXCLS'].diff(1)
        X['ΔVIX_5d'] = df_macro['VIXCLS'].diff(5)
        X['z_VIX_60d'] = zscore(df_macro['VIXCLS'], window=60)
        # VIX regime dummy
        p90 = df_macro['VIXCLS'].quantile(0.9)
        X['VIX_regime'] = (df_macro['VIXCLS'] > p90).astype(int)

    if 'SP500' in df_macro.columns:
        X['SP500_ret_1d'] = np.log(df_macro['SP500']).diff(1)
        X['SP500_ret_5d'] = np.log(df_macro['SP500']).diff(5)
        X['SP500_ret_20d'] = np.log(df_macro['SP500']).diff(20)
        # rolling corr between ΔS and SP500 returns
        X['corr_ΔS_SP500_60d'] = X['ΔS_1d'].rolling(60).corr(X['SP500_ret_1d'])

    # 3) Yield curve
    if 'T10Y2Y' in df_macro.columns:
        X['YC'] = df_macro['T10Y2Y']
        X['ΔYC_5d'] = df_macro['T10Y2Y'].diff(5)
        X['z_YC_60d'] = zscore(df_macro['T10Y2Y'], window=60)
    elif 'T10Y3M' in df_macro.columns:
        X['YC'] = df_macro['T10Y3M']
        X['ΔYC_5d'] = df_macro['T10Y3M'].diff(5)
        X['z_YC_60d'] = zscore(df_macro['T10Y3M'], window=60)

    # 4) Funding & liquidity
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

    # 5) Macro cycle
    if 'CPIAUCSL' in df_macro.columns:
        # Year-over-year CPI
        X['CPI_yoy'] = df_macro['CPIAUCSL'].pct_change(12)
        X['ΔCPI_yoy'] = X['CPI_yoy'].diff(1)
        X['inflation_shock'] = (X['ΔCPI_yoy'] > X['ΔCPI_yoy'].quantile(0.9)).astype(int)

    if 'UNRATE' in df_macro.columns:
        X['UNRATE'] = df_macro['UNRATE']
        X['ΔUNRATE_3m'] = df_macro['UNRATE'].diff(63)  # approx 3 months trading days

    # 6) Cross-credit
    if 'BAMLH0A0HYM2' in df_macro.columns and 'BAMLC0A0CM' in df_spread.columns:
        X['HY_OAS'] = df_macro['BAMLH0A0HYM2']
        X['IG_OAS'] = df_spread['S_t']
        X['HY_IG_ratio'] = X['HY_OAS'] / X['IG_OAS']
        X['ΔHY_IG_ratio'] = X['HY_IG_ratio'].diff(1)

    # 7) Regime dummies and interactions
    # stress = VIX > P90 or NFCI > 0.5
    stress = pd.Series(False, index=idx)
    if 'VIXCLS' in df_macro.columns:
        stress = stress | (df_macro['VIXCLS'] > df_macro['VIXCLS'].quantile(0.9))
    if 'NFCI' in df_macro.columns:
        stress = stress | (df_macro['NFCI'] > 0.5)
    X['stress'] = stress.astype(int)

    X['liquidity_tight'] = (X.get('ΔRRP_5d', pd.Series(0, index=idx)) > 0).astype(int)
    X['curve_inverted'] = (X.get('YC', pd.Series(0, index=idx)) < 0).astype(int)

    # Interactions
    X['ΔS_x_stress'] = X['ΔS_1d'] * X['stress']
    X['ΔS_x_liquidity'] = X['ΔS_1d'] * X['liquidity_tight']

    # Avoid lookahead: shift all predictors by 1 day so they are known at time t
    X = X.shift(1).dropna(how='any')

    # Target (aligned to X index)
    y = df_spread.reindex(X.index)['y_{t+1}']

    if out_path is None:
        out_path = 'data/processed/features_dataset.parquet'
    if save:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        X.to_parquet(p)
        y.to_csv(p.with_suffix('.target.csv'), index=True)

    return X, y


if __name__ == '__main__':
    print('Building features dataset (this may fetch data)')
    X, y = build_features_dataset(start='2016-01-01', save=False)
    print('Features shape:', X.shape)
"""Feature engineering pipeline for credit-spread-tracer.

Produces a DataFrame of predictors aligned to the target (no look-ahead).
Computes the credit-core signals, volatility, market/risk appetite features,
yield-curve, liquidity/funding, macro-cycle variables, cross-credit features,
and regime/interaction dummies.

Functions:
 - build_features_dataset(start, end, save, out_path): returns features DF and target y
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
                           save: bool = True, out_path: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Orchestrate features creation.

    Returns (X, y) where y is the target `y_{t+1}` present in the processed spread dataset.
    Features are shifted by 1 day to avoid lookahead.
    """
    # Load core spread dataset (daily) and macro dataset
    df_spread = build_processed_dataset(start=start, end=end, resample_weekly=False, save=False)
    df_macro, _ = build_macro_dataset(start=start, end=end, include_bonus=True, save=False)

    # Align indices
    idx = df_spread.index.intersection(df_macro.index)
    df_spread = df_spread.reindex(idx)
    df_macro = df_macro.reindex(idx)

    X = pd.DataFrame(index=idx)

    # 1) Credit-specific core signals
    X['S_t'] = df_spread['S_t']
    X['ΔS_1d'] = df_spread['S_t'].diff(1)
    X['ΔS_5d'] = df_spread['S_t'].diff(5)
    X['ΔS_20d'] = df_spread['S_t'].diff(20)
    X['z_S_20d'] = zscore(df_spread['S_t'], window=20)
    X['z_S_60d'] = zscore(df_spread['S_t'], window=60)

    # Volatility of spread
    X['vol_S_20d'] = df_spread['S_t'].rolling(20).std()
    X['vol_S_60d'] = df_spread['S_t'].rolling(60).std()
    X['Δvol_S_5d'] = X['vol_S_20d'].diff(5)

    # 2) Risk appetite / equities
    if 'VIXCLS' in df_macro.columns:
        X['VIX_t'] = df_macro['VIXCLS']
        X['ΔVIX_1d'] = df_macro['VIXCLS'].diff(1)
        X['ΔVIX_5d'] = df_macro['VIXCLS'].diff(5)
        X['z_VIX_60d'] = zscore(df_macro['VIXCLS'], window=60)
        # VIX regime dummy
        p90 = df_macro['VIXCLS'].quantile(0.9)
        X['VIX_regime'] = (df_macro['VIXCLS'] > p90).astype(int)

    if 'SP500' in df_macro.columns:
        X['SP500_ret_1d'] = np.log(df_macro['SP500']).diff(1)
        X['SP500_ret_5d'] = np.log(df_macro['SP500']).diff(5)
        X['SP500_ret_20d'] = np.log(df_macro['SP500']).diff(20)
        # rolling corr between ΔS and SP500 returns
        X['corr_ΔS_SP500_60d'] = X['ΔS_1d'].rolling(60).corr(X['SP500_ret_1d'])

    # 3) Yield curve
    if 'T10Y2Y' in df_macro.columns:
        X['YC'] = df_macro['T10Y2Y']
        X['ΔYC_5d'] = df_macro['T10Y2Y'].diff(5)
        X['z_YC_60d'] = zscore(df_macro['T10Y2Y'], window=60)
    elif 'T10Y3M' in df_macro.columns:
        X['YC'] = df_macro['T10Y3M']
        X['ΔYC_5d'] = df_macro['T10Y3M'].diff(5)
        X['z_YC_60d'] = zscore(df_macro['T10Y3M'], window=60)

    # 4) Funding & liquidity
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

    # 5) Macro cycle
    if 'CPIAUCSL' in df_macro.columns:
        # Year-over-year CPI
        X['CPI_yoy'] = df_macro['CPIAUCSL'].pct_change(12)
        X['ΔCPI_yoy'] = X['CPI_yoy'].diff(1)
        X['inflation_shock'] = (X['ΔCPI_yoy'] > X['ΔCPI_yoy'].quantile(0.9)).astype(int)

    if 'UNRATE' in df_macro.columns:
        X['UNRATE'] = df_macro['UNRATE']
        X['ΔUNRATE_3m'] = df_macro['UNRATE'].diff(63)  # approx 3 months trading days

    # 6) Cross-credit
    if 'BAMLH0A0HYM2' in df_macro.columns and 'BAMLC0A0CM' in df_spread.columns:
        X['HY_OAS'] = df_macro['BAMLH0A0HYM2']
        X['IG_OAS'] = df_spread['S_t']
        X['HY_IG_ratio'] = X['HY_OAS'] / X['IG_OAS']
        X['ΔHY_IG_ratio'] = X['HY_IG_ratio'].diff(1)

    # 7) Regime dummies and interactions
    # stress = VIX > P90 or NFCI > 0.5
    stress = pd.Series(False, index=idx)
    if 'VIXCLS' in df_macro.columns:
        stress = stress | (df_macro['VIXCLS'] > df_macro['VIXCLS'].quantile(0.9))
    if 'NFCI' in df_macro.columns:
        stress = stress | (df_macro['NFCI'] > 0.5)
    X['stress'] = stress.astype(int)

    X['liquidity_tight'] = (X.get('ΔRRP_5d', pd.Series(0, index=idx)) > 0).astype(int)
    X['curve_inverted'] = (X.get('YC', pd.Series(0, index=idx)) < 0).astype(int)

    # Interactions
    X['ΔS_x_stress'] = X['ΔS_1d'] * X['stress']
    X['ΔS_x_liquidity'] = X['ΔS_1d'] * X['liquidity_tight']

    # Avoid lookahead: shift all predictors by 1 day so they are known at time t
    X = X.shift(1).dropna(how='any')

    # Target (aligned to X index)
    y = df_spread.reindex(X.index)['y_{t+1}']

    if out_path is None:
        out_path = 'data/processed/features_dataset.parquet'
    if save:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        X.to_parquet(p)
        y.to_csv(p.with_suffix('.target.csv'), index=True)

    return X, y


if __name__ == '__main__':
    print('Building features dataset (this may fetch data)')
    X, y = build_features_dataset(start='2016-01-01', save=False)
    print('Features shape:', X.shape)
"""Feature engineering pipeline for credit-spread-tracer.

Produces a DataFrame of predictors aligned to the target (no look-ahead).
Computes the credit-core signals, volatility, market/risk appetite features,
yield-curve, liquidity/funding, macro-cycle variables, cross-credit features,
and regime/interaction dummies.

Functions:
 - build_features_dataset(start, end, save, out_path): returns features DF and target y
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
                           save: bool = True, out_path: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Orchestrate features creation.

    Returns (X, y) where y is the target `y_{t+1}` present in the processed spread dataset.
    Features are shifted by 1 day to avoid lookahead.
    """
    # Load core spread dataset (daily) and macro dataset
    df_spread = build_processed_dataset(start=start, end=end, resample_weekly=False, save=False)
    df_macro, _ = build_macro_dataset(start=start, end=end, include_bonus=True, save=False)

    # Align indices
    idx = df_spread.index.intersection(df_macro.index)
    df_spread = df_spread.reindex(idx)
    df_macro = df_macro.reindex(idx)

    X = pd.DataFrame(index=idx)

    # 1) Credit-specific core signals
    X['S_t'] = df_spread['S_t']
    X['ΔS_1d'] = df_spread['S_t'].diff(1)
    X['ΔS_5d'] = df_spread['S_t'].diff(5)
    X['ΔS_20d'] = df_spread['S_t'].diff(20)
    X['z_S_20d'] = zscore(df_spread['S_t'], window=20)
    X['z_S_60d'] = zscore(df_spread['S_t'], window=60)

    # Volatility of spread
    X['vol_S_20d'] = df_spread['S_t'].rolling(20).std()
    X['vol_S_60d'] = df_spread['S_t'].rolling(60).std()
    X['Δvol_S_5d'] = X['vol_S_20d'].diff(5)

    # 2) Risk appetite / equities
    if 'VIXCLS' in df_macro.columns:
        X['VIX_t'] = df_macro['VIXCLS']
        X['ΔVIX_1d'] = df_macro['VIXCLS'].diff(1)
        X['ΔVIX_5d'] = df_macro['VIXCLS'].diff(5)
        X['z_VIX_60d'] = zscore(df_macro['VIXCLS'], window=60)
        # VIX regime dummy
        p90 = df_macro['VIXCLS'].quantile(0.9)
        X['VIX_regime'] = (df_macro['VIXCLS'] > p90).astype(int)

    if 'SP500' in df_macro.columns:
        X['SP500_ret_1d'] = np.log(df_macro['SP500']).diff(1)
        X['SP500_ret_5d'] = np.log(df_macro['SP500']).diff(5)
        X['SP500_ret_20d'] = np.log(df_macro['SP500']).diff(20)
        # rolling corr between ΔS and SP500 returns
        X['corr_ΔS_SP500_60d'] = X['ΔS_1d'].rolling(60).corr(X['SP500_ret_1d'])

    # 3) Yield curve
    if 'T10Y2Y' in df_macro.columns:
        X['YC'] = df_macro['T10Y2Y']
        X['ΔYC_5d'] = df_macro['T10Y2Y'].diff(5)
        X['z_YC_60d'] = zscore(df_macro['T10Y2Y'], window=60)
    elif 'T10Y3M' in df_macro.columns:
        X['YC'] = df_macro['T10Y3M']
        X['ΔYC_5d'] = df_macro['T10Y3M'].diff(5)
        X['z_YC_60d'] = zscore(df_macro['T10Y3M'], window=60)

    # 4) Funding & liquidity
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

    # 5) Macro cycle
    if 'CPIAUCSL' in df_macro.columns:
        # Year-over-year CPI
        X['CPI_yoy'] = df_macro['CPIAUCSL'].pct_change(12)
        X['ΔCPI_yoy'] = X['CPI_yoy'].diff(1)
        X['inflation_shock'] = (X['ΔCPI_yoy'] > X['ΔCPI_yoy'].quantile(0.9)).astype(int)

    if 'UNRATE' in df_macro.columns:
        X['UNRATE'] = df_macro['UNRATE']
        X['ΔUNRATE_3m'] = df_macro['UNRATE'].diff(63)  # approx 3 months trading days

    # 6) Cross-credit
    if 'BAMLH0A0HYM2' in df_macro.columns and 'BAMLC0A0CM' in df_spread.columns:
        X['HY_OAS'] = df_macro['BAMLH0A0HYM2']
        X['IG_OAS'] = df_spread['S_t']
        X['HY_IG_ratio'] = X['HY_OAS'] / X['IG_OAS']
        X['ΔHY_IG_ratio'] = X['HY_IG_ratio'].diff(1)

    # 7) Regime dummies and interactions
    # stress = VIX > P90 or NFCI > 0.5
    stress = pd.Series(False, index=idx)
    if 'VIXCLS' in df_macro.columns:
        stress = stress | (df_macro['VIXCLS'] > df_macro['VIXCLS'].quantile(0.9))
    if 'NFCI' in df_macro.columns:
        stress = stress | (df_macro['NFCI'] > 0.5)
    X['stress'] = stress.astype(int)

    X['liquidity_tight'] = (X.get('ΔRRP_5d', pd.Series(0, index=idx)) > 0).astype(int)
    X['curve_inverted'] = (X.get('YC', pd.Series(0, index=idx)) < 0).astype(int)

    # Interactions
    X['ΔS_x_stress'] = X['ΔS_1d'] * X['stress']
    X['ΔS_x_liquidity'] = X['ΔS_1d'] * X['liquidity_tight']

    # Avoid lookahead: shift all predictors by 1 day so they are known at time t
    X = X.shift(1).dropna(how='any')

    # Target (aligned to X index)
    y = df_spread.reindex(X.index)['y_{t+1}']

    if out_path is None:
        out_path = 'data/processed/features_dataset.parquet'
    if save:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        X.to_parquet(p)
        y.to_csv(p.with_suffix('.target.csv'), index=True)

    return X, y


if __name__ == '__main__':
    print('Building features dataset (this may fetch data)')
    X, y = build_features_dataset(start='2016-01-01', save=False)
    print('Features shape:', X.shape)
"""Build features for the credit spread tracer PoC.

