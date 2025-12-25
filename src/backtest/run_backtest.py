"""Expanding-window backtest runner for the PoC.

Implements the exact modeling and trading rule specified in the project brief.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
 


@dataclass
class BacktestResult:
    predictions: pd.Series
    pnl: pd.Series
    cum_pnl: pd.Series
    hit_ratio: float
    coef_history: pd.DataFrame


def run_backtest(df: pd.DataFrame, train_start_year: int = 1999, train_end_year: int = 2006, alpha: float = 1.0) -> BacktestResult:
    """Run expanding-window weekly backtest.

    df must have columns: S_t, ΔS_t, DGS10_t, SP500_ret_4w, VIX_t, y_{t+1}
    Index is Date in ascending order.
    """
    # Determine which SP500 return column is present (daily vs weekly)
    sp500_col = None
    if 'SP500_ret_20d' in df.columns:
        sp500_col = 'SP500_ret_20d'
    elif 'SP500_ret_4w' in df.columns:
        sp500_col = 'SP500_ret_4w'

    required = ["S_t", "ΔS_t", "DGS10_t", sp500_col, "VIX_t", "y_{t+1}"]
    if not all((c in df.columns) for c in required if c is not None):
        raise ValueError("Input df missing required columns")

    # Determine train initial end index
    start_date = pd.Timestamp(f"{train_start_year}-01-01")
    end_init = pd.Timestamp(f"{train_end_year}-12-31")

    # features X_t
    X_cols = ["S_t", "ΔS_t", "DGS10_t", sp500_col, "VIX_t"]
    X = df[X_cols].copy()
    y = df["y_{t+1}"].copy()

    # Identify rows where features and target are non-null
    valid_mask = X.notnull().all(axis=1) & y.notnull()

    # We'll train on rows where index <= end_init, starting from first index >= start_date
    train_mask = (df.index >= start_date) & (df.index <= end_init) & valid_mask
    first_train_idx = df.index[train_mask].min()
    if pd.isna(first_train_idx):
        # Fall back to an available initial training window instead of raising.
        # Choose a pragmatic initial training end by position (approx 7 years of weekly rows)
        import warnings

        warnings.warn(
            "Requested initial training window (" + str(start_date.date()) + " - " + str(end_init.date()) + ") "
            "contained no rows in the dataset. Falling back to the earliest available training window.",
            UserWarning,
        )

        # Use the earliest available index as the first training row
        first_train_idx = df.index.min()
        # pick an end index roughly 7 years later by row-count (52 weeks * 7)
        approx_rows = 52 * 7
        if len(df) > approx_rows + 1:
            end_init = df.index[approx_rows]
        else:
            # choose the penultimate index so we leave at least one row for testing
            end_init = df.index[-2] if len(df) >= 2 else df.index[-1]

    # Prepare stores
    preds = pd.Series(index=df.index, dtype=float)
    coefs = []
    coef_idx = []

    # Expanding window: for each t in test (after end_init) predict y_{t+1} using model trained on all data up to t
    # We will step through each date t where we can form a prediction (i.e., t <= last_index-1)
    for i, date in enumerate(df.index):
        if date < first_train_idx:
            continue
        if date <= end_init:
            # skip prediction during initial training window
            continue

        # training set: all rows with index <= date and valid (no NaNs)
        train_idx = (df.index <= date) & valid_mask
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]

        # Skip if training set is empty (shouldn't normally happen due to first_train_idx)
        if X_train.shape[0] == 0:
            continue

        # fit ridge (lazy import so module can be imported even if sklearn is missing)
        try:
            from sklearn.linear_model import Ridge
        except Exception as e:  # pragma: no cover - environment dependent
            raise ImportError("scikit-learn is required to run the backtest. Install with: pip install scikit-learn") from e

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # store coefs
        coefs.append(np.concatenate([[model.intercept_], model.coef_.ravel()]))
        coef_idx.append(date)

        # predict for date (prediction corresponds to y_{t+1})
        # Use a DataFrame slice so sklearn receives column names
        # Only predict if current feature row is valid (no NaNs)
        if not valid_mask.loc[date]:
            # cannot form a valid feature row for prediction
            continue
        xt_df = X.loc[[date]]
        yhat = float(model.predict(xt_df)[0])
        preds.loc[date] = yhat

    # Compute PnL using the dataset's target y (which may represent multi-day horizon)
    # preds at index t correspond to predicted change over the dataset's target horizon.
    # Use actual y values aligned to preds to compute economic profit proxy.
    actual_full = y
    pnl = -np.sign(preds) * actual_full
    pnl = pnl.dropna()

    cum_pnl = pnl.cumsum()

    # Hit ratio: fraction where sign(pred) == sign(actual y_{t+1})
    # Exclude actual==0 from the denominator to avoid ambiguity with sign(0)==0.
    preds_nonan = preds.dropna()
    actual = y.loc[preds_nonan.index]
    actual_sign = np.sign(actual)
    nonzero_mask = actual_sign != 0
    if nonzero_mask.sum() == 0:
        hit_ratio = float('nan')
    else:
        aligned_preds = np.sign(preds_nonan.loc[nonzero_mask.index[nonzero_mask]])
        aligned_actual = actual_sign[nonzero_mask]
        hit_ratio = (aligned_preds == aligned_actual).sum() / len(aligned_actual)

    # Build coef history DataFrame (if any coefficients were recorded)
    if coefs:
        coef_cols = ["intercept", "S_t", "ΔS_t", "DGS10_t", sp500_col, "VIX_t"]
        coef_history = pd.DataFrame(coefs, index=coef_idx, columns=coef_cols)
    else:
        coef_history = pd.DataFrame()

    return BacktestResult(predictions=preds, pnl=pnl, cum_pnl=cum_pnl, hit_ratio=hit_ratio, coef_history=coef_history)


def save_results(result: BacktestResult, out_dir: str = "data/processed") -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    # save predictions, pnl, coef_history
    result.predictions.to_csv(p / "predictions.csv", index_label="Date")
    result.pnl.to_csv(p / "pnl.csv", index_label="Date")
    result.cum_pnl.to_csv(p / "cum_pnl.csv", index_label="Date")
    result.coef_history.to_csv(p / "coef_history.csv", index_label="Date")
    print(f"Saved backtest results to {p}")


if __name__ == "__main__":
    # quick runner: load dataset and run backtest
    ds_path = Path("data/processed/dataset.csv")
    if not ds_path.exists():
        raise SystemExit("Run feature builder first to create data/processed/dataset.csv")

    df = pd.read_csv(ds_path, index_col=0, parse_dates=True)
    res = run_backtest(df)
    save_results(res)
    print(f"Hit ratio: {res.hit_ratio:.3f}")
