"""Build features for the credit spread tracer PoC.

Produces a weekly dataset sampled on Fridays with the exact columns required by the spec:
Date | S_t | ΔS_t | DGS10_t | SP500_ret_4w | VIX_t | y_{t+1}

The script writes the dataset to data/processed/dataset.csv when run as __main__.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.load_data import fetch_fred, resample_to_fridays


def build_dataset(start: str = "1995-01-01", out_path: str | None = None, mode: str = "weekly", h: int = 20) -> pd.DataFrame:
    """Build dataset.

    mode: 'weekly' (default) or 'daily'.
    h: target horizon in business days (default 20 for 20-day horizon).
    """
    # fetch daily raw series (includes DGS10 and DGS2 now)
    df = fetch_fred(start)

    if mode == "weekly":
        src = resample_to_fridays(df)
        idx = src.index
        S = src["S"]
        DGS10 = src["DGS10"]
        SP500 = src["SP500"]
        VIX = src["VIX"]
    elif mode == "daily":
        # keep the daily index as returned by fetch_fred (business days)
        src = df.copy()
        idx = src.index
        S = src["S"]
        DGS10 = src["DGS10"]
        # DGS2 may be present
        DGS2 = src.get("DGS2")
        SP500 = src["SP500"]
        VIX = src["VIX"]
    else:
        raise ValueError("mode must be 'weekly' or 'daily'")

    ds = pd.DataFrame(index=idx)
    ds["S_t"] = S

    # ΔS_t = S_t - S_{t-1}
    ds["ΔS_t"] = ds["S_t"].diff()

    # DGS10_t
    ds["DGS10_t"] = DGS10

    # SP500 returns: weekly uses 4-week diffs, daily uses 20-business-day diffs
    if mode == "weekly":
        ds["SP500_ret_4w"] = np.log(SP500.astype(float)).diff(4)
    else:
        # explicit 20-day return name for clarity in daily mode
        ds["SP500_ret_20d"] = np.log(SP500.astype(float)).diff(20)

    # VIX_t
    ds["VIX_t"] = VIX

    # Yield curve slope (10y - 2y) and normalized z-score over 252d
    if mode == "daily" and 'DGS2' in src.columns:
        ds["YC_slope_t"] = src["DGS10"].astype(float) - src["DGS2"].astype(float)
        roll = 252
        med = ds["YC_slope_t"].rolling(roll, min_periods=30).median()
        std = ds["YC_slope_t"].rolling(roll, min_periods=30).std().replace(0, 1e-8)
        ds["YC_slope_z"] = ((ds["YC_slope_t"] - med) / std).clip(-5, 5)

    # Target with horizon h business days
    ds["y_{t+1}"] = ds["S_t"].shift(-h) - ds["S_t"]

    # Drop any rows with NaN (strict requirement)
    ds = ds.dropna(how="any")

    # Ensure minimal columns order (YC_slope columns optional)
    if mode == "weekly":
        cols = ["S_t", "ΔS_t", "DGS10_t", "SP500_ret_4w", "VIX_t", "y_{t+1}"]
    else:
        cols = ["S_t", "ΔS_t", "DGS10_t", "SP500_ret_20d", "VIX_t", "y_{t+1}"]
    if "YC_slope_t" in ds.columns:
        cols.insert(4, "YC_slope_t")
        cols.insert(5, "YC_slope_z")
    ds = ds[cols]

    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        ds.to_csv(outp, index_label="Date")
        print(f"Saved dataset to {outp}")

    return ds


if __name__ == "__main__":
    ds = build_dataset(start="1995-01-01", out_path="data/processed/dataset.csv")
    print(ds.head())
