"""Simple FRED data loader for the PoC.

Fetches required series via pandas_datareader and returns a DataFrame indexed by date.
Only the 4 series in the spec are fetched.
"""
from __future__ import annotations

from datetime import datetime
import pandas as pd
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

import io
import requests


SERIES = {
    "S": "BAMLC0A0CM",  # ICE BofA US Corporate OAS (in %)
    "DGS10": "DGS10",  # 10Y Treasury
    "DGS2": "DGS2",    # 2Y Treasury (for yield curve slope)
    "SP500": "SP500",  # S&P 500 index level
    "VIX": "VIXCLS",  # VIX index level
}


def fetch_fred(start: str = "1987-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch the required FRED series and return a DataFrame with columns [S, DGS10, SP500, VIX].

    Dates are business daily as returned by pandas_datareader.
    Default start: 1987-01-01 to capture Black Monday and all subsequent major crises.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    frames = []
    for name, series in SERIES.items():
        print(f"Fetching {series} as {name} from FRED")
        df = None
        # try pandas_datareader first
        if pdr is not None:
            try:
                df = pdr.DataReader(series, "fred", start, end)
            except Exception as e:
                print(f"pandas_datareader failed for {series}: {e}. Falling back to direct HTTP CSV download.")

        if df is None:
            # fallback to direct CSV download from FRED
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&download_format=csv"
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text), parse_dates=[0], index_col=0)
            except Exception as e:
                raise RuntimeError(f"Failed to fetch series {series} via HTTP CSV from FRED: {e}") from e

        # Normalize column name to the short name
        df = df.rename(columns={df.columns[0]: name})
        frames.append(df)

    df = pd.concat(frames, axis=1)
    # cast index to datetime and sort
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def resample_to_fridays(df: pd.DataFrame) -> pd.DataFrame:
    """Resample the daily series to weekly observations taken on Fridays.

    If Friday is missing (holiday), take the last available observation on or before Friday.
    Ensures deterministic, non-leaky weekly alignment.
    """
    # Create a weekly index of Fridays between min and max dates
    start = df.index.min().date()
    end = df.index.max().date()
    weekly_idx = pd.date_range(start=start, end=end, freq="W-FRI")

    # For each Friday, take last available observation on or before that date
    weekly = df.reindex(df.index.union(weekly_idx))
    weekly = weekly.sort_index().ffill().reindex(weekly_idx)
    weekly.index.name = "Date"
    return weekly


def load_raw_sources(start: str = "1987-01-01", end: str | None = None) -> dict:
    """Load available raw sources and return a dict of DataFrames.

    Currently loads FRED series via `fetch_fred`. This function centralizes
    raw-source loading so additional readers (local CSVs, vendor APIs) can be
    added later and used by the same ETL pipeline.
    
    Default start: 1987-01-01 to capture Black Monday and all subsequent major crises.
    """
    sources = {}
    sources["fred"] = fetch_fred(start=start, end=end)
    # Placeholder: detect any CSVs in data/raw and load them if present
    # (kept conservative for the PoC - explicit additions preferred)
    return sources


def clean_fred(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the FRED DataFrame and normalize series.

    - Convert OAS series `S` from percent to basis points (bps).
    - Ensure numeric types and consistent datetime index.
    - Rename columns to canonical names expected downstream.
    """
    df = df.copy()
    # Ensure datetime index
    df.index = pd.to_datetime(df.index)

    # Convert OAS (S) from % to bps if values look like percentages (<= 100)
    if "S" in df.columns:
        # if values are small (<100) we assume percent; multiply to get bps
        sample = df["S"].dropna()
        if not sample.empty and sample.abs().median() < 20:
            df["S"] = df["S"].astype(float) * 100.0

        # canonical name
        df = df.rename(columns={"S": "S_t"})

    # standardize other columns
    mapping = {"DGS10": "DGS10_t", "DGS2": "DGS2_t", "SP500": "SP500_t", "VIX": "VIX_t"}
    for k, v in mapping.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    # Ensure numeric dtype
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_index()


def merge_sources(dfs: dict) -> pd.DataFrame:
    """Merge multiple source DataFrames on the datetime index.

    Uses outer join and forward-fills to create a continuous time series.
    """
    if not dfs:
        return pd.DataFrame()

    frames = []
    for name, df in dfs.items():
        frames.append(df)

    combined = pd.concat(frames, axis=1, join="outer")
    combined = combined.sort_index().ffill()
    return combined


def build_processed_dataset(start: str = "1987-01-01", end: str | None = None, *,
                            resample_weekly: bool = False, save: bool = True,
                            out_path: str | None = None) -> pd.DataFrame:
    """ETL orchestrator: load raw sources, clean, merge and engineer minimal features.

    Returns a DataFrame with canonical columns used by downstream features:
      - S_t: spread level in bps
      - ΔS_t: daily change in bps
      - VIX_t, DGS10_t, DGS2_t, SP500_t

    If `resample_weekly` is True, returns weekly observations taken on Fridays.
    When `save` is True, writes a parquet file to `data/processed/credit_spreads.parquet`
    unless `out_path` is provided.
    
    Default start: 1987-01-01 to capture Black Monday (Oct 1987) and all subsequent crises:
      - 1987: Black Monday
      - 1990-91: Gulf War recession
      - 1994: Mexican Peso Crisis
      - 1997-98: Asian Financial Crisis + LTCM collapse
      - 2000-02: Dot-com bubble burst
      - 2007-09: Global Financial Crisis
      - 2010-12: European Sovereign Debt Crisis
      - 2015-16: China crash, oil collapse
      - 2018: Q4 sell-off
      - 2020: COVID-19 pandemic
      - 2022: Russia-Ukraine, inflation surge
      - 2023: Banking crisis
    """
    sources = load_raw_sources(start=start, end=end)
    # Clean each source
    cleaned = {name: clean_fred(df) for name, df in sources.items()}

    # Merge
    df = merge_sources(cleaned)

    if df.empty:
        raise RuntimeError("No data available after loading sources")

    # Feature engineering: daily change in spread (ΔS_t)
    if "S_t" in df.columns:
        df["ΔS_t"] = df["S_t"].diff()

    # Optionally resample weekly
    if resample_weekly:
        df = resample_to_fridays(df)

    # Save processed dataset
    if save:
        out_path = out_path or "data/processed/credit_spreads.parquet"
        df.to_parquet(out_path)

    return df


# Backward compatibility alias
build_dataset = build_processed_dataset


if __name__ == "__main__":
    # simple runnable fetcher for development
    df = fetch_fred("1987-01-01")  # Extended to capture Black Monday and subsequent crises
    weekly = resample_to_fridays(df)
    print(f"Data coverage: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Total observations: {len(df)} daily, {len(weekly)} weekly")
    print("\nLast 5 weekly observations:")
    print(weekly.tail())
