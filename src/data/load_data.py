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


def fetch_fred(start: str = "1990-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch the required FRED series and return a DataFrame with columns [S, DGS10, SP500, VIX].

    Dates are business daily as returned by pandas_datareader.
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


if __name__ == "__main__":
    # simple runnable fetcher for development
    df = fetch_fred("1995-01-01")
    weekly = resample_to_fridays(df)
    print(weekly.tail())
