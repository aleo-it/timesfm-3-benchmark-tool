"""Dataset loaders for the forecasting benchmark."""

import ssl
from dataclasses import dataclass
from pathlib import Path

import certifi
import pandas as pd
import numpy as np


def _make_ssl_context() -> ssl.SSLContext:
    """Build an SSL context using certifi's CA bundle.

    The standard macOS Python.org build does not link the system cert
    store, so HTTPS downloads from GitHub fail cert verification. Pinning
    to certifi's bundle resolves this.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    return ctx


def _download(url: str, dest: Path, ssl_context: ssl.SSLContext) -> None:
    """Download a file to dest using the given SSL context."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "forecasting-lab/0.1"})
    with urllib.request.urlopen(req, context=ssl_context) as resp:
        data = resp.read()
    dest.write_bytes(data)


@dataclass
class Dataset:
    """Standardized dataset format."""

    series_id: str
    y: pd.Series  # With DatetimeIndex
    freq: str  # Pandas frequency string
    horizon: int  # Forecast horizon


def load_m4_monthly(
    n_series: int = 100, cache_dir: str | Path = "data"
) -> list[Dataset]:
    """Load M4 Monthly competition dataset.

    Downloads from M4 competition official data. Returns a list of Dataset
    objects with standardized format.

    Args:
        n_series: Number of series to load (M4 Monthly has 48,000 total).
        cache_dir: Directory to cache downloaded data.

    Returns:
        List of Dataset objects.
    """
    import urllib.request

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ssl_ctx = _make_ssl_context()

    train_url = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv"
    info_url = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/M4-info.csv"
    train_file = cache_dir / "m4_monthly_train.csv"
    info_file = cache_dir / "m4_info.csv"

    if not train_file.exists():
        print("Downloading M4 Monthly training set (~90MB)...")
        _download(train_url, train_file, ssl_ctx)

    if not info_file.exists():
        print("Downloading M4 metadata...")
        _download(info_url, info_file, ssl_ctx)

    # M4-info.csv: M4id, category, Frequency(1=Y,4=Q,12=M,24=H), Horizon, SP, StartingDate
    info_df = pd.read_csv(info_file)
    monthly_info = info_df[info_df["Frequency"] == 12].set_index("M4id")

    horizon = 18

    df = pd.read_csv(train_file)
    series_list = []
    for _idx, row in df.head(n_series).iterrows():
        series_id = str(row.iloc[0])
        values = row.iloc[1:].dropna().values.astype(float)

        # Use actual starting date from M4-info
        start = "2000-01-01"
        if series_id in monthly_info.index:
            raw_start = monthly_info.loc[series_id, "StartingDate"]
            try:
                start = pd.to_datetime(raw_start).strftime("%Y-%m-%d")
            except Exception:
                pass

        dates = pd.date_range(start=start, periods=len(values), freq="MS")
        y = pd.Series(values, index=dates, name="y")
        series_list.append(
            Dataset(series_id=series_id, y=y, freq="MS", horizon=horizon)
        )

    return series_list


def load_airline(
    csv_path: str | Path = "data/airline.csv",
) -> Dataset:
    """Load the classic Airline Passengers dataset (1949-01 to 1960-12, monthly)."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    y = df.set_index("date")["passengers"]

    return Dataset(
        series_id="Airline-Passengers",
        y=y,
        freq="MS",
        horizon=18,
    )


def _retail_frequency() -> str:
    """Return the pandas frequency string for retail data."""
    return "W-MON"


def load_retail(
    n_series: int | None = None,
    csv_path: str | Path = "data/retail.csv",
    fill: float = 0.0,
) -> list[Dataset]:
    """Load retail sales data as weekly time series (one dataset per product).

    The retail CSV has columns Product;week;month;year;unit. Each unique product
    becomes a weekly time series. Missing weeks (absent rows) are filled with
    `fill` to produce a contiguous index.

    Args:
        n_series: Number of products to load (default: all).
        csv_path: Path to the retail CSV.
        fill: Value used to fill missing weeks (default 0.0).

    Returns:
        List of weekly Dataset objects, one per product.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Retail CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2) + "-1",
        format="%Y-W%W-%w",
    )

    freq = _retail_frequency()
    horizon = 13  # one quarter of weekly data

    datasets = []
    for product in df["Product"].unique():
        if n_series is not None and len(datasets) >= n_series:
            break

        p = df[df["Product"] == product].sort_values("date")
        full_index = pd.date_range(p["date"].min(), p["date"].max(), freq=freq)
        series = p.set_index("date")["unit"].reindex(full_index).fillna(fill)

        datasets.append(
            Dataset(
                series_id=product,
                y=series.astype(float),
                freq=freq,
                horizon=horizon,
            )
        )

    return datasets
