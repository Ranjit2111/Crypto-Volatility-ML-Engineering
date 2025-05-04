"""
Smoke‑test the Phase 2 preprocessing logic on a tiny synthetic fixture.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import pandas as pd
import ta  # ensure import works


def generate_fixture() -> list[list[int | float]]:
    """
    Build a 5‑day &times; 2‑coin hourly OHLC list mimicking CoinGecko shape:
        [timestamp_ms, open, high, low, close]
    Prices are arbitrary but consistent.
    """
    start = datetime.now(timezone.utc) - timedelta(days=5)
    rows = []
    for coin_idx, coin_price in enumerate([100.0, 50.0]):  # two coins
        for day in range(5):
            for hour in range(24):
                ts = start + timedelta(days=day, hours=hour)
                ts_ms = int(ts.timestamp() * 1000)
                price = coin_price + math.sin(hour)  # wiggle
                rows.append([coin_idx, ts_ms, price, price + 1, price - 1, price + 0.2])
    return rows


def preprocess_fixture(raw_rows: list[list[int | float]]) -> pd.DataFrame:
    df = pd.DataFrame(
        raw_rows, columns=["coin_idx", "timestamp_ms", "open", "high", "low", "close"]
    )
    df["coin"] = df["coin_idx"].map({0: "coinA", 1: "coinB"})
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("datetime")

    daily = (
        df.groupby("coin")
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(subset=["open"])
        .reset_index()
    )
    daily["date"] = daily["datetime"].dt.strftime("%Y-%m-%d")
    daily["return_pct"] = ((daily["close"] - daily["open"]) / daily["open"]) * 100
    daily = daily.sort_values(["coin", "date"])

    daily["ret_std_3d"] = (
        daily.groupby("coin")["return_pct"].rolling(window=3).std().reset_index(level=0, drop=True)
    )
    daily["ma_5d"] = (
        daily.groupby("coin")["close"].rolling(window=5).mean().reset_index(level=0, drop=True)
    )
    daily["ma_10d"] = (
        daily.groupby("coin")["close"].rolling(window=10).mean().reset_index(level=0, drop=True)
    )

    daily["rsi_14d"] = daily.groupby("coin")["close"].transform(
        lambda s: ta.momentum.RSIIndicator(close=s, window=14).rsi()
    )

    daily = daily.dropna()

    daily["next_return"] = daily.groupby("coin")["return_pct"].shift(-1)
    daily["abs_next_return"] = daily["next_return"].abs()
    median_abs = daily.groupby("coin")["abs_next_return"].transform("median")
    daily["target"] = (daily["abs_next_return"] > median_abs).astype(int)
    daily = daily.dropna(subset=["target"])

    return daily[
        [
            "coin",
            "date",
            "return_pct",
            "ret_std_3d",
            "ma_5d",
            "ma_10d",
            "rsi_14d",
            "target",
        ]
    ]


def test_preprocessing_smoke() -> None:
    raw = generate_fixture()
    df = preprocess_fixture(raw)
    # runs without error & returns rows
    assert not df.empty
    expected_cols = {
        "coin",
        "date",
        "return_pct",
        "ret_std_3d",
        "ma_5d",
        "ma_10d",
        "rsi_14d",
        "target",
    }
    assert expected_cols.issubset(df.columns)