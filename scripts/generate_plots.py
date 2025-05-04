#!/usr/bin/env python3
"""
Create nice‑looking price & volatility PNGs for each coin
 over three horizons: 1y, 30d, 1d.

Plots saved to plots/{coin}_{period}.png
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.dates import DateFormatter

from config import COINS

OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

VS_CURRENCY = "usd"

def ts_log(level: str, msg: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}")

def fetch_ohlc(coin: str, days: int) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc?vs_currency={VS_CURRENCY}&days={days}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("datetime")

def make_plot(df: pd.DataFrame, coin: str, period_name: str) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(df.index, df["close"], linewidth=2)
    plt.title(f"{coin.capitalize()} price – {period_name}")
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.grid(alpha=0.3)
    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y‑%m‑%d"))
    plt.gcf().autofmt_xdate()
    out_path = OUT_DIR / f"{coin}_{period_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    ts_log("INFO", f"Saved plot -> {out_path}")

def main() -> None:
    horizons = {
        "1y": 365,
        "30d": 30,
        "1d": 1,
    }
    for coin in COINS:
        ts_log("INFO", f"Generating plots for {coin} ...")
        for tag, days in horizons.items():
            df = fetch_ohlc(coin, days)
            make_plot(df, coin, tag)

if __name__ == "__main__":
    main()