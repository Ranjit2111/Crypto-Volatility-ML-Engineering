#!/usr/bin/env python3
"""
Create nice‑looking price & volatility PNGs for each coin
 over three horizons: 1y, 30d, 1d.

Plots saved to plots/{coin}_{period}.png
"""
from __future__ import annotations
import time
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.dates import DateFormatter
from crypto_vol.config import COINS

OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models")
FEATURE_IMPORTANCE_FILE = MODEL_DIR / "feature_importance.json"

VS_CURRENCY = "usd"

# Dark theme style parameters
DARK_BACKGROUND = "#1e1e1e"
TEXT_COLOR = "#E0E0E0"
ACCENT_COLOR = "#00BCD4"
GRID_COLOR = "#424242"
PIE_COLORS = ['#00BCD4', '#4DD0E1', '#80DEEA', '#B2EBF2', '#00ACC1', '#0097A7', '#00838F', '#006064', '#26C6DA', '#4DB6AC']

def ts_log(level: str, msg: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {msg}")

def fetch_ohlc(coin: str, days: int) -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/ohlc?vs_currency={VS_CURRENCY}&days={days}"
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                ts_log("WARN", f"Rate limit hit for {coin}. Waiting 60 seconds before retry...")
                time.sleep(60)
                continue
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close"])
            df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df.set_index("datetime")
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == max_attempts:
                ts_log("ERROR", f"Failed to fetch data for {coin} after {max_attempts} attempts: {e}")
                raise
            ts_log("WARN", f"Error fetching {coin}, attempt {attempt}/{max_attempts}. Retrying in 60s...")
            time.sleep(60)

def make_plot(df: pd.DataFrame, coin: str, period_name: str) -> None:
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = DARK_BACKGROUND
    plt.rcParams['axes.facecolor'] = DARK_BACKGROUND
    plt.rcParams['axes.edgecolor'] = GRID_COLOR
    plt.rcParams['axes.labelcolor'] = TEXT_COLOR
    plt.rcParams['xtick.color'] = TEXT_COLOR
    plt.rcParams['ytick.color'] = TEXT_COLOR
    plt.rcParams['text.color'] = TEXT_COLOR
    plt.rcParams['grid.color'] = GRID_COLOR
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["close"], linewidth=2, color=ACCENT_COLOR)
    plt.title(f"{coin.capitalize()} Price – {period_name}", color=TEXT_COLOR, fontsize=16)
    plt.xlabel("Date", color=TEXT_COLOR, fontsize=12)
    plt.ylabel(f"{VS_CURRENCY.upper()}", color=TEXT_COLOR, fontsize=12)
    plt.grid(True, alpha=0.3, color=GRID_COLOR)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)

    plt.gcf().autofmt_xdate()
    out_path = OUT_DIR / f"{coin}_{period_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=DARK_BACKGROUND)
    plt.close()
    ts_log("INFO", f"Saved plot -> {out_path}")

def make_feature_importance_plot() -> None:
    if not FEATURE_IMPORTANCE_FILE.exists():
        ts_log("WARN", f"Feature importance file not found at {FEATURE_IMPORTANCE_FILE}. Skipping plot.")
        return

    with FEATURE_IMPORTANCE_FILE.open() as f:
        importances = json.load(f)

    if not importances:
        ts_log("WARN", "Feature importance data is empty. Skipping plot.")
        return

    # Sort by importance and select top N (e.g., top 7 for pie chart clarity)
    sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    top_n = 7
    labels = [item[0] for item in sorted_importances[:top_n]]
    sizes = [item[1] for item in sorted_importances[:top_n]]

    if len(sorted_importances) > top_n:
        other_sum = sum(item[1] for item in sorted_importances[top_n:])
        labels.append("Other Features")
        sizes.append(other_sum)
    
    custom_pie_colors = PIE_COLORS[:len(labels)]

    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = DARK_BACKGROUND
    plt.rcParams['axes.facecolor'] = DARK_BACKGROUND
    plt.rcParams['text.color'] = TEXT_COLOR
    plt.rcParams['patch.edgecolor'] = DARK_BACKGROUND

    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=None,
        autopct='%1.1f%%', 
        startangle=140, 
        colors=custom_pie_colors,
        pctdistance=0.85,
        wedgeprops={'edgecolor': DARK_BACKGROUND, 'linewidth': 1.5}
    )

    for text in texts:
        text.set_color(TEXT_COLOR)
    for autotext in autotexts:
        autotext.set_color('#111111')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    ax.axis('equal')
    plt.title("Feature Importance Distribution", color=TEXT_COLOR, fontsize=16, pad=20)
    
    legend = ax.legend(
        wedges, labels,
        title="Features",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10,
        title_fontsize=12,
        labelcolor=TEXT_COLOR,
        facecolor=DARK_BACKGROUND,
        edgecolor=GRID_COLOR
    )
    plt.setp(legend.get_texts(), color=TEXT_COLOR)
    plt.setp(legend.get_title(), color=TEXT_COLOR)

    out_path = OUT_DIR / "feature_importance_pie.png"
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(out_path, dpi=150, facecolor=DARK_BACKGROUND)
    plt.close()
    ts_log("INFO", f"Saved feature importance plot -> {out_path}")

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
    
    ts_log("INFO", "Generating feature importance plot...")
    make_feature_importance_plot()

if __name__ == "__main__":
    main()