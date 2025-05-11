#!/usr/bin/env python3
"""
Phase 2: fetch hourly OHLC for 90 days for 10 coins.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from crypto_vol.config import COINS, ML_DAYS
import requests

VS_CURRENCY: Final[str] = "usd"
DAYS: Final[int] = ML_DAYS 
MAX_ATTEMPTS: Final[int] = 3
BACKOFFS: Final[list[int]] = [60, 90, 120] 

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATE_STR: Final[str] = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ts_log(level: str, message: str) -> None:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}")


def fetch_coin(coin_id: str) -> None:
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        f"?vs_currency={VS_CURRENCY}&days={DAYS}"
    )
    ts_log("INFO", f"Fetching data for {coin_id}...")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 429:
                raise requests.HTTPError("HTTP 429 Rate Limit")
            response.raise_for_status()
            data = response.json()
            out_file = RAW_DIR / f"{coin_id}_{DATE_STR}.json"
            with out_file.open("w") as fp:
                json.dump(data, fp)
            ts_log(
                "INFO",
                f"Fetched data for {coin_id} successfully -> {out_file.relative_to(Path('.'))}",
            )
            return
        except (requests.RequestException, json.JSONDecodeError) as exc:
            if attempt >= MAX_ATTEMPTS:
                ts_log(
                    "ERROR",
                    f"Failed to fetch data for {coin_id} after {MAX_ATTEMPTS} attempts: {exc}",
                )
            else:
                delay = BACKOFFS[attempt - 1]
                ts_log(
                    "WARN",
                    f"Attempt {attempt}/{MAX_ATTEMPTS} failed for {coin_id}. "
                    f"Retrying in {delay}s...",
                )
                time.sleep(delay)


def main() -> None:
    for coin in COINS:
        fetch_coin(coin)


if __name__ == "__main__":
    main()