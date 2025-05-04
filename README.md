# Crypto‑Volatility‑Watcher 🪙📈

> **Predict which cryptocurrency is most likely to experience the highest absolute price move tomorrow (UTC).**

---

## 0. Why?

Volatility traders need an automated hint on *where* to look each morning.
This MVP fetches recent OHLC data for 10 major coins, engineers signals, trains an **XGBoost** classifier, and exposes the daily probabilities via a FastAPI endpoint.

---

## 1. Project Structure (final)

crypto-volatility-watcher/
├── api/               # FastAPI service
├── data/              # raw & processed artifacts
├── models/            # trained model + CV metrics
├── scripts/           # ETL + ML pipeline
├── tests/             # pytest smoke test
├── run_pipeline.sh    # one-shot launcher
├── requirements.txt
└── README.md          # you are here

---

## 2. Quick start (local)

```bash
git clone https://github.com/<yourusername>/crypto-volatility-watcher.git
cd crypto-volatility-watcher
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run the full pipeline
./run_pipeline.sh

# serve the API (loads latest prediction JSON)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
