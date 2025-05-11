# Crypto Volatility Watcher

Predicts daily crypto volatility using machine learning, featuring a FastAPI backend and data processing pipeline. This project demonstrates an end-to-end MLOps workflow, from data ingestion and feature engineering to model tuning, prediction, and API deployment.

## Project Structure

```
crypto-volatility-watcher/
├── api/
│   └── main.py             # FastAPI application
├── crypto_vol/
│   ├── scripts/
│   │   ├── fetch_data.py       # Fetches raw coin data
│   │   ├── preprocess.py       # Cleans data and engineers features
│   │   ├── tune_model.py       # Tunes XGBoost model and selects features
│   │   ├── predict.py          # Generates predictions with the tuned model
│   │   └── generate_plots.py   # Creates price and feature importance plots
│   ├── config.py           # Configuration for coins, API keys (if any), etc.
│   └── __init__.py
├── data/
│   ├── raw/                  # Stores raw downloaded data (e.g., coin_data.csv)
│   ├── processed/            # Stores features.csv after preprocessing
│   └── predictions_*.json    # Stores daily predictions from the tuned model
├── models/
│   ├── tuned_model.pkl       # Saved tuned XGBoost model, scaler, and important features
│   ├── best_xgb_params.json  # Best hyperparameters for XGBoost
│   ├── feature_importance.json # Feature importances from the tuned model
│   └── tuned_metrics.json    # Performance metrics of the tuned model
├── plots/
│   ├── *.png                 # Generated coin price plots and feature importance plot
├── .gitignore                # Specifies intentionally untracked files
├── README.md                 # This file
├── requirements.txt          # Project dependencies
├── run_pipeline.bat          # Batch script to run the pipeline on Windows
└── run_pipeline.sh           # Shell script to run the pipeline (e.g., on Linux/EC2)
```

## Project Evolution & Model Performance

This project underwent several iterations to improve predictive performance:

### 1. Initial Model (Untuned XGBoost with Simple Features)

The first version utilized an XGBoost model with a basic set of features. While it provided a baseline, its performance was modest. The key metrics for this initial model were (as found in `models/metrics.json` before it was removed):

* **Mean Accuracy**: ~0.57
* **Mean Precision**: ~0.46
* **Mean Recall**: ~0.38
* **Mean F1-Score**: ~0.41

These results indicated significant room for improvement, particularly in identifying true positive cases (recall) and the overall balance between precision and recall (F1-score).

### 2. Advanced Feature Engineering & Hyperparameter Tuning

To enhance performance, a significant effort was dedicated to feature engineering. The goal was to provide the model with more informative signals derived from raw price and volume data. Key engineered features include:

- **Realized Volatility (5d, 10d, 30d):** Historical volatility calculated over different rolling windows. Important for capturing recent and medium-term price fluctuation trends.
 **Exponential Weighted Moving Average (EWMA) Volatility (5d, 10d, 30d):** Similar to realized volatility but gives more weight to recent observations.
- **Parkinson Volatility:** A volatility measure using high and low prices, often considered more efficient than close-to-close volatility.
- **RSI (Relative Strength Index - 14d):** A momentum oscillator that measures the speed and change of price movements, indicating overbought or oversold conditions.
- **MACD (Moving Average Convergence Divergence):** A trend-following momentum indicator showing the relationship between two moving averages of a security's price.
- **Bollinger Bands (BB Width, BB Position):** Measures volatility and provides a relative definition of high and low prices. `BB Width` indicates market tightness, and `BB Position` shows where the current price is relative to the bands.
- **Momentum (5d, 10d):** Rate of acceleration of a security's price.
Following feature engineering, the XGBoost model's hyperparameters were tuned using `RandomizedSearchCV` with time-series aware cross-validation. Feature selection was also performed, retaining only the most impactful features for the final model. This iterative process led to a substantial improvement in performance, as reflected in `models/tuned_metrics.json`:


* **Mean Accuracy**: ~0.89
* **Mean Precision**: ~0.70
* **Mean Recall**: ~0.71
* **Mean F1-Score**: ~0.69
* **Mean ROC AUC**: ~0.88

These metrics represent excellent performance considering the inherent challenges of cryptocurrency volatility prediction:

1. **High Noise-to-Signal Ratio**: Cryptocurrency markets are notoriously noisy with frequent random fluctuations unrelated to underlying patterns, making reliable prediction extremely difficult.

2. **Non-Stationarity**: Crypto markets evolve rapidly with changing correlations and volatility regimes, causing models to degrade if not constantly updated.

3. **External Influence Factors**: Regulatory news, social media sentiment, and market manipulation can cause sudden price movements that technical indicators alone cannot anticipate.

4. **Market Inefficiency**: Unlike traditional markets, crypto markets operate 24/7 with varying liquidity across exchanges, creating price discrepancies that add prediction complexity.

5. **Black Swan Events**: Crypto markets are prone to extreme, unpredictable events (exchange hacks, protocol failures) that dramatically impact prices and volatility.

Achieving ~0.89 accuracy and ~0.70 precision/recall balance in this environment demonstrates that our model has successfully captured meaningful patterns despite these challenges, making it a valuable tool for volatility forecasting.


### 3. Ensemble Model Exploration

An ensemble model combining XGBoost, RandomForest, and Logistic Regression was also explored. While the ensemble performed significantly better than the initial untuned model, its metrics were slightly below those of the standalone tuned XGBoost model. The additional complexity and marginal performance decrease led to the decision to proceed with the tuned XGBoost model for production.

### 4. Production Model: Tuned XGBoost

The final production model is the tuned XGBoost classifier, leveraging the engineered features and selected hyperparameters. Code related to the initial untuned model and the ensemble experiments has been streamlined from the repository to maintain clarity and focus on the production-ready pipeline.

## Plots

The system generates several plots to visualize market data and model insights:

* **Coin Price Plots:** For each coin, OHLC (Open, High, Low, Close) price plots are generated for 1-day, 30-day, and 1-year horizons. These plots are dark-themed to align with modern dashboard aesthetics.
* **Feature Importance Plot:** A dark-themed pie chart visualizing the relative importance of the features used by the tuned XGBoost model.

These plots are accessible via the API.

## API Endpoints

The project exposes a FastAPI to serve predictions and plots:

* `/`: Welcome message.
* `/health`: Health check for the API.
* `/predict`: Returns the latest volatility predictions from the tuned XGBoost model.
* `/coins`: Lists the coins currently configured in the system.
* `/plot/{coin}?period={period}`: Serves the price plot for the specified `coin` and `period` (1d, 30d, 1y).
* `/plot/feature_importance`: Serves the feature importance pie chart.

## Pipeline Workflow (`run_pipeline.sh` / `run_pipeline.bat`)

The core data and modeling pipeline consists of the following steps:

1. **Fetch Coin Data (`fetch_data.py`):** Downloads the latest historical price/volume data for configured cryptocurrencies.
2. **Preprocess Data (`preprocess.py`):** Cleans the raw data, engineers the advanced features listed above, and prepares the dataset for modeling.
3. **Tune XGBoost Model (`tune_model.py`):** Performs hyperparameter tuning for the XGBoost classifier using time-series cross-validation and selects the most important features. Saves the tuned model, scaler, selected features, and performance metrics.
4. **Make Predictions (`predict.py`):** Loads the tuned model and latest preprocessed data to generate daily volatility predictions.
5. **Generate Plots (`generate_plots.py`):** Creates and saves the dark-themed price and feature importance plots.

## Getting Started

### Prerequisites

* Python 3.8+
* Pip (Python package installer)

### Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Ranjit2111/crypto-volatility-watcher.git
   cd crypto-volatility-watcher
   ```
2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

* **Locally (Windows):**
  ```batch
  run_pipeline.bat
  ```
* **Locally (Linux/macOS) or on a Server (e.g., AWS EC2):**
  ```bash
  chmod +x run_pipeline.sh
  ./run_pipeline.sh
  ```

### Running the API

After running the pipeline (which generates necessary model files and predictions):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

You can then access the API at `http://localhost:8000`.

## Deployed Endpoints

* **Backend API URL:** `[YOUR_AWS_EC2_FASTAPI_ENDPOINT_HERE]`
* **Frontend URL:** `[YOUR_VERCEL_FRONTEND_URL_HERE]`

## Contributing

Contributions, issues, and feature requests are welcome! Please feel free to check the issues page.
