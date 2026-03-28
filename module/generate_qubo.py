import numpy as np
import pandas as pd
import os

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading data...")

# Load OHLCV, alpha predictions, and graph adjacency
ohlcv = pd.read_parquet("data/raw/ohlcv.parquet")
alpha = pd.read_parquet("data/processed/alpha_predictions.parquet")
graph_adj = np.load("data/processed/graph_adj.npy")

# Flatten multi-index columns if present
if isinstance(ohlcv.columns, pd.MultiIndex):
    ohlcv.columns = ['_'.join([str(i) for i in col]).strip() for col in ohlcv.columns]

# Check tickers
tickers = sorted(alpha['ticker'].unique())
num_assets = len(tickers)
print("Tickers:", tickers)
print("Number of assets:", num_assets)
print("Graph adjacency shape:", graph_adj.shape)

# -----------------------------
# EXTRACT CLOSE PRICES
# -----------------------------
# Find all close price columns
close_cols = [c for c in ohlcv.columns if 'Close' in c]
price = ohlcv[close_cols]

# Rename columns to tickers
price.columns = [c.split('_')[-1] for c in price.columns]

# Keep only tickers in alpha
price = price[[t for t in tickers if t in price.columns]]

# Forward fill missing values and drop rows with any remaining NaNs
price = price.fillna(method='ffill').dropna()

# -----------------------------
# COMPUTE RETURNS & COVARIANCE
# -----------------------------
rets = price.pct_change().dropna()
cov = rets.cov().values

# Ensure cov matches number of assets
cov = cov[:num_assets, :num_assets]

# -----------------------------
# EXTRACT RETURNS FROM ALPHA
# -----------------------------
# Take the latest alpha value per ticker
latest_alpha = alpha.sort_values('date').groupby('ticker').tail(1).set_index('ticker')
returns = latest_alpha.loc[tickers, 'alpha'].values

# -----------------------------
# BUILD QUBO MATRIX
# -----------------------------
print("Building QUBO matrix...")

risk_aversion = 0.2
graph_penalty = 0.1

Q = np.zeros((num_assets, num_assets))

# 1. Risk term (quadratic)
Q += risk_aversion * cov

# 2. Return term (linear, on diagonal)
Q -= np.diag(returns)

# 3. Graph penalty (diversification)
Q += graph_penalty * graph_adj[:num_assets, :num_assets]

# Convert to DataFrame for saving
qubo_df = pd.DataFrame(Q, index=tickers, columns=tickers)

# -----------------------------
# SAVE QUBO MATRIX
# -----------------------------
save_path = "data/processed/qubo_matrix.parquet"
os.makedirs("data/processed", exist_ok=True)
qubo_df.to_parquet(save_path)
print("Saved QUBO matrix to:", save_path)
