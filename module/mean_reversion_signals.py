# Step 0: Install necessary packages if you haven't already
# pip install pandas numpy statsmodels

import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# 1️⃣ Load OHLCV data
df = pd.read_parquet('data/raw/ohlcv.parquet')
print("Columns in your DataFrame:\n", df.columns)

# 2️⃣ Extract Close prices from MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    close_prices = df['Close']  # level=0 'Price'
else:
    close_prices = df[['AAPL', 'MSFT', 'GOOG', 'AMZN']]  # fallback

# Drop tickers that are all NaN
close_prices = close_prices.dropna(axis=1, how='all')
tickers = close_prices.columns.tolist()
print("\nClose prices extracted:\n", close_prices.head())
print("Valid tickers:", tickers)

# 3️⃣ Generate all unique pairs
pairs = list(itertools.combinations(tickers, 2))
print("\nPairs to test:", pairs)

# 4️⃣ Initialize results
results = []

# 5️⃣ Cointegration test + spread + z-score
for i, j in pairs:
    # Keep only overlapping data
    combined = close_prices[[i, j]].dropna()
    if combined.empty:
        print(f"⚠️ Skipping pair {i}-{j}: no overlapping data")
        continue

    y = combined[i]
    X = sm.add_constant(combined[j])

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        print(f"⚠️ Skipping pair {i}-{j}: regression failed ({e})")
        continue

    beta = model.params[1]
    spread = y - beta * combined[j]

    # ADF test on residual (spread)
    adf_result = adfuller(spread)
    p_value = adf_result[1]

    if p_value < 0.05:
        spread_mean = spread.mean()
        spread_std = spread.std()
        zscore = (spread - spread_mean) / spread_std

        signal = np.where(zscore > 2, -1, np.where(zscore < -2, 1, 0))
        signal = pd.Series(signal, index=spread.index)

        results.append(pd.DataFrame({
            'date': spread.index,
            'pair': f'{i}-{j}',
            'zscore': zscore,
            'signal': signal,
            'p_value': p_value,
            'beta': beta,
            'spread': spread
        }))
    else:
        print(f"⚠️ Skipping pair {i}-{j}: not cointegrated (p={p_value:.4f})")

# 6️⃣ Concatenate and save
if results:
    pairs_signals = pd.concat(results).reset_index(drop=True)
    pairs_signals.to_parquet('data/processed/pairs_signals.parquet')
    print("\n✅ Mean-reversion signals generated and saved!")
else:
    print("\n⚠️ No cointegrated pairs found!")
