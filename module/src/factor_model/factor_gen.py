import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# -------------------------------------------------------
# Factor Computation
# -------------------------------------------------------
def compute_factors(df):
    df = df.sort_values('date').copy()
    df['ret'] = df['Close'].pct_change(fill_method=None)
    df['momentum_12m'] = df['Close'] / df['Close'].shift(252) - 1
    df['momentum_3m'] = df['Close'] / df['Close'].shift(63) - 1
    df['volatility_60d'] = df['ret'].rolling(60).std()

    # Only compute size if Volume exists
    if 'Volume' in df.columns:
        df['size'] = np.log(df['Close'] * df['Volume'])
    else:
        df['size'] = np.nan
        print(f"⚠️ Warning: 'Volume' missing for {df['ticker'].iloc[0]}, 'size' factor set to NaN")

    return df

# -------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------
def main():
    raw_path = "data/raw/ohlcv.parquet"
    save_path = "data/processed"
    os.makedirs(save_path, exist_ok=True)

    # Load OHLCV data
    ohlcv = pd.read_parquet(raw_path)

    # Ensure Date is a column
    if ohlcv.index.name in ['Date', 'date']:
        ohlcv = ohlcv.reset_index()

    # Flatten MultiIndex columns if they exist
    if isinstance(ohlcv.columns, pd.MultiIndex):
        ohlcv.columns = [
            '_'.join([str(c) for c in col if c]).strip('_')
            for col in ohlcv.columns
        ]

    # ---------------------------------------------------
    # Robust ticker detection
    # ---------------------------------------------------
    ohlcv_suffixes = {'Open', 'High', 'Low', 'Close', 'Volume'}
    tickers = set()
    for col in ohlcv.columns:
        for suffix in ohlcv_suffixes:
            if suffix in col:
                ticker = col.replace(suffix, '').replace('_', '').strip()
                if ticker:  # avoid empty strings
                    tickers.add(ticker)
    tickers = sorted(tickers)

    if not tickers:
        print("❌ Column names detected:", list(ohlcv.columns))
        raise ValueError("No tickers detected! Check your column names.")

    print("Detected tickers:", tickers)

    # ---------------------------------------------------
    # Reshape wide -> long
    # ---------------------------------------------------
    long_list = []
    for t in tickers:
        subcols = [c for c in ohlcv.columns if t in c]
        tmp = ohlcv[['Date'] + subcols].copy()

        # Map canonical column names
        col_map = {}
        for c in subcols:
            for suffix in ohlcv_suffixes:
                if suffix in c:
                    col_map[c] = suffix
        tmp.rename(columns=col_map, inplace=True)

        tmp['date'] = tmp['Date']
        tmp['ticker'] = t
        keep_cols = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']
        tmp = tmp[[c for c in keep_cols if c in tmp.columns]]
        long_list.append(tmp)

    if not long_list:
        raise ValueError("No data to concatenate. Check your column detection logic.")

    ohlcv_long = pd.concat(long_list, ignore_index=True)

    # ---------------------------------------------------
    # Compute factors
    # ---------------------------------------------------
    results = []
    for t in tqdm(ohlcv_long['ticker'].unique(), desc="Computing factors"):
        df = ohlcv_long[ohlcv_long['ticker'] == t].copy()
        df = compute_factors(df)
        results.append(df)

    full = pd.concat(results, ignore_index=True)

    # ---------------------------------------------------
    # Save factor library
    # ---------------------------------------------------
    file_path = os.path.join(save_path, "factor_library.parquet")
    full.to_parquet(file_path)
    print(f"✅ Saved factor library to {file_path}")

# -------------------------------------------------------
# Run
# -------------------------------------------------------
if __name__ == "__main__":
    main()
