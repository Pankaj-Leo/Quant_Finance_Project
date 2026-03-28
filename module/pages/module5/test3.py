import pandas as pd

# Use your real file
df = pd.read_parquet("data/processed/factor_library.parquet")

# Find correct date column
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
elif "date_x" in df.columns:
    df["date"] = pd.to_datetime(df["date_x"])
else:
    raise Exception("No valid date column found.")

# Identify price column
if "Close" in df.columns:
    price_col = "Close"
else:
    raise Exception("No close price column found.")

# Clean data
df = df.dropna(subset=["ticker", price_col])
df = df.sort_values(["ticker", "date"])

# Compute returns
df["ret"] = df.groupby("ticker")[price_col].pct_change()
df = df.dropna(subset=["ret"])

# Save cleaned price/return dataset
df.to_parquet("data/processed/clean_prices.parquet", index=False)

print("Saved clean_prices.parquet")
