import numpy as np
import pandas as pd
from pathlib import Path

def simulate_lob_data(n_steps=10000):
    np.random.seed(42)
    mid_price = np.cumsum(np.random.randn(n_steps)) + 100
    spread = np.abs(np.random.randn(n_steps) * 0.05)
    ofi = np.random.randn(n_steps) * 10  # order flow imbalance

    return pd.DataFrame({
        "mid_price": mid_price,
        "spread": spread,
        "ofi": ofi,
    })

if __name__ == "__main__":
    data = simulate_lob_data()
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    data.to_parquet("data/raw/lob_synthetic.parquet")
    print("✅ Synthetic LOB data saved at data/raw/lob_synthetic.parquet")
