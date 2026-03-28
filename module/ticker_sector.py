import pandas as pd
import os

# Path to save
sector_file = "data/raw/ticker_sector.csv"
os.makedirs("data/raw", exist_ok=True)

# Create dataframe
sectors = pd.DataFrame({
    "ticker": ["AAPL", "MSFT", "GOOG", "AMZN"],
    "sector": ["Tech", "Tech", "Tech", "Consumer"]
})

# Save CSV
sectors.to_csv(sector_file, index=False)
print(f"✅ ticker_sector.csv saved to {sector_file}")
sectors
