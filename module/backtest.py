import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

save_path = "data/processed"
alpha_path = os.path.join(save_path, "alpha_predictions.parquet")
factor_path = os.path.join(save_path, "factor_library.parquet")

# --------------------------
# ✅ Load Data (cached)
# --------------------------
alpha = pd.read_parquet(alpha_path)
factors = pd.read_parquet(factor_path)
merged_path = os.path.join(save_path, "merged.parquet")
input("Testing")
if os.path.exists(merged_path):
    merged = pd.read_parquet(merged_path)
else:
    print(factors.shape, alpha.shape)
    print(factors['ticker'].nunique(), alpha['ticker'].nunique())
    print(factors['ticker'].dtype, alpha['ticker'].dtype)
    input("Testing on merege")
    # merged = factors.merge(alpha, on="ticker", how="inner")
    merged = factors.head(10000).merge(alpha.head(10000), on="ticker", how="inner")
    print(merged.shape)
    input("Step 2")
    
    if "date_y" in merged.columns:
        merged.rename(columns={"date_y": "date"}, inplace=True)
    merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
    merged.to_parquet(merged_path)

print(f"✅ Merged shape: {merged.shape}")

# --------------------------
# 🧮 Compute or load IC
# --------------------------
ic_path = os.path.join(save_path, "ic_results.parquet")

def compute_ic(df):
    ic_values = []
    for date, group in df.groupby("date"):
        group = group.dropna(subset=["alpha", "ret"])
        if group["alpha"].nunique() > 1:
            ic, _ = spearmanr(group["alpha"], group["ret"])
            ic_values.append({"date": date, "IC": ic})
    return pd.DataFrame(ic_values)

if os.path.exists(ic_path):
    ic_df = pd.read_parquet(ic_path)
else:
    ic_df = compute_ic(merged)
    ic_df.to_parquet(ic_path)

print(f"📊 Mean Daily IC: {ic_df['IC'].mean():.4f}" if not ic_df.empty else "⚠️ IC not computed.")

# --------------------------
# 💰 Backtest Portfolio
# --------------------------
port_path = os.path.join(save_path, "portfolio_returns.parquet")

def backtest_portfolio(df, top_quantile=0.3, bottom_quantile=0.3):
    portfolio_returns = []
    for date, group in df.groupby("date"):
        group = group.dropna(subset=["alpha", "ret"])
        if group["alpha"].nunique() <= 1 or len(group) < 5:
            continue
        q_hi = group["alpha"].quantile(1 - top_quantile)
        q_lo = group["alpha"].quantile(bottom_quantile)
        long = group[group["alpha"] >= q_hi]["ret"].mean()
        short = group[group["alpha"] <= q_lo]["ret"].mean()
        daily_ret = long - short
        portfolio_returns.append({"date": date, "daily_ret": daily_ret})
    return pd.DataFrame(portfolio_returns)

if os.path.exists(port_path):
    port_ret = pd.read_parquet(port_path)
else:
    port_ret = backtest_portfolio(merged)
    port_ret.to_parquet(port_path)

if not port_ret.empty:
    sharpe = port_ret["daily_ret"].mean() / port_ret["daily_ret"].std() * np.sqrt(252)
    print(f"⚙️ Sharpe Ratio: {sharpe:.2f}")

# --------------------------
# 📊 Plot
# --------------------------
if not port_ret.empty:
    port_ret["cum_ret"] = (1 + port_ret["daily_ret"]).cumprod() - 1
    plt.figure(figsize=(8, 4))
    plt.plot(port_ret["date"], port_ret["cum_ret"], color="blue", linewidth=2)
    plt.title("Cumulative Return Curve (Long–Short Portfolio)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_path, "cumulative_returns.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved plot: {plot_path}")

print("🎯 Backtest complete.")
