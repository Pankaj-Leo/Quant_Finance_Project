import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Load files YOU actually have
# ---------------------------------------------------------
merged = pd.read_parquet("data/processed/merged.parquet")

merged["date"] = pd.to_datetime(merged["date_x"])
merged = merged.drop(columns=["date_x"])

merged = merged[["date", "ticker", "alpha", "ret"]].dropna()

# Returns table
returns = merged.pivot_table(index="date", columns="ticker", values="ret").fillna(0)

# Alpha vector
mu = merged.groupby("ticker")["alpha"].mean()

# Covariance matrix
Sigma = returns.cov()
inv = np.linalg.pinv(Sigma.values)

# ---------------------------------------------------------
# Weight Construction
# ---------------------------------------------------------
# MV portfolio
w_mv = inv @ mu.values
w_mv = w_mv / w_mv.sum()

# Risk Parity
n = len(mu)
w_rp = np.ones(n) / n

# QAOA (mock)
w_q = (w_mv + w_rp) / 2
w_q = w_q / w_q.sum()

# ---------------------------------------------------------
# Portfolio Returns
# ---------------------------------------------------------
port_mv   = returns @ w_mv
port_rp   = returns @ w_rp
port_qaoa = returns @ w_q

# Rolling Vol
vol_mv = port_mv.rolling(20).std()
vol_rp = port_rp.rolling(20).std()
vol_q  = port_qaoa.rolling(20).std()

# ---------------------------------------------------------
# Drawdown
# ---------------------------------------------------------
def max_drawdown(series):
    cum = series.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    return dd.min()

dd_mv = max_drawdown(port_mv)
dd_rp = max_drawdown(port_rp)
dd_q  = max_drawdown(port_qaoa)

# ---------------------------------------------------------
# Turnover (weight change)
# ---------------------------------------------------------
turnover_mv = np.sum(np.abs(np.diff(w_mv)))
turnover_rp = np.sum(np.abs(np.diff(w_rp)))
turnover_q  = np.sum(np.abs(np.diff(w_q)))

# ---------------------------------------------------------
# Build final DataFrame
# ---------------------------------------------------------
df = pd.DataFrame({
    "date": returns.index,
    "ret_mv": port_mv.values,
    "vol_mv": vol_mv.values,
    "ret_rp": port_rp.values,
    "vol_rp": vol_rp.values,
    "ret_qaoa": port_qaoa.values,
    "vol_qaoa": vol_q.values,
})

df["sharpe_mv"] = df["ret_mv"] / df["vol_mv"]
df["sharpe_rp"] = df["ret_rp"] / df["vol_rp"]
df["sharpe_qaoa"] = df["ret_qaoa"] / df["vol_qaoa"]

# Single-value KPIs (added to DataFrame tail)
df["dd_mv"] = dd_mv
df["dd_rp"] = dd_rp
df["dd_qaoa"] = dd_q

df["turnover_mv"] = turnover_mv
df["turnover_rp"] = turnover_rp
df["turnover_qaoa"] = turnover_q

df.to_parquet("data/processed/portfolio_returns.parquet", index=False)
print("Saved: data/processed/portfolio_returns.parquet")
