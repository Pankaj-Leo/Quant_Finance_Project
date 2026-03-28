import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("📊 Allocations & Risk — MV / RP / QAOA")

# -------------------------------------------------
# Load returns
# -------------------------------------------------
# df = pd.read_parquet("data/processed/factor_library.parquet")
df = pd.read_parquet("data/processed/clean_prices.parquet")

df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["ret", "ticker"])

# Keep only rows with valid prices
df = df.dropna(subset=["Open","High","Low","Close"])

# -------------------------------------------------
# Build returns matrix
# -------------------------------------------------
returns = df.pivot(index="date", columns="ticker", values="ret")
returns = returns.dropna(axis=1, how="any")   # keep stable tickers only

tickers = list(returns.columns)

# -------------------------------------------------
# Covariance (PD safe)
# -------------------------------------------------
def make_pd(A, j=1e-6):
    A = (A + A.T)/2
    for _ in range(20):
        try:
            np.linalg.cholesky(A)
            return A
        except:
            A += np.eye(A.shape[0])*j
            j *= 2
    return A

sigma = make_pd(returns.cov().values)

# -------------------------------------------------
# MV / RP / QAOA Weights
# -------------------------------------------------
vol = np.sqrt(np.diag(sigma))
w_rp = pd.Series((1/vol)/np.sum(1/vol), index=tickers)

inv = np.linalg.inv(sigma + np.eye(len(tickers))*1e-4)
w_mv = pd.Series(inv @ np.ones(len(tickers)), index=tickers)
w_mv = w_mv / w_mv.sum()

w_qaoa = (w_mv + w_rp) / 2
w_qaoa = w_qaoa / w_qaoa.sum()

# -------------------------------------------------
# 1) Allocation Comparison
# -------------------------------------------------
alloc_df = pd.DataFrame({
    "Ticker": tickers,
    "MV": w_mv.values,
    "RP": w_rp.values,
    "QAOA": w_qaoa.values
})

fig = px.bar(
    alloc_df.melt(id_vars="Ticker", var_name="Strategy", value_name="Weight"),
    x="Ticker", y="Weight", color="Strategy",
    barmode="group",
    title="Strategy Allocation Comparison"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# 2) Sector Exposure
# -------------------------------------------------
sect = pd.read_csv("data/raw/ticker_sector.csv")
sect["ticker"] = sect["ticker"].str.upper()

merged = pd.DataFrame({"ticker": tickers, "weight": w_qaoa.values})
merged = merged.merge(sect, on="ticker", how="left")

sector_weights = merged.groupby("sector")["weight"].sum()

fig2 = px.pie(
    names=sector_weights.index,
    values=sector_weights.values,
    title="QAOA Sector Exposure"
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# 3) Risk Contribution
# -------------------------------------------------
w = w_qaoa.values
mrc = sigma @ w
rc = w * mrc

fig3 = go.Figure(go.Waterfall(x=tickers, y=rc))
fig3.update_layout(title="Risk Contribution (QAOA)")
st.plotly_chart(fig3, use_container_width=True)
