import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Optimization Overview – Tab 1")

# ---------------------------
# Load your files
# ---------------------------
merged = pd.read_parquet("data/processed/merged.parquet")
port   = pd.read_parquet("data/processed/portfolio_returns.parquet")

print(port.head())

# ---------------------------
# Fix date columns
# ---------------------------
merged["date"] = pd.to_datetime(merged["date_x"])
merged = merged.drop(columns=["date_x"])

# ---------------------------
# Clean required columns only
# ---------------------------
merged = merged[["date", "ticker", "alpha", "ret"]].dropna()

# Rename ret for clarity
merged = merged.rename(columns={"ret": "return"})

# ---------------------------
# Return matrix
# ---------------------------
returns = (
    merged.pivot_table(
        index="date",
        columns="ticker",
        values="return",
        aggfunc="mean"
    )
    .fillna(0)
)

# ---------------------------
# Alpha vector μ
# ---------------------------
alpha_vec = merged.groupby("ticker")["alpha"].mean()

# ---------------------------
# Covariance matrix Σ
# ---------------------------
cov_matrix = returns.cov()

# ---------------------------
# Heatmap: Alpha
# ---------------------------
st.subheader("Alpha (μ) Heatmap")
fig1, ax1 = plt.subplots(figsize=(7,2))
ax1.imshow(alpha_vec.values.reshape(1,-1), cmap="viridis")
ax1.set_xticks(range(len(alpha_vec)))
ax1.set_xticklabels(alpha_vec.index, rotation=90)
ax1.set_yticks([])
st.pyplot(fig1)

# ---------------------------
# Heatmap: Covariance Σ
# ---------------------------
st.subheader("Covariance Matrix (Σ)")
fig2, ax2 = plt.subplots(figsize=(7,6))
ax2.imshow(cov_matrix, cmap="viridis")
ax2.set_xticks(range(len(cov_matrix)))
ax2.set_yticks(range(len(cov_matrix)))
ax2.set_xticklabels(cov_matrix.index, rotation=90)
ax2.set_yticklabels(cov_matrix.index)
st.pyplot(fig2)

# ---------------------------
# Efficient Frontier Comparison
# ---------------------------
st.subheader("Efficient Frontier – MV vs RP vs QAOA")

fig3, ax3 = plt.subplots(figsize=(7,4))
ax3.plot(port["vol_mv"], port["ret_mv"], label="MV")
ax3.plot(port["vol_rp"], port["ret_rp"], label="RP")
ax3.plot(port["vol_qaoa"], port["ret_qaoa"], label="QAOA")
ax3.set_xlabel("Volatility")
ax3.set_ylabel("Return")
ax3.legend()
st.pyplot(fig3)

# ---------------------------
# KPI Metrics
# ---------------------------
st.subheader("Key Performance Metrics")
st.write(port[[
    "sharpe_mv","sharpe_rp","sharpe_qaoa",
    "dd_mv","dd_rp","dd_qaoa",
    "turnover_mv","turnover_rp","turnover_qaoa"
]])
