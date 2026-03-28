import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("⚠️ Stress & Latency Tests — PnL, Sharpe, Heatmap")

# ---------------------------------------------------------
# Load available data (portfolio_returns)
# ---------------------------------------------------------
df = pd.read_parquet("data/processed/portfolio_returns.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Use QAOA returns as base "agent" performance
base_ret = df["ret_qaoa"].fillna(0).values

def compute_stats(returns):
    pnl = returns.sum()
    sharpe = returns.mean() / (returns.std() + 1e-6)
    return pnl, sharpe

base_pnl, base_sharpe = compute_stats(base_ret)

# ---------------------------------------------------------
# Stress Scenarios
# ---------------------------------------------------------
vol_levels = [1.0, 1.5, 2.0]         # volatility multipliers
latencies = [0, 5, 10, 20]          # additional ms penalty

results = []

for vol in vol_levels:
    for lat in latencies:
        stressed = base_ret * vol
        
        # latency reduces execution quality => reduce returns
        latency_penalty = 1 - (lat * 0.002)    # every +5ms ≈ -1% performance
        stressed = stressed * latency_penalty
        
        pnl, sharpe = compute_stats(stressed)
        results.append({
            "vol": vol,
            "latency_ms": lat,
            "pnl": pnl,
            "sharpe": sharpe,
            "delta_pnl": pnl - base_pnl,
            "delta_sharpe": sharpe - base_sharpe
        })

res = pd.DataFrame(results)

# ---------------------------------------------------------
# Table of performance deltas
# ---------------------------------------------------------
st.subheader("📉 Performance Delta Table (ΔPnL, ΔSharpe)")
st.dataframe(
    res[["vol","latency_ms","delta_pnl","delta_sharpe"]]
    .style.format("{:.4f}"),
    use_container_width=True
)

st.write("---")

# ---------------------------------------------------------
# Heatmap: PnL vs. Latency × Volatility
# ---------------------------------------------------------
pivot = res.pivot(index="vol", columns="latency_ms", values="pnl")

st.subheader("🔥 PnL Heatmap — Latency × Volatility Regime")

fig_heat = px.imshow(
    pivot,
    labels={"x": "Latency (ms)", "y": "Volatility Multiplier", "color": "PnL"},
    color_continuous_scale="RdBu_r",
    aspect="auto"
)

st.plotly_chart(fig_heat, use_container_width=True)

st.write("---")

# ---------------------------------------------------------
# Stress Line Visual (optional)
# ---------------------------------------------------------
st.subheader("📈 PnL Change by Latency (Grouped by Volatility)")

fig_line = px.line(
    res,
    x="latency_ms",
    y="pnl",
    color="vol",
    markers=True,
    labels={"latency_ms": "Latency (ms)", "pnl": "PnL", "vol": "Volatility ×"},
    title="PnL Under Combined Stress"
)

st.plotly_chart(fig_line, use_container_width=True)
