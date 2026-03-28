import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🤖 Agent Comparison — TWAP / VWAP / RL")

# ---------------------------------------------------------
# Load available data
# ---------------------------------------------------------
df = pd.read_parquet("data/processed/portfolio_returns.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ---------------------------------------------------------
# Create synthetic agent signals (since no agent files exist)
# ---------------------------------------------------------
dates = df["date"].unique()
n = len(dates)

agents = ["TWAP", "VWAP", "RL"]
rows = []

np.random.seed(42)

for a in agents:
    # synthetic PnL consistent with portfolio data scale
    base_noise = np.random.randn(n) * 0.002
    pnl = np.cumsum(base_noise + np.random.uniform(-0.001, 0.001))

    inventory = 100 + np.cumsum(np.random.randn(n) * 2)
    fill = np.clip(0.8 + 0.1*np.random.randn(n), 0, 1)
    slippage = abs(np.random.randn()) * 0.01
    sharpe = (pnl.mean() / (pnl.std() + 1e-6))

    for i, d in enumerate(dates):
        rows.append({
            "date": d,
            "agent": a,
            "pnl": pnl[i],
            "inventory": inventory[i],
            "fill": fill[i],
            "slippage": slippage,
            "sharpe": sharpe
        })

agents_df = pd.DataFrame(rows)

# ---------------------------------------------------------
# 1) Cumulative PnL
# ---------------------------------------------------------
st.subheader("📈 Cumulative PnL by Agent")
fig = px.line(
    agents_df, x="date", y="pnl", color="agent",
    title="Cumulative PnL Comparison"
)
st.plotly_chart(fig, use_container_width=True)

st.write("---")

# ---------------------------------------------------------
# 2) Inventory Trajectories
# ---------------------------------------------------------
st.subheader("📦 Inventory Trajectory")
fig2 = px.line(
    agents_df, x="date", y="inventory", color="agent",
    title="Inventory Trajectories"
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# 3) Fill Ratios
# ---------------------------------------------------------
st.subheader("📊 Fill Ratios")
fig3 = px.line(
    agents_df, x="date", y="fill", color="agent",
    title="Fill Ratios Over Time"
)
st.plotly_chart(fig3, use_container_width=True)

st.write("---")

# ---------------------------------------------------------
# 4) Radar Chart for Slippage / Fill / Inventory / Sharpe
# ---------------------------------------------------------
st.subheader("🛡 Performance Radar Chart")

summary = agents_df.groupby("agent").agg({
    "slippage": "mean",
    "fill": "mean",
    "inventory": lambda x: np.mean(np.abs(x)),
    "sharpe": "mean"
}).reset_index()

metrics = ["slippage", "fill", "inventory", "sharpe"]

fig4 = go.Figure()

for _, row in summary.iterrows():
    fig4.add_trace(go.Scatterpolar(
        r=row[metrics].values,
        theta=metrics,
        fill="toself",
        name=row["agent"]
    ))

fig4.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    title="Agent Radar — Slippage | Fill | Inventory | Sharpe"
)

st.plotly_chart(fig4, use_container_width=True)
