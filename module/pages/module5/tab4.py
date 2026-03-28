import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("📉 Backtest & Comparison — MV / RP / QAOA")

# -------------------------------------------------
# Load portfolio backtest results
# -------------------------------------------------
df = pd.read_parquet("data/processed/portfolio_returns.parquet")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# -------------------------------------------------
# 1️⃣ Rolling Sharpe (60-day)
# -------------------------------------------------
window = 60

df["sharpe_mv"] = df["ret_mv"].rolling(window).mean() / df["ret_mv"].rolling(window).std()
df["sharpe_rp"] = df["ret_rp"].rolling(window).mean() / df["ret_rp"].rolling(window).std()
df["sharpe_qaoa"] = df["ret_qaoa"].rolling(window).mean() / df["ret_qaoa"].rolling(window).std()

fig_sharpe = px.line(
    df,
    x="date",
    y=["sharpe_mv", "sharpe_rp", "sharpe_qaoa"],
    labels={"value": "Sharpe", "variable": "Strategy"},
    title="Rolling Sharpe Ratio (60-day)"
)
st.plotly_chart(fig_sharpe, use_container_width=True)

st.write("---")

# -------------------------------------------------
# 2️⃣ Cumulative PnL
# -------------------------------------------------
df["pnl_mv"] = (1 + df["ret_mv"]).cumprod()
df["pnl_rp"] = (1 + df["ret_rp"]).cumprod()
df["pnl_qaoa"] = (1 + df["ret_qaoa"]).cumprod()

fig_pnl = px.line(
    df,
    x="date",
    y=["pnl_mv", "pnl_rp", "pnl_qaoa"],
    labels={"value": "Cumulative PnL", "variable": "Strategy"},
    title="Cumulative PnL Comparison"
)
st.plotly_chart(fig_pnl, use_container_width=True)

st.write("---")

# -------------------------------------------------
# 3️⃣ Drawdown Heatmap
# -------------------------------------------------
dd = df[["dd_mv","dd_rp","dd_qaoa"]].T
dd.index = ["MV","RP","QAOA"]

fig_dd = px.imshow(
    dd,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Drawdown Heatmap (MV / RP / QAOA)"
)
st.plotly_chart(fig_dd, use_container_width=True)

st.write("---")

# -------------------------------------------------
# 4️⃣ Turnover & Exposure Stability Table
# -------------------------------------------------
summary = pd.DataFrame({
    "Strategy": ["MV","RP","QAOA"],
    "Avg Turnover": [
        df["turnover_mv"].mean(),
        df["turnover_rp"].mean(),
        df["turnover_qaoa"].mean()
    ],
    "Return Std": [
        df["ret_mv"].std(),
        df["ret_rp"].std(),
        df["ret_qaoa"].std()
    ]
})

st.subheader("📊 Turnover & Stability Comparison")
st.dataframe(summary, use_container_width=True)
