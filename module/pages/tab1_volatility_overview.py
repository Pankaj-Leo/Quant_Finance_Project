# pages/tab1_volatility_overview.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import skew

st.title("📈 Volatility Overview")

# --- Load your data ---
# You can replace this with your own file path
df = pd.read_parquet("data/processed/pairs_signals.parquet")

# --- Basic setup ---
assets = df["pair"].unique() if "pair" in df.columns else ["Synthetic"]
asset_choice = st.selectbox("Select Asset / Pair", assets)
data = df[df["pair"] == asset_choice].copy() if "pair" in df.columns else df.copy()

if "date" not in data.columns:
    st.error("No 'date' column found in data.")
    st.stop()

data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date")

# --- Compute rolling volatility ---
window = st.slider("Rolling Window (Days)", 10, 100, 30)
data["returns"] = data["spread"].pct_change() if "spread" in data.columns else data["signal"].diff()
data["rolling_sigma"] = data["returns"].rolling(window).std()
mean_sigma = data["rolling_sigma"].mean()
vol_of_vol = data["rolling_sigma"].std()
skewness = skew(data["rolling_sigma"].dropna())

# --- ±1σ, ±2σ bands ---
data["+1σ"] = mean_sigma + data["rolling_sigma"]
data["-1σ"] = mean_sigma - data["rolling_sigma"]
data["+2σ"] = mean_sigma + 2 * data["rolling_sigma"]
data["-2σ"] = mean_sigma - 2 * data["rolling_sigma"]

# --- Plot Rolling Volatility ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["date"], y=data["rolling_sigma"], mode="lines", name="Rolling σ", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=data["date"], y=data["+1σ"], mode="lines", name="+1σ Band", line=dict(color="green", dash="dot")))
fig.add_trace(go.Scatter(x=data["date"], y=data["-1σ"], mode="lines", name="-1σ Band", line=dict(color="green", dash="dot")))
fig.add_trace(go.Scatter(x=data["date"], y=data["+2σ"], mode="lines", name="+2σ Band", line=dict(color="red", dash="dot")))
fig.add_trace(go.Scatter(x=data["date"], y=data["-2σ"], mode="lines", name="-2σ Band", line=dict(color="red", dash="dot")))

fig.update_layout(title=f"Rolling Realized Volatility — {asset_choice}",
                  xaxis_title="Date",
                  yaxis_title="Volatility (σ)",
                  template="plotly_dark")

st.plotly_chart(fig, use_container_width=True)

# --- Quick Stats ---
st.subheader("📊 Quick Volatility Stats")
st.metric("Mean σ", f"{mean_sigma:.4f}")
st.metric("Volatility of Volatility", f"{vol_of_vol:.4f}")
st.metric("Skewness", f"{skewness:.4f}")

st.success("✅ Volatility Overview Loaded Successfully")
