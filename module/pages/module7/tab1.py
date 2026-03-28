import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("📈 Market Replay — LOB Dynamics")

# -------------------------------------------------
# 1️⃣ Select Data Source
# -------------------------------------------------
mode = st.radio(
    "Select LOB Data",
    ["Real LOB", "Synthetic LOB"],
    horizontal=True
)

if mode == "Real LOB":
    lob = np.load("data/processed/lob_sequences.npy")   # shape: (T, levels, 2)
else:
    lob = np.load("data/processed/lob_synthetic_timegan.npy")

T, levels, _ = lob.shape

st.write(f"Loaded LOB: {T} timesteps, {levels} price levels")

# -------------------------------------------------
# 2️⃣ Animated Depth Heatmap
# -------------------------------------------------
st.subheader("📊 LOB Depth Heatmap (Animated)")

fig = px.imshow(
    lob,
    animation_frame=0,
    color_continuous_scale="Viridis",
    title="Depth Heatmap Over Time (0 = Bids, 1 = Asks)"
)
st.plotly_chart(fig, use_container_width=True)

st.write("---")

# -------------------------------------------------
# 3️⃣ Midprice + Spread Evolution
# -------------------------------------------------
best_bid = lob[:, 0, 0]
best_ask = lob[:, 0, 1]

mid = (best_bid + best_ask) / 2
spread = best_ask - best_bid

# Midprice plot
fig_mid = px.line(
    x=np.arange(T),
    y=mid,
    labels={"x": "Time", "y": "Midprice"},
    title="📈 Midprice Evolution"
)
st.plotly_chart(fig_mid, use_container_width=True)

# Spread plot
fig_spread = px.line(
    x=np.arange(T),
    y=spread,
    labels={"x": "Time", "y": "Spread"},
    title="📉 Bid-Ask Spread Evolution"
)
st.plotly_chart(fig_spread, use_container_width=True)
