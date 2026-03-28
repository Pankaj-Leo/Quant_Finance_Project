import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("🌋 Surface & Regimes — Volatility Structure")

# --- Load Data ---
df = pd.read_parquet("data/processed/pairs_signals.parquet")
pair = st.selectbox("Select Pair", df["pair"].unique())
data = df[df["pair"] == pair].copy()

data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date")

# --- Simulate Vol Surface (strike × maturity × σ) ---
st.subheader("3D Implied Volatility Surface (SABR/Heston Style)")

strikes = np.linspace(80, 120, 25)
maturities = np.linspace(0.1, 2, 25)
strike_grid, maturity_grid = np.meshgrid(strikes, maturities)

# Fake surface – replace later with model
sigma_surface = 0.2 + 0.1 * np.exp(-((strike_grid - 100)**2 / 200 + (maturity_grid - 1)**2))

fig = go.Figure(data=[go.Surface(
    x=strike_grid,
    y=maturity_grid,
    z=sigma_surface,
    colorscale="Viridis"
)])
fig.update_layout(
    title=f"Implied Volatility Surface — {pair}",
    scene=dict(
        xaxis_title="Strike",
        yaxis_title="Maturity (Years)",
        zaxis_title="Volatility (σ)"
    ),
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# --- Regime Heatmap (σ Clustering) ---
st.subheader("Market Regime Heatmap (Volatility Clusters)")

data["volatility"] = data["spread"].pct_change().rolling(20).std()
data = data.dropna(subset=["volatility"])

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
data["regime"] = kmeans.fit_predict(data[["volatility"]])

pivot = data.pivot_table(index="date", values="volatility", aggfunc="mean")

fig2, ax = plt.subplots(figsize=(10, 3))
plt.scatter(data["date"], data["volatility"], c=data["regime"], cmap="viridis", s=15)
plt.title("Volatility Regime Heatmap")
plt.xlabel("Date")
plt.ylabel("Volatility (σ)")
st.pyplot(fig2)

st.caption("🧠 Replace simulated SABR/Heston surface with your model outputs when ready.")
