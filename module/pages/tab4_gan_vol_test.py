import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

st.title("🤖 GAN Volatility Test — Real vs Synthetic Comparison")

# --- Load Data ---
df = pd.read_parquet("data/processed/pairs_signals.parquet")
pair = st.selectbox("Select Pair", df["pair"].unique())
data = df[df["pair"] == pair].copy()
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date")
data["returns"] = data["spread"].pct_change().dropna()

# --- Simulated GAN Volatility (placeholder) ---
np.random.seed(42)
real_vol = data["returns"].rolling(30).std().dropna()
gan_vol = real_vol * (1 + np.random.normal(0, 0.1, len(real_vol)))  # mock GAN σ

# --- ACF(|r|) Comparison ---
st.subheader("📈 ACF(|r|) — Real vs GAN")
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_acf(np.abs(data["returns"].dropna()), lags=30, ax=ax1)
ax1.set_title("Real Data |r| ACF")
plot_acf(np.abs(np.random.normal(0, real_vol.mean(), len(real_vol))), lags=30, ax=ax2)
ax2.set_title("GAN Synthetic |r| ACF")
st.pyplot(fig1)

# --- Distribution Overlay of Rolling σ ---
st.subheader("🎯 Distribution of Rolling Volatility")
fig2, ax3 = plt.subplots(figsize=(8, 4))
sns.kdeplot(real_vol, label="Real", fill=True, color="cyan")
sns.kdeplot(gan_vol, label="GAN", fill=True, color="magenta")
ax3.set_title("Rolling σ Distribution: Real vs GAN")
ax3.legend()
st.pyplot(fig2)

# --- Cluster Persistence Comparison ---
st.subheader("📊 Cluster Persistence — Regime Duration")
real_clusters = pd.cut(real_vol, bins=3, labels=["Low", "Medium", "High"])
gan_clusters = pd.cut(gan_vol, bins=3, labels=["Low", "Medium", "High"])
real_persistence = real_clusters.value_counts(normalize=True)
gan_persistence = gan_clusters.value_counts(normalize=True)

persistence_df = pd.DataFrame({
    "Regime": real_persistence.index,
    "Real": real_persistence.values,
    "GAN": gan_persistence.values
})

fig3, ax4 = plt.subplots(figsize=(7, 4))
persistence_df.plot(x="Regime", kind="bar", ax=ax4)
ax4.set_title("Cluster Persistence Comparison")
ax4.set_ylabel("Proportion")
st.pyplot(fig3)

st.caption("🧬 Replace simulated GAN σ with real GAN model outputs once trained.")
