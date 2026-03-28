# pages/tab1_factor_universe.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Factor Universe")

# --- Load Core Factor Data ---
factor_df = pd.read_parquet("data/processed/factor_library.parquet")

st.sidebar.header("⚙️ Settings")
include_gnn = st.sidebar.checkbox("Include GNN Latent Factors", value=False)

# --- Add GNN Factors if Selected ---
if include_gnn:
    try:
        gnn_emb = np.load("data/processed/node_embeddings.npy")
        gnn_df = pd.DataFrame(gnn_emb, columns=[f"GNN_{i+1}" for i in range(gnn_emb.shape[1])])
        factor_df = pd.concat([factor_df.reset_index(drop=True), gnn_df], axis=1)
        st.sidebar.success(f"✅ Added {gnn_emb.shape[1]} GNN latent factors")
    except Exception as e:
        st.sidebar.error(f"⚠️ Could not load GNN factors: {e}")

# --- Display Factor Table ---
st.subheader("Factor Overview")
st.dataframe(factor_df.head(10), use_container_width=True)

# --- Compute Numeric Correlation ---
numeric_df = factor_df.select_dtypes(include=[np.number])
if numeric_df.empty:
    st.warning("⚠️ No numeric columns found for correlation.")
else:
    st.subheader("📈 Pairwise Factor Correlations")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, cbar=True)
    st.pyplot(fig)

# --- Quick Stats ---
st.subheader("📊 Quick Stats Summary")
stats = {}
for col in ["value", "momentum", "volatility", "quality", "size"]:
    if col in factor_df.columns:
        stats[f"Mean {col.title()}"] = factor_df[col].mean()

st.write(pd.DataFrame([stats]))

st.caption("Toggle GNN factors to include graph-based latent signals in the analysis.")
