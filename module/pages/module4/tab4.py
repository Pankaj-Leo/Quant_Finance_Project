import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

st.set_page_config(page_title="Risk Decomposition Dashboard", layout="wide")
st.title("📉 Tab 4 – Risk Decomposition (Interactive)")

FACTOR_PATH = "data/processed/factor_library.parquet"
PORTFOLIO_PATH = "data/processed/portfolio_returns.parquet"

# -------------------------------
# Helper to detect factor columns
# -------------------------------
def factor_cols(df):
    return [c for c in df.columns if any(x in c for x in ["momentum", "volatility", "size", "value", "beta"])]

# =========================================================
# 1️⃣ Factor Contributions to Portfolio Variance
# =========================================================
st.header("1️⃣ Factor Contributions to Total Portfolio Variance")

try:
    df_fac = pd.read_parquet(FACTOR_PATH)
    cols = factor_cols(df_fac)
    if not cols:
        st.warning("No factor exposure columns found.")
    else:
        cov = df_fac[cols].cov()
        var_contrib = cov.sum(axis=1)
        var_pct = 100 * var_contrib / var_contrib.sum()
        var_df = pd.DataFrame({"Factor": var_pct.index, "Variance Contribution (%)": var_pct.values})

        fig1 = px.bar(var_df, x="Factor", y="Variance Contribution (%)", 
                      color="Factor", text="Variance Contribution (%)",
                      title="Factor Contributions to Total Variance (%)")
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Could not compute factor variance decomposition: {e}")

# =========================================================
# 2️⃣ Eigenvalue Spectrum (Principal Components)
# =========================================================
st.header("2️⃣ Eigenvalue Spectrum (Principal Components)")

try:
    X = df_fac[cols].fillna(0)
    pca = PCA()
    pca.fit(X)
    explained = 100 * pca.explained_variance_ratio_
    pc_df = pd.DataFrame({
        "PC": np.arange(1, len(explained)+1),
        "Variance Explained (%)": explained
    })

    fig2 = px.bar(pc_df, x="PC", y="Variance Explained (%)", text="Variance Explained (%)",
                  title="Eigenvalue Spectrum (Principal Components)")
    st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Could not compute PCA eigenvalue spectrum: {e}")

# =========================================================
# 3️⃣ Top Contributors to Residual Risk
# =========================================================
st.header("3️⃣ Top Contributors to Residual Risk")

try:
    pca = PCA(n_components=min(5, len(cols)))
    X_pca = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_pca)
    residual = ((X - X_recon) ** 2).mean().sort_values(ascending=False)
    top_resid = residual.head(10).reset_index()
    top_resid.columns = ["Factor", "Residual Variance"]

    fig3 = px.bar(top_resid, x="Factor", y="Residual Variance", text="Residual Variance",
                  title="Top Contributors to Residual Risk", color="Factor")
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Could not compute residual risk: {e}")
