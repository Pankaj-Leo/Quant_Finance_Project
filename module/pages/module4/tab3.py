import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("📊 Factor Portfolios")

# -----------------------------
# LOAD MAIN FACTOR DATA
# -----------------------------
df = pd.read_parquet("data/processed/factor_library.parquet")
df["date"] = pd.to_datetime(df["date"])

# clean
df = df.dropna(subset=["ret"])

# ---------------------------------
# BUILD FACTORS FROM AVAILABLE DATA
# ---------------------------------
df["value"] = 1 / (df["size"] + 1e-6)
df["momentum"] = df["momentum_12m"]
df["short_momentum"] = df["momentum_3m"]
df["volatility"] = df["volatility_60d"]
df["size_factor"] = df["size"]

factors = ["value", "momentum", "short_momentum", "volatility", "size_factor"]

# ---------------------------------
# FACTOR RETURNS: mean by date
# ---------------------------------
factor_returns = df.groupby("date")[factors].mean().sort_index()

# ---------------------------------
# ROLLING SHARPE (252d)
# ---------------------------------
rolling_sharpe = (
    factor_returns.rolling(252).mean()
    / factor_returns.rolling(252).std()
)

# ---------------------------------
# CUMULATIVE RETURNS
# ---------------------------------
cum_ret = (1 + factor_returns).cumprod()

# ---------------------------------
# DUMMY SECTOR & STYLE EXPOSURE
# ---------------------------------
np.random.seed(42)

sector_exposure = pd.DataFrame({
    "sector": ["Tech", "Finance", "Energy", "Healthcare"],
    "exposure": np.abs(np.random.rand(4))
})
sector_exposure["exposure"] /= sector_exposure["exposure"].sum()

style_exposure = pd.DataFrame({
    "style": factors,
    "exposure": np.abs(np.random.rand(len(factors)))
})
style_exposure["exposure"] /= style_exposure["exposure"].sum()

# ---------------------------------
# TURNOVER (change in factor returns)
# ---------------------------------
turnover = factor_returns.diff().abs().mean(axis=1)

# ---------------------------------
# CAPACITY (inverse of volatility)
# ---------------------------------
capacity = 1 / (factor_returns.std(axis=1) + 1e-6)

# ---------------------------------
# PLOTS
# ---------------------------------

st.subheader("Rolling Sharpe (252d)")
st.plotly_chart(px.line(rolling_sharpe), use_container_width=True)

st.subheader("Cumulative Returns of Factor Portfolios")
st.plotly_chart(px.line(cum_ret), use_container_width=True)

st.subheader("Sector Exposure")
st.plotly_chart(px.pie(sector_exposure, names="sector", values="exposure"), use_container_width=True)

st.subheader("Style Exposure")
st.plotly_chart(px.pie(style_exposure, names="style", values="exposure"), use_container_width=True)

st.subheader("Turnover")
st.plotly_chart(px.line(turnover, labels={"value": "Turnover"}), use_container_width=True)

st.subheader("Capacity Proxy")
st.plotly_chart(px.line(capacity, labels={"value": "Capacity"}), use_container_width=True)
