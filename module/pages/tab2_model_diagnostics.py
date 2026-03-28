# pages/tab2_model_diagnostics.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")
st.title("🔍 Model Diagnostics — GARCH vs EGARCH vs ML")

# --- Load Data ---
df = pd.read_parquet("data/processed/pairs_signals.parquet")
pair = st.selectbox("Select Pair", df['pair'].unique())
data = df[df['pair'] == pair].copy()

data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date")
data["returns"] = data["spread"].pct_change().dropna()

# --- Fit Models ---
returns = data["returns"].dropna() * 100
garch = arch_model(returns, vol="Garch", p=1, q=1).fit(disp="off")
egarch = arch_model(returns, vol="EGarch", p=1, q=1).fit(disp="off")
ml_forecast = np.sqrt(returns.rolling(30).var())  # placeholder ML model

data["GARCH_vol"] = garch.conditional_volatility
data["EGARCH_vol"] = egarch.conditional_volatility
data["ML_vol"] = ml_forecast

# --- Layout in Grid ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Volatility Forecast Overlay")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"], y=data["GARCH_vol"], name="GARCH", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=data["date"], y=data["EGARCH_vol"], name="EGARCH", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=data["date"], y=data["ML_vol"], name="ML Forecast", line=dict(color="magenta")))
    fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Forecasted Volatility")
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("📉 Residual QQ & ACF")
    resid = garch.resid / garch.conditional_volatility

    fig1, ax1 = plt.subplots()
    qqplot(resid, line='s', ax=ax1)
    st.pyplot(fig1, use_container_width=True)

    fig2, ax2 = plt.subplots()
    plot_acf(resid.dropna(), ax=ax2, lags=30)
    st.pyplot(fig2, use_container_width=True)

# --- AIC/BIC Comparison ---
st.subheader("📊 Model Selection Metrics (AIC/BIC)")
aic_bic = pd.DataFrame({
    "Model": ["GARCH(1,1)", "EGARCH(1,1)", "ML (Simulated)"],
    "AIC": [garch.aic, egarch.aic, np.nan],
    "BIC": [garch.bic, egarch.bic, np.nan]
})
best_model = aic_bic.loc[aic_bic["AIC"].idxmin(), "Model"]
st.dataframe(aic_bic.style.highlight_min(subset=["AIC", "BIC"], color="lightgreen"))
st.success(f"✅ Best Model Based on AIC: **{best_model}**")

st.caption("ML forecast is simulated — replace later with your real ML model output.")
