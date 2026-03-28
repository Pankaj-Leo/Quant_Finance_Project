import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time

st.set_page_config(layout="wide")
st.title("🎲 Monte Carlo Diagnostics")

# ---------------------------------------------------------
# Parameters
# ---------------------------------------------------------
col = st.columns(4)
with col[0]:
    S0 = st.number_input("Spot S0", value=100.0)
with col[1]:
    K = st.number_input("Strike K", value=100.0)
with col[2]:
    r = st.number_input("Rate r", value=0.01)
with col[3]:
    sigma = st.slider("Vol σ", 0.01, 1.0, 0.2)

T = 1.0
N = st.slider("Number of Paths", 1_000, 200_000, 20_000)

# ---------------------------------------------------------
# 1️⃣ Monte Carlo Payoff (Base vs Control Variates)
# ---------------------------------------------------------
st.subheader("📊 Histogram of Discounted Payoffs (MC)")

np.random.seed(42)
Z = np.random.randn(N)
ST = S0 * np.exp((r - 0.5*sigma*sigma)*T + sigma*np.sqrt(T)*Z)

payoff = np.maximum(ST - K, 0)
disc_payoff = np.exp(-r*T) * payoff

fig1 = px.histogram(
    disc_payoff,
    nbins=45,
    title="Discounted Payoffs Distribution (MC)"
)
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------
# 2️⃣ Variance Reduction: Control Variates
# ---------------------------------------------------------
st.subheader("📉 Variance Reduction — Base vs Control Variates")

# Control variate uses ST itself (analytic expectation = S0*exp(rT))
control_var = ST
control_mean = S0*np.exp(r*T)

cov = np.cov(disc_payoff, control_var)[0,1]
var_cv = np.var(control_var)
beta = cov / var_cv

cv_estimator = disc_payoff - beta*(control_var - control_mean)

summary_vr = pd.DataFrame({
    "Method": ["Base MC", "Control Variate"],
    "Mean": [disc_payoff.mean(), cv_estimator.mean()],
    "Std Dev": [disc_payoff.std(), cv_estimator.std()]
})

st.dataframe(summary_vr, use_container_width=True)

fig2 = px.histogram(
    cv_estimator,
    nbins=45,
    title="Control Variate Adjusted Payoffs"
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# 3️⃣ CPU vs GPU Timing
# ---------------------------------------------------------
st.subheader("⚙️ GPU Timing Diagnostics (CPU vs GPU)")

cpu_start = time.time()
_ = S0 * np.exp((r - 0.5*sigma*sigma)*T + sigma*np.sqrt(T)*np.random.randn(N))
cpu_time = time.time() - cpu_start

# GPU test (if cupy available)
try:
    import cupy as cp
    gpu_available = True
except:
    gpu_available = False

if gpu_available:
    cp_start = time.time()
    Zg = cp.random.randn(N)
    _ = S0 * cp.exp((r - 0.5*sigma*sigma)*T + sigma*cp.sqrt(T)*Zg)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - cp_start
else:
    gpu_time = None

timing_table = pd.DataFrame({
    "Device": ["CPU", "GPU"],
    "Time (s)": [cpu_time, gpu_time if gpu_available else "N/A"]
})

st.dataframe(timing_table, use_container_width=True)

st.success("Monte Carlo diagnostics calculated successfully.")
 