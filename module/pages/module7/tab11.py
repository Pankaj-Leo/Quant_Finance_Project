import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 Pricing Sandbox")

# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    option_type = st.selectbox("Option Type", ["Call", "Put"])
    S0 = st.number_input("Spot S0", value=100.0)

with col2:
    K = st.number_input("Strike K", value=100.0)
    T = st.number_input("Maturity T (years)", value=1.0)

with col3:
    r = st.number_input("Risk-free Rate r", value=0.01)
    sigma0 = st.slider("Volatility σ", 0.01, 1.00, 0.2)

vol_model = st.radio("Volatility Model", ["Constant", "Local Vol"], horizontal=True)
alpha = st.slider("Local Vol Curvature α", 0.0, 1.5, 0.3) if vol_model == "Local Vol" else 0.0

# Local vol function
def local_sigma(S):
    return sigma0 * (1 + alpha * ((S / S0) - 1)**2)

# ---------------------------------------------------------
# Analytic BSM
# ---------------------------------------------------------
def bs_price(S, K, T, r, sigma, typ="Call"):
    if T <= 0:
        return max(0.0, (S-K) if typ=="Call" else (K-S))
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if typ == "Call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

bs_val = bs_price(S0, K, T, r, sigma0, option_type)

# ---------------------------------------------------------
# Monte Carlo Pricing
# ---------------------------------------------------------
def mc_price(n_paths, n_steps=200):
    dt = T / n_steps
    S = np.full(n_paths, S0)
    Z = np.random.randn(n_paths, n_steps)

    for t in range(n_steps):
        sigma = local_sigma(S) if vol_model == "Local Vol" else sigma0
        S = S * np.exp((r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z[:, t])

    payoff = np.maximum(S - K, 0) if option_type == "Call" else np.maximum(K - S, 0)
    return np.exp(-r * T) * np.mean(payoff)

mc_val = mc_price(10000, 200)

# ---------------------------------------------------------
# PDE Pricing (Crank–Nicolson Approximation)
# ---------------------------------------------------------
def pde_price(M=200, N=200):
    S_max = 4 * K
    dS = S_max / M
    dt = T / N

    S_vals = np.linspace(0, S_max, M + 1)
    V = np.maximum(S_vals - K, 0) if option_type == "Call" else np.maximum(K - S_vals, 0)

    for step in range(N):
        V_new = V.copy()
        for i in range(1, M):
            sigma = sigma0
            delta = (V[i+1] - V[i-1]) / (2*dS)
            gamma = (V[i+1] - 2*V[i] + V[i-1]) / (dS*dS)
            V_new[i] = V[i] + dt * (0.5*sigma*sigma*S_vals[i]**2 * gamma + r*S_vals[i]*delta - r*V[i])
        V = V_new

    return np.interp(S0, S_vals, V)

pde_val = pde_price()

# ---------------------------------------------------------
# Output Cards
# ---------------------------------------------------------
st.subheader("Pricing Results")

c1, c2, c3 = st.columns(3)
c1.metric("Monte Carlo", f"{mc_val:.5f}")
c2.metric("PDE Price", f"{pde_val:.5f}")
c3.metric("Analytic BSM", f"{bs_val:.5f}")

# ---------------------------------------------------------
# Convergence Plot
# ---------------------------------------------------------
st.subheader("Convergence Analysis")
mode = st.radio("Converge vs", ["MC Paths", "PDE Time Steps"], horizontal=True)

if mode == "MC Paths":
    paths_list = np.unique(np.logspace(2, 5, 10).astype(int))
    prices = [mc_price(p) for p in paths_list]

    fig = px.line(
        x=paths_list,
        y=prices,
        log_x=True,
        labels={"x": "# Paths", "y": "Price"},
        title="MC Price Convergence"
    )
    fig.add_hline(y=bs_val, line_dash="dash", annotation_text="BSM Price")
    st.plotly_chart(fig, use_container_width=True)

else:
    steps_list = [10, 30, 60, 120, 240, 400]
    prices = [pde_price(N=n) for n in steps_list]

    fig = px.line(
        x=steps_list,
        y=prices,
        labels={"x": "Time Steps (Δt)", "y": "Price"},
        title="PDE Price Convergence"
    )
    fig.add_hline(y=bs_val, line_dash="dash", annotation_text="BSM Price")
    st.plotly_chart(fig, use_container_width=True)
