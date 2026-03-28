import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("📐 Greeks Explorer — Δ, Γ, Vega Surfaces")

# ---------------------------------------
# Black–Scholes Greeks
# ---------------------------------------
def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S,K,T,r,sigma) - sigma*np.sqrt(T)

def delta_call(S, K, T, r, sigma):
    return norm.cdf(d1(S,K,T,r,sigma))

def gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S,K,T,r,sigma)) / (S*sigma*np.sqrt(T))

def vega(S, K, T, r, sigma):
    return S * norm.pdf(d1(S,K,T,r,sigma)) * np.sqrt(T)

# ---------------------------------------
# Controls
# ---------------------------------------
col = st.columns(3)
with col[0]:
    r = st.number_input("Risk-free rate r", value=0.01)
with col[1]:
    K = st.number_input("Strike K", value=100.0)
with col[2]:
    sigma = st.slider("Volatility σ", 0.01, 1.0, 0.2)

S_vals = np.linspace(50, 150, 50)
T_vals = np.linspace(0.03, 2.0, 50)
S_grid, T_grid = np.meshgrid(S_vals, T_vals)

# ---------------------------------------
# Compute Greek Surfaces
# ---------------------------------------
Delta = delta_call(S_grid, K, T_grid, r, sigma)
Gamma = gamma(S_grid, K, T_grid, r, sigma)
Vega  = vega(S_grid, K, T_grid, r, sigma)

# ---------------------------------------
# 3D Plots
# ---------------------------------------
tab1, tab2, tab3 = st.tabs(["Δ Surface", "Γ Surface", "Vega Surface"])

with tab1:
    fig = go.Figure(data=[go.Surface(x=S_grid, y=T_grid, z=Delta, colorscale="Viridis")])
    fig.update_layout(scene=dict(xaxis_title="Spot S", yaxis_title="T", zaxis_title="Delta"))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure(data=[go.Surface(x=S_grid, y=T_grid, z=Gamma, colorscale="Plasma")])
    fig.update_layout(scene=dict(xaxis_title="Spot S", yaxis_title="T", zaxis_title="Gamma"))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = go.Figure(data=[go.Surface(x=S_grid, y=T_grid, z=Vega, colorscale="Blues")])
    fig.update_layout(scene=dict(xaxis_title="Spot S", yaxis_title="T", zaxis_title="Vega"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------
# Delta-Hedged PnL
# ---------------------------------------
st.write("---")
st.subheader("⚖️ Delta-Hedged PnL Under Price Perturbation")

S0 = 100
T0 = 0.5

# Base Greeks
delta0 = delta_call(S0, K, T0, r, sigma)
gamma0 = gamma(S0, K, T0, r, sigma)

# Price shocks
shocks = np.linspace(-5, 5, 100)
pnl = delta0 * shocks + 0.5 * gamma0 * shocks**2

fig_pnl = px.line(
    x=shocks,
    y=pnl,
    labels={"x":"Shock ΔS", "y":"PNL"},
    title="Delta-Hedged Approximate PnL"
)
st.plotly_chart(fig_pnl, use_container_width=True)
