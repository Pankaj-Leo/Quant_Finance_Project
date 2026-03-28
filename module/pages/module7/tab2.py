import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from scipy.interpolate import griddata

st.set_page_config(layout="wide")
st.title("🧭 Volatility Surface — Implied vs Local")

# -------------------------------------------------
# 0️⃣ Black–Scholes & Implied Vol Solver
# -------------------------------------------------
def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    if sigma <= 0:
        return max(S - K, 0)

    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return S * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_vol(price, S, K, T, r):
    if price < max(S-K, 0):
        return 0.0

    sigma = 0.2
    for _ in range(40):
        price_est = bs_call(S, K, T, r, sigma)
        vega = S * norm.pdf((np.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*np.sqrt(T))) * np.sqrt(T)

        if vega < 1e-8:
            break

        sigma -= (price_est - price) / vega
        sigma = max(sigma, 1e-6)

    return float(sigma)

# -------------------------------------------------
# 1️⃣ Load or Synthesize Option Data
# -------------------------------------------------
st.sidebar.header("Data Input")
use_real = st.sidebar.checkbox("Load real options file", value=False)

opt = None

if use_real:
    uploaded = st.sidebar.file_uploader("Upload CSV/Parquet", type=["csv", "parquet"])
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                opt = pd.read_csv(uploaded)
            else:
                opt = pd.read_parquet(uploaded)
            st.sidebar.success("Loaded real option data.")
        except:
            st.sidebar.error("Failed to load file.")

# Synthetic fallback
if opt is None:
    st.sidebar.info("No file → using synthetic volatility surface.")
    S0 = 100
    Ks = np.linspace(60,140,20)
    Ts = np.linspace(0.05,2.0,20)
    rows = []
    for T in Ts:
        for K in Ks:
            base = 0.18 + 0.05*np.exp(-T)
            smile = base*(1 + 0.3*((K/S0)-1)**2)
            price = bs_call(S0, K, T, 0.01, smile)
            rows.append({"K":K,"T":T,"mid_price":price,"S":S0})
    opt = pd.DataFrame(rows)

# -------------------------------------------------
# 2️⃣ Compute Implied Vol Surface
# -------------------------------------------------
r = 0.01

if "imp_vol" in opt.columns:
    opt["iv"] = opt["imp_vol"]
elif "mid_price" in opt.columns:
    opt["iv"] = opt.apply(lambda row: implied_vol(
        row["mid_price"],
        float(row.get("S",100)),
        float(row["K"]),
        float(row["T"]),
        r
    ), axis=1)
else:
    st.error("Option file must contain 'mid_price' or 'imp_vol'.")
    st.stop()

Ks = np.sort(opt["K"].unique())
Ts = np.sort(opt["T"].unique())
K_grid, T_grid = np.meshgrid(Ks, Ts)

points = opt[["K","T"]].values
values = opt["iv"].values

IV_grid = griddata(points, values, (K_grid,T_grid), method="cubic")

mask = np.isnan(IV_grid)
IV_grid[mask] = griddata(points, values, (K_grid[mask], T_grid[mask]), method="nearest")

# -------------------------------------------------
# 3️⃣ Local Vol via Dupire (Safe Approximation)
# -------------------------------------------------
def dupire_local_vol(Kg, Tg, ivg):
    epsK = (Kg[0,1] - Kg[0,0]) if Kg.shape[1]>1 else 1
    epsT = (Tg[1,0] - Tg[0,0]) if Tg.shape[0]>1 else 1

    w = ivg**2 * Tg
    dw_dT = np.gradient(w, axis=0) / epsT
    dw_dK = np.gradient(w, axis=1) / epsK
    d2w_dK2 = np.gradient(np.gradient(w, axis=1), axis=1) / (epsK**2)

    local = np.zeros_like(ivg)

    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            if w[i,j] <= 1e-8:
                local[i,j] = np.nan
                continue

            denom = 1 - (Kg[i,j]*dw_dK[i,j])/w[i,j] + 0.25*(Kg[i,j]**2)*d2w_dK2[i,j]/w[i,j]
            if denom <= 1e-8 or dw_dT[i,j] <= 0:
                local[i,j] = np.nan
            else:
                local[i,j] = np.sqrt(dw_dT[i,j]/denom)

    return local

LOC_grid = dupire_local_vol(K_grid, T_grid, IV_grid)

# Calibration error = placeholder (same grid)
error_grid = np.abs(IV_grid - IV_grid)

# -------------------------------------------------
# 4️⃣ 3D Surface Plots
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Vol Surfaces", "Contour Slices", "Calibration Error"])

with tab1:
    st.subheader("Implied Vol Surface")
    fig = go.Figure(data=[go.Surface(x=K_grid, y=T_grid, z=IV_grid, colorscale="Viridis")])
    fig.update_layout(scene=dict(xaxis_title="Strike", yaxis_title="Maturity", zaxis_title="IV"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Local Vol Surface (Dupire)")
    fig2 = go.Figure(data=[go.Surface(x=K_grid, y=T_grid, z=LOC_grid, colorscale="Plasma")])
    fig2.update_layout(scene=dict(xaxis_title="Strike", yaxis_title="Maturity", zaxis_title="Local Vol"))
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# 5️⃣ Contour slices
# -------------------------------------------------
with tab2:
    st.subheader("IV Slices")
    slice_T = st.selectbox("Select T", Ts)
    slice_K = st.selectbox("Select K", Ks)

    idxT = np.argmin(np.abs(Ts - slice_T))
    idxK = np.argmin(np.abs(Ks - slice_K))

    fig3 = px.line(x=Ks, y=IV_grid[idxT], title=f"IV Slice @ T={slice_T}")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(x=Ts, y=IV_grid[:,idxK], title=f"IV Slice @ K={slice_K}")
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------------
# 6️⃣ Calibration Error Map
# -------------------------------------------------
with tab3:
    st.subheader("Calibration Error Heatmap")
    fig5 = px.imshow(error_grid, x=Ks, y=Ts, color_continuous_scale="RdBu_r",
                     labels={"x":"Strike","y":"Maturity","color":"Error"})
    st.plotly_chart(fig5, use_container_width=True)
