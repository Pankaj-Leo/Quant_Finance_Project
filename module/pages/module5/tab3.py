import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("⚛️ Quantum Diagnostics — QAOA & QUBO")

# -------------------------------------------------
# Load QUBO Matrix
# -------------------------------------------------
qubo = pd.read_parquet("data/processed/qubo_matrix.parquet")
qubo_mat = qubo.values

qubo_labels = [f"Var {i}" for i in range(qubo_mat.shape[0])]
proper_labels = [f"QUBO_{i}" for i in range(qubo_mat.shape[0])]

fig_q = px.imshow(
    qubo_mat,
    x=proper_labels,
    y=proper_labels,
    color_continuous_scale="Viridis",
    title="QUBO Matrix — Asset Bit Encoding"
)
st.plotly_chart(fig_q, use_container_width=True)

st.write("---")

# -------------------------------------------------
# Load QAOA Logs
# -------------------------------------------------
logs = pd.read_parquet("data/processed/qaoa_logs.parquet")

# 1) QAOA Iteration vs Objective
fig_obj = px.line(
    logs,
    x="iteration",
    y="objective",
    markers=True,
    title="QAOA Objective Convergence"
)
st.plotly_chart(fig_obj, use_container_width=True)

st.write("---")

# -------------------------------------------------
# Diagnostics: Runtime, Shots, Fidelity
# -------------------------------------------------
c1, c2, c3 = st.columns(3)

c1.metric("⏱ Avg Runtime (s)", f"{logs['runtime'].mean():.3f}")
c2.metric("🔫 Avg Shots", f"{logs['shots'].mean():.0f}")
c3.metric("🎯 Avg Fidelity", f"{logs['fidelity'].mean():.3f}")

# Optional deep-dive charts
fig_runtime = px.line(
    logs,
    x="iteration",
    y="runtime",
    title="Runtime per Iteration"
)
fig_shots = px.line(
    logs,
    x="iteration",
    y="shots",
    title="Shot Count per Iteration"
)
fig_fid = px.line(
    logs,
    x="iteration",
    y="fidelity",
    title="Solution Fidelity per Iteration"
)

st.plotly_chart(fig_runtime, use_container_width=True)
st.plotly_chart(fig_shots, use_container_width=True)
st.plotly_chart(fig_fid, use_container_width=True)
