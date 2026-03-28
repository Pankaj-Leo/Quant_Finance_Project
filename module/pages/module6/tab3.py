import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧠 RL Training Monitor — Reward, Loss, Exploration")

# -----------------------------------------------------------
# Using synthetic RL logs because no rl_logs.parquet exists
# -----------------------------------------------------------
st.sidebar.info("Synthetic RL training logs generated (no RL log file found).")

episodes = np.arange(1, 501)

# Reward curve (increasing trend + noise)
rewards = np.tanh(np.linspace(-3, 3, 500)) * 200 + np.random.randn(500) * 10

# Policy loss (decaying)
policy_loss = np.exp(-np.linspace(0, 6, 500)) + np.random.rand(500) * 0.02

# Value loss (decaying slower)
value_loss = np.exp(-np.linspace(0, 4, 500)) + np.random.rand(500) * 0.03

# Exploration rate (ε or entropy)
exploration = np.linspace(1.0, 0.05, 500) + np.random.randn(500) * 0.01
exploration = np.clip(exploration, 0.05, 1.0)

df = pd.DataFrame({
    "episode": episodes,
    "reward": rewards,
    "policy_loss": policy_loss,
    "value_loss": value_loss,
    "exploration": exploration
})

# -----------------------------------------------------------
# 1) Episode Reward vs Iteration
# -----------------------------------------------------------
st.subheader("📈 Episode Reward vs Training Iteration")

fig_reward = px.line(
    df, x="episode", y="reward",
    labels={"episode": "Episode", "reward": "Reward"},
    title="Training Reward Curve"
)
st.plotly_chart(fig_reward, use_container_width=True)

st.write("---")

# -----------------------------------------------------------
# 2) Policy / Value Loss
# -----------------------------------------------------------
st.subheader("📉 Policy & Value Loss")

fig_loss = go.Figure()
fig_loss.add_trace(go.Scatter(x=df["episode"], y=df["policy_loss"], name="Policy Loss"))
fig_loss.add_trace(go.Scatter(x=df["episode"], y=df["value_loss"], name="Value Loss"))

fig_loss.update_layout(
    title="Loss Curves",
    xaxis_title="Episode",
    yaxis_title="Loss"
)

st.plotly_chart(fig_loss, use_container_width=True)

st.write("---")

# -----------------------------------------------------------
# 3) Exploration Rate (ε) / Entropy Plot
# -----------------------------------------------------------
st.subheader("🌀 Exploration Rate / Entropy")

fig_exp = px.line(
    df, x="episode", y="exploration",
    labels={"episode": "Episode", "exploration": "Exploration Rate (ε)"},
    title="Exploration Decay"
)

st.plotly_chart(fig_exp, use_container_width=True)
