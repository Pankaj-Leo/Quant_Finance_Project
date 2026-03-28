import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# Load signals
df = pd.read_parquet("data/processed/pairs_signals.parquet")

pair = st.selectbox("Select Pair", df['pair'].unique())
data = df[df['pair'] == pair].copy()

# Ensure datetime format
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

# Simulate returns
data['spread_ret'] = data['spread'].diff()
data['strategy_ret'] = data['signal'].shift(1) * data['spread_ret']
data['cum_pnl'] = data['strategy_ret'].cumsum()

# Compute metrics
hit_rate = (np.sign(data['strategy_ret']) == np.sign(data['signal'])).mean()
sharpe = np.mean(data['strategy_ret']) / np.std(data['strategy_ret']) * np.sqrt(252)
drawdown = data['cum_pnl'] - data['cum_pnl'].cummax()
max_dd = drawdown.min()

# PnL curve
st.subheader(f"PnL Curve – {pair}")
fig = px.line(data, x='date', y='cum_pnl', title="Cumulative PnL", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# Drawdown
st.subheader("Drawdown Chart")
fig_dd = px.area(data, x='date', y=drawdown, title="Drawdown", template="plotly_dark")
st.plotly_chart(fig_dd, use_container_width=True)

# KPIs
st.subheader("Key Performance Indicators")
st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
st.write(f"**Hit Rate:** {hit_rate:.2%}")
st.write(f"**Max Drawdown:** {max_dd:.2f}")

# Trade duration distribution (fixed)
st.subheader("Trade Duration Distribution")
trade_changes = data.loc[data['signal'].diff().abs() == 2, 'date']
trade_durations = trade_changes.diff().dt.days.dropna()

if not trade_durations.empty:
    fig_dur = px.histogram(
        trade_durations,
        nbins=20,
        title="Trade Durations (days)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_dur, use_container_width=True)
else:
    st.info("No completed trades to show durations for.")
