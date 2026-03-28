# pages/tab4_model_analytics.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import roc_curve, auc

st.title("Tab 4 – Model Analytics 📊")

# --- Load existing data ---
signals = pd.read_parquet("data/processed/pairs_signals.parquet")
alpha = pd.read_parquet("data/processed/alpha_predictions.parquet")

# --- Merge by pair & date if possible ---
if {"pair", "date"}.issubset(signals.columns) and {"pair", "date"}.issubset(alpha.columns):
    data = pd.merge(signals, alpha, on=["pair", "date"], how="inner")
else:
    data = signals.copy()

# --- 1. Feature/Signal Correlation Heatmap ---
st.subheader("Feature Correlation Heatmap")
numeric_cols = data.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    corr = data[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis",
                         title="Correlation Between Features/Signals")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Not enough numeric columns to show correlations.")

# --- 2. ROC Curve from Alpha Predictions ---
# st.subheader("ROC Curve for Mean Reversion Probability")
# if {"true_label", "pred_prob"}.issubset(alpha.columns):
#     fpr, tpr, _ = roc_curve(alpha["true_label"], alpha["pred_prob"])
#     roc_auc = auc(fpr, tpr)
#     fig_roc = px.area(
#         x=fpr, y=tpr,
#         title=f"ROC Curve (AUC = {roc_auc:.3f})",
#         labels={"x": "False Positive Rate", "y": "True Positive Rate"},
#         width=700, height=400
#     )
#     st.plotly_chart(fig_roc, use_container_width=True)
# else:
    # st.warning("Missing columns 'true_label' or 'pred_prob' in alpha_predictions.parquet.")

# --- 3. Pair Summary Table ---
# st.subheader("Pair Performance Summary (Simulated AUC Metrics)")
# summary_data = []
# if {"pair", "pred_prob", "true_label"}.issubset(alpha.columns):
#     for pair, grp in alpha.groupby("pair"):
#         try:
#             fpr, tpr, _ = roc_curve(grp["true_label"], grp["pred_prob"])
#             auc_val = auc(fpr, tpr)
#             summary_data.append({"pair": pair, "AUC": auc_val})
#         except Exception:
#             continue
#     summary = pd.DataFrame(summary_data)
#     st.dataframe(summary.round(3), use_container_width=True)
# else:
#     st.info("Could not compute summary; required prediction columns missing.")

st.success("✅ Model Analytics Dashboard Loaded Successfully")
