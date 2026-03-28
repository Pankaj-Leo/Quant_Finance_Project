# AlphaForge — Full-Stack Quantitative Research Pipeline

AlphaForge is a modular, end-to-end quantitative finance research system built in Python. It assembles a complete research stack: config-driven OHLCV ingestion, classical equity factor engineering, a Graph Attention Network (GATv2Conv) autoencoder for alpha signal generation, cointegration-based statistical arbitrage, mean-variance / risk parity / QUBO-formulated portfolio construction, GARCH-family volatility modeling, TimeGAN-powered limit-order-book simulation, and Black-Scholes / Monte Carlo derivatives tooling — all wired into interactive Streamlit dashboards.

> **Honesty note:** This is a research and education codebase. The QAOA portfolio uses a classical MV+RP blend as a solver hookpoint — a live quantum solver is not wired in. RL execution logs are simulated. GAN volatility comparisons are illustrative. These are labeled throughout. If you are evaluating this for hiring purposes, that transparency is deliberate.

> **Module versions:** The repository contains progressive snapshots (`module2` through `module7`). Use **`module 7/`** as the canonical version. Earlier modules track the build progression.

---
![](info.png)

## What this project demonstrates

| Competency | Where it appears |
|---|---|
| Market microstructure | TimeGAN-LOB, LOB simulator, spread/OFI/depth modeling |
| Statistical arbitrage | ADF cointegration, z-score pairs signals, walk-forward evaluation |
| Factor modeling | Fama-French-style factor library + GATv2 latent factor extraction |
| Volatility modeling | GARCH/EGARCH, implied vol surfaces, regime clustering, GAN diagnostics |
| Portfolio optimization | Mean-variance (CVXPY/pseudo-inverse), risk parity, QUBO formulation |
| Quantum-inspired optimization | QUBO matrix encoding; drop-in hookpoint for QAOA / D-Wave solvers |
| Market simulation | Gym-style LOB environment, TWAP/VWAP/RL agent comparison |
| Derivatives pricing | Black-Scholes, local vol calibration, PDE solver, Greeks surfaces |
| MLOps discipline | Hydra configs, MLflow tracking, Parquet artifact chain, reproducible seeds |

---

## Architecture

```
yfinance → ohlcv.parquet
                │
                ▼
        factor_gen.py
                │
        factor_library.parquet
        (momentum_12m, momentum_3m, volatility_60d, size)
                │
                ▼
        Alpha_module.ipynb          ← GATv2Conv autoencoder
        graph_adj.npy ──────────────┘
                │
        alpha_predictions.parquet
                │
        ┌───────┴────────┐
        ▼                ▼
  backtest.py     mean_reversion_signals.py
  IC + L/S port   ADF cointegration + z-score
        │                │
        └───────┬─────────┘
                ▼
  build_portfolio_returns.py
  MV │ Risk Parity │ QAOA-blend
                │
                ▼
        generate_qubo.py
        Q = λ·Σ − diag(α) + γ·A
                │
        qubo_matrix.parquet
        (quantum solver hookpoint)
                │
                ▼
          dashboard.py  (Streamlit)
```

Parallel modules (not in the main chain, visualized separately):

```
simulate_lob.py  →  timegean_training.py  →  LOB dashboard tabs
vol_model/       →  GARCH / surface / regime tabs
pricing/         →  BSM / Greeks / MC tabs
```

---

## Pipeline stages

### Stage 1 — Data ingestion (`src/data_ingest.py`)

- **Config:** Hydra-driven. All tickers, date ranges, and paths live in `src/conf/config.yaml`. No hardcoded values.
- **Download:** yfinance for AAPL, MSFT, GOOG, AMZN — 2018-2025 by default (~1,757 trading days).
- **Output:** `data/raw/ohlcv.parquet`
- **Tracking:** MLflow logs params + artifact path for every run.

### Stage 2 — Factor engineering (`src/factor_model/factor_gen.py`)

Reshapes wide multi-index OHLCV to tidy long format. Computes four classical factors:

| Factor | Formula |
|---|---|
| `momentum_12m` | `Close / Close.shift(252) − 1` |
| `momentum_3m` | `Close / Close.shift(63) − 1` |
| `volatility_60d` | Rolling 60-day std of daily log returns |
| `size` | `log(Close × Volume)` |

Output: `data/processed/factor_library.parquet` (~28,000 rows × 12 columns)

### Stage 3 — Graph alpha model (`Alpha_module.ipynb`)

Architecture: **GATv2Conv autoencoder** (PyTorch + torch-geometric)

```
Input: padded return sequences × adjacency matrix
  └─ GATv2Conv(in → 32 hidden, heads=4)  → ELU
  └─ GATv2Conv(128 → 16-dim latent, heads=1)
  └─ Linear decoder → reconstruct return sequences
```

- Loss: MSE reconstruction; 200 epochs; Adam lr=0.005
- Final reconstruction loss ≈ 0.055
- Alpha signal: **first column of the 16-dim latent embedding `z`**
- Why GATv2 over GAT? Standard GAT computes attention from each node's own features — static and symmetric. GATv2 applies the scoring function to the joint query-key representation, enabling asymmetric attention. In financial terms: MSFT's dynamics can be informative for AAPL's embedding differently than the reverse.
- Output: `data/processed/alpha_predictions.parquet`

### Stage 4 — Statistical arbitrage (`mean_reversion_signals.py`)

- Tests all 6 pair combinations across 4 tickers
- For each pair: OLS regression → ADF test on residual spread
- Accept pair if ADF p-value < 0.05 (stationary spread → cointegrated)
- Z-score: `z = (spread − μ) / σ`; signals at ±2 standard deviations
- Output: `data/processed/pairs_signals.parquet`

### Stage 5 — Backtesting (`backtest.py`)

- Merges factor library + alpha predictions on ticker
- Daily cross-sectional **Spearman IC**: rank-correlation of predicted alpha vs realized next-day return
- Long-short portfolio: long top-30% by alpha, short bottom-30%, equal weighted within legs
- Outputs: `ic_results.parquet`, `cumulative_returns.png`

Reference benchmark: IC > 0.05 is considered meaningful; IC > 0.10 is strong in institutional equity research.

### Stage 6 — Portfolio construction (`build_portfolio_returns.py`)

Three strategies compared side-by-side:

| Strategy | Method |
|---|---|
| Mean-Variance | `w = Σ⁻¹ · μ` via Moore-Penrose pseudo-inverse |
| Risk Parity | `w = 1/n` equal weight |
| QAOA-inspired | `(w_MV + w_RP) / 2`, normalized |

The QAOA-inspired blend is a classical approximation. The real value is the QUBO matrix it feeds.

Metrics tracked per strategy: daily returns, 20-day rolling vol, rolling Sharpe, max drawdown, turnover.

### Stage 7 — QUBO formulation (`generate_qubo.py`)

```
Q = λ_risk · Σ  −  diag(α)  +  λ_graph · A
```

Where:
- `Σ` = historical covariance matrix (risk penalty, λ_risk = 0.2)
- `diag(α)` = GNN alpha vector on diagonal (return incentive)
- `A` = graph adjacency matrix (diversification penalty, λ_graph = 0.1)

This is a complete QUBO encoding. Any classical simulated annealer, D-Wave quantum annealer, or QAOA circuit can take this matrix directly. The current pipeline uses the classical blend as a placeholder.

Output: `data/processed/qubo_matrix.parquet`

### Stage 8 — LOB microstructure simulation

**`src/simulate_lob.py`:** Synthetic limit-order book with Brownian-motion midprice, Gaussian spread, and random order flow imbalance (OFI). Produces baseline LOB sequences.

**`timegean_training.py`:** TimeGAN (via ydata-synthetic) trained on real LOB sequences. Learns the multivariate temporal structure — spread regimes, OFI persistence, depth clustering — and generates synthetic paths that preserve statistical properties.

Dashboard: depth heatmap, midprice replay, spread dynamics, real vs synthetic comparison.

### Stage 9 — Volatility modeling

- **GARCH / EGARCH** via `arch` library: realized vol forecasting and residual diagnostics
- **Implied vol surface:** 3D strike × maturity × IV surfaces via Plotly
- **K-means clustering:** vol regime detection (high / transition / low)
- **GAN diagnostics:** ACF, fat-tail, and clustering comparison of real vs synthetic vol paths
- **SABR / Heston (extended):** stochastic vol calibration for derivatives pricing context

### Stage 10 — Derivatives pricing

- **Black-Scholes:** closed-form pricing and implied vol inversion
- **Greeks surfaces:** delta, gamma, vega over (strike, maturity) grids
- **Monte Carlo:** payoff distribution histograms, variance reduction (antithetic, control variates)
- **Local vol calibration:** Dupire formula applied to the implied vol grid

---

## Streamlit dashboards

Launch from `module 7/` with `streamlit run dashboard.py`. Populate data artifacts first.

### Root pages (`pages/`)

| File | Content |
|---|---|
| `tab1_volatility_overview.py` | Rolling σ bands, quick stats |
| `tab2_model_diagnostics.py` | GARCH vs EGARCH vs ML vol forecast comparison |
| `tab3_surface_regimes.py` | 3D implied vol surface + K-means regime heatmap |
| `tab4_gan_vol_test.py` | Real vs synthetic vol: ACF, fat tails, clustering |
| `tab4_model_analytics.py` | Signal and alpha correlations |

### Dashboard modules (`dashboard/`)

| Module | Content |
|---|---|
| `module4/` | Factor library, GNN embeddings (UMAP), PCA spectrum, rolling Sharpe |
| `module5/` | Efficient frontier, MV/RP/QAOA weight bars, QUBO heatmap |
| `module6/` | TWAP/VWAP/RL PnL, inventory trajectories, stress/latency heatmaps |
| `module7/` | LOB replay (real vs TimeGAN), Greeks surfaces, MC payoffs, pricing sandbox |

---

## Quickstart

```bash
cd "module 7"
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Ingest OHLCV data
python src/data_ingest.py

# 2. Engineer factor library
python src/factor_model/factor_gen.py

# 3. Train GNN alpha model
#    Open and run Alpha_module.ipynb (requires factor_library.parquet + graph_adj.npy)

# 4. Generate pairs trading signals
python mean_reversion_signals.py

# 5. Run long-short backtest
python backtest.py

# 6. Build portfolios and QUBO matrix
python build_portfolio_returns.py
python generate_qubo.py
python qaoa_logs.py

# 7. Simulate LOB sequences
python src/simulate_lob.py

# 8. Launch interactive dashboard
streamlit run dashboard.py
```

All paths assume the working directory is `module 7/`.

---

## Data artifacts

| Artifact | Produced by | Description |
|---|---|---|
| `data/raw/ohlcv.parquet` | `data_ingest.py` | Raw OHLCV for 4 tickers, 7 years |
| `data/raw/ticker_sector.csv` | `ticker_sector.py` | Sector labels for universe |
| `data/processed/factor_library.parquet` | `factor_gen.py` | 4 classical factor scores, tidy long format |
| `data/processed/alpha_predictions.parquet` | `Alpha_module.ipynb` | GATv2 latent dim-1 alpha per ticker per date |
| `data/processed/graph_adj.npy` | Pre-computed / `Alpha_module.ipynb` | 4×4 adjacency matrix |
| `data/processed/node_embeddings.npy` | `Alpha_module.ipynb` | Full 16-dim latent embeddings |
| `data/processed/merged.parquet` | `backtest.py` | Factors + alpha joined on ticker |
| `data/processed/ic_results.parquet` | `backtest.py` | Daily Spearman IC series |
| `data/processed/pairs_signals.parquet` | `mean_reversion_signals.py` | Z-score signals for all pairs |
| `data/processed/portfolio_returns.parquet` | `build_portfolio_returns.py` | Daily returns for MV, RP, QAOA |
| `data/processed/qubo_matrix.parquet` | `generate_qubo.py` | 4×4 QUBO matrix Q |
| `data/processed/qaoa_logs.parquet` | `qaoa_logs.py` | Placeholder solver log |
| `data/raw/lob_synthetic.parquet` | `simulate_lob.py` | Synthetic LOB sequences |

---

## Tech stack

| Layer | Libraries |
|---|---|
| Data & storage | `pandas`, `numpy`, `yfinance`, `pyarrow` |
| Config & tracking | `hydra-core`, `omegaconf`, `mlflow` |
| Stats & finance | `statsmodels`, `scipy`, `arch` |
| Machine learning | `tensorflow`, `keras`, `torch`, `torch-geometric`, `ydata-synthetic` |
| Visualization | `streamlit`, `plotly`, `matplotlib`, `seaborn` |
| Optimization | QUBO matrix construction; QAOA-ready (solver-agnostic) |

See `module 7/requirements.txt` for pinned versions.

---

## What is real vs. placeholder

Being explicit about this matters for any reader doing due diligence:

| Component | Status |
|---|---|
| Data ingestion, factor engineering | Real — runs against live yfinance data |
| GATv2Conv autoencoder training | Real — PyTorch training loop, real loss convergence |
| ADF cointegration tests | Real — statsmodels on actual price series |
| Spearman IC backtest | Real — on actual alpha predictions vs realized returns |
| Mean-variance & risk parity construction | Real — CVXPY / pseudo-inverse |
| QUBO matrix formulation | Real — economically meaningful encoding, solver-ready |
| QAOA optimization | Placeholder — classical blend used; quantum solver not wired |
| LOB simulation (Brownian) | Real synthetic — mathematically generated |
| TimeGAN LOB generation | Real — ydata-synthetic training on LOB sequences |
| RL execution agent | Placeholder — architecture present, not fully trained |
| Derivatives Greeks surfaces | Real — BSM analytical formulas |
| Monte Carlo pricing | Real — numerical simulation with variance reduction |

---

## Further reading

- `BLOGPOST.md` — narrative on what, how, and why AlphaForge matters
- `NOTEBOOKLM_INFOGRAPHIC_SCRIPT.md` — panel-by-panel source for infographic generation
- `VIDEO_SCRIPT.md` — spoken explainer script for video production
- `Quant_Project_Documentation.docx` — full module specifications and hiring signals

---

## Disclaimer

This is a **research and education** codebase. It is **not** financial advice and is **not** production-ready without rigorous out-of-sample validation on a larger universe, live data feeds, compliance review, and proper risk governance. Use it to learn, extend, and build from — not to trade.

---
## Author

**Pankaj Somkuwar** - AI Engineer / AI Product Manager / AI Solutions Architect

- LinkedIn: [Pankaj Somkuwar](https://www.linkedin.com/in/pankaj-somkuwar/)
- GitHub: [@Pankaj-Leo](https://github.com/Pankaj-Leo)
- Website: [Pankaj Somkuwar](https://www.pankajsomkuwarai.com)
- Email: [pankaj.som1610@gmail.com](mailto:pankaj.som1610@gmail.com)
