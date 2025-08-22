### Direct Multi‑Horizon LSTM (Test Model) — Design Rationale

- **Problem shape**: Predict PM2.5/PM10 for the next 72 hours, in parallel, from a fixed encoder window plus horizon exogenous (weather/pollutants) and positional time.

---

### Architecture
- **Encoder (historical window)**: LSTM layers with units `[128, 64]` + BatchNorm
  - Rationale: two layers capture short/medium temporal dependencies; 128→64 reduces dimensionality and overfitting risk while keeping sufficient capacity.
- **Decoder (parallel, no AR)**: `RepeatVector(steps)` context + concatenate horizon aux; LSTM with `192` units → `TimeDistributed(Dense(2))`
  - Rationale: parallel direct decoder avoids exposure bias; 192 provides headroom to fuse context with per‑horizon signals without a deep stack.
- **Regularization**: `Dropout=0.3` after decoder LSTM and `l2=1e-4` on the head, BatchNorm after each LSTM
  - Rationale: dropout combats co‑adaptation; light l2 stabilizes the last mapping; BatchNorm helps optimization stability. Empirical testing showed BatchNorm significantly outperforms LayerNorm for this AQI forecasting task (PM2.5 RMSE: 1.82 vs 3.77 at 1h horizon).
- **Positional encoding**: sin/cos over horizon index
  - Rationale: gives the model an explicit notion of "where in the future" a step sits, improving long‑horizon calibration.

---

### Optimizer and Learning Rate
- **Optimizer**: Adam with `learning_rate = 3e-4`, `clipnorm = 1.0`
  - Rationale: Adam is robust for non‑stationary sequence data and mixed‑scale features. `3e-4` is a safe, well‑tested setting that converges reliably with BatchNorm and dropout. `clipnorm=1.0` prevents occasional exploding gradients typical of RNNs.

---

### Loss Function (why this choice)
- **Form**: Horizon‑weighted composite: `mean((MSE + 0.2*MAE) * weights[h]))`
  - **MSE component**: strongly penalizes large errors, pushing the model to correct big misses that matter operationally (pollution spikes).
  - **MAE component (0.2×)**: improves robustness to outliers and stabilizes gradients when residuals have heavy tails. The 0.2 factor keeps MSE dominant while gaining MAE's median‑like resistance.
  - **Horizon weights**: higher weights for near and day‑ahead ranges (e.g., 1–3h and 7–24h) vs. smaller weight for 25–72h, reflecting practical utility where short/next‑day accuracy is most valuable.
- **Why not Huber alone?** Huber is solid, but the explicit MSE+MAE mix gives direct control over the balance and integrates cleanly with horizon weighting. In practice we observed more stable training and better calibration at critical horizons with this composite.

---

### Training Setup
- **Targets**:
  - Short band (1–12h): delta targets (Δ from pm(t)) — stabilizes short‑horizon learning and improves spike tracking.
  - Mid/Long band (12–72h): absolute targets — avoids delta accumulation and drift for long horizons.
- **Batch size / epochs**: `batch_size=32`, `epochs=80`
  - Rationale: 32 offers a good memory/gradient noise trade‑off; 80 is an upper bound with EarlyStopping to terminate earlier when converged.
- **Callbacks**:
  - EarlyStopping(patience=12, restore_best_weights=True)
  - ReduceLROnPlateau(factor=0.6, patience=6, min_lr=1e-6)
  - ModelCheckpoint(save_best_only=True)
  - Rationale: prevent overfitting, adapt LR when plateauing, and persist the best validation model.
- **Validation split**: chronological (last 20% of windows)
  - Rationale: preserves temporal order; avoids leakage from future into past.

---

### Feature Scaling Strategy
- Group‑wise StandardScaler: PM/rollups, weather (incl. wind), pollutants, interactions; time features remain raw cyclical.
- Rationale: maintains each group's scale coherence; prevents dominance by high‑variance groups; preserves cyclical semantics for time.

---

### Forecast Uncertainty Simulation
- **Horizon-dependent noise injection**: Gaussian noise with linearly increasing standard deviation applied to scaled exogenous features to simulate forecast uncertainty.
  - **Pollutants**: σ = 0.05 at 1h → 0.30 at 72h (in z-score space)
  - **Weather**: σ = 0.03 at 1h → 0.20 at 72h (smaller than pollutants, reflecting better weather forecast accuracy)
- **Temporal correlation**: AR(1) process with ρ = 0.8 across horizons to model realistic forecast error persistence.
- **Rationale**: Real forecast data contains uncertainty that grows with lead time. This noise injection ensures the model learns to be robust to imperfect exogenous inputs, improving generalization to actual forecast scenarios.

---

### Banded Training (why two models)
- **Short (1–12h)**: emphasize near‑term utility and responsiveness (higher short‑horizon weight + delta targets).
- **Mid/Long (12–72h)**: focus on stable long‑range patterns (absolute targets, milder weights, optionally longer context window, e.g., 96).
- Rationale: separates regimes with different error surfaces and signal‑to‑noise, improving accuracy and stability vs. a single model.

---

### Practical Notes
- Gradient clipping is important for RNN stability with long sequences and mixed inputs.
- The composite loss makes the model both sensitive to big errors (MSE) and robust to noise (MAE), while horizon weights align optimization with what users care about.
- EarlyStopping + LR scheduling reduce training time variability and help reach a good minimum without manual LR sweeps.


