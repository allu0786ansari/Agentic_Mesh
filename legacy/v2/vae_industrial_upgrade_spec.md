# Agentic Mesh — Industrial-Grade Detection Upgrade Spec

Target: move ROC-AUC from ~0.60–0.71 / PR-AUC ~0.12–0.23 to a genuinely deployable
detector (target ROC-AUC ≥ 0.90, PR-AUC well above the attack prevalence rate,
F1 ≥ 0.80 as stated in your own success criteria). Every item below has four
parts: **What** it is, **Where** it lives in the repo, **How** to implement it,
and **Why** it fixes a diagnosed problem.

---

## PART 1 — Data Preprocessing & Feature Engineering

### 1.1 Feature-separability gate (governance layer for engineered features)

**What:** Every candidate engineered feature (rolling stats, deltas, residuals,
etc.) must clear a minimum univariate separability bar before it is allowed
into the final feature set. No more "add everything and hope."

**Where:**
- New module: `scripts/feature_selection.py`
  - `def screen_features(frame: pd.DataFrame, candidate_columns: list[str], label_col="label") -> pd.DataFrame`
    — reuses the exact logic already prototyped in
    `experiment/feture_engineering.ipynb` cell `feature_separability()`, but
    promoted to production code with a proper docstring and unit test.
- Modified: `scripts/prepare_data.py::_engineer_hai_features()`
  — after building all candidate engineered columns, call `screen_features()`
  on a **held-out slice of training data only** (never on test) and drop any
  column whose `roc_auc < min_auc` (default `0.55`) and `pr_auc` below the
  attack prevalence rate.

**How:**
```python
def screen_features(frame, candidate_columns, label_col="label"):
    labels = frame[label_col].to_numpy()
    rows = []
    for col in candidate_columns:
        values = frame[col].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() == 0 or np.unique(labels[finite]).size < 2:
            continue
        auc = roc_auc_score(labels[finite], values[finite])
        auc = max(auc, 1 - auc)  # direction-agnostic
        pr = average_precision_score(labels[finite], values[finite])
        rows.append({"feature": col, "roc_auc": auc, "pr_auc": pr})
    return pd.DataFrame(rows)
```
Use this to screen candidate features on a **labeled validation split carved
out of training data's known attack windows if any exist, or on a synthetic
injected-anomaly validation set** (see 1.6) if the true training partition is
100% normal, since HAI train splits are normal-only.

**Why:** Your `_engineer_hai_features()` currently keeps every rolling-std /
local-deviation column unconditionally. This tripled input dimensionality
(38→90 features on node_01) with no filter, which is exactly why ROC-AUC
*dropped* (0.71→0.66) when you added engineering — you diluted the
reconstruction signal with noisy, non-discriminative dimensions. Gating
features by measured separability turns "more features" into "better
features."

---

### 1.2 Control-loop residual features (setpoint vs. measured-value deviation)

**What:** For every actuator command/measured-value pair in the HAI schema
(e.g. `P1_FCV01D` command vs `P1_FCV01Z` measured position — HAI's `D`/`Z`
suffix convention marks demand vs. actual for several tag families), compute
the tracking-error residual and its rate.

**Where:**
- New function in `scripts/prepare_data.py`: `_control_loop_residuals(frame, pairs)`
- Called from `_engineer_hai_features()` before the rolling-stat step, using
  a `pairs` list you build once via `_infer_setpoint_pairs(feature_columns)`
  (match `*_D` / `*_Z` / `*_R` suffix families programmatically, then confirm
  with a correlation check — see How below).
- Persist the discovered pairs list into `meta.json["feature_engineering"]["control_loop_pairs"]`
  for auditability.

**How:**
```python
def _infer_setpoint_pairs(columns: list[str]) -> list[tuple[str, str]]:
    # HAI naming: <tag>D (demand/command), <tag>Z (measured position) share a stem
    by_stem: dict[str, dict[str, str]] = defaultdict(dict)
    for col in columns:
        if col.endswith(("D", "Z")) and len(col) > 1:
            stem, suffix = col[:-1], col[-1]
            by_stem[stem][suffix] = col
    return [(pair["D"], pair["Z"]) for pair in by_stem.values() if "D" in pair and "Z" in pair]

def _control_loop_residuals(frame: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    blocks = {}
    for command_col, measured_col in pairs:
        residual = frame[command_col] - frame[measured_col]
        blocks[f"{command_col}__tracking_error"] = residual
        blocks[f"{command_col}__tracking_error_rate"] = residual.diff().fillna(0.0)
    return pd.DataFrame(blocks, index=frame.index)
```
Run `_infer_setpoint_pairs` once per node and log the discovered pairs so you
can manually verify them against the HAI documentation (some `D`/`Z` pairs
are not true setpoint/measurement pairs — confirm via a normal-period
correlation check: keep the pair only if `corr(command, measured) > 0.6`
on normal training rows).

**Why:** Per-sensor reconstruction error treats every value independently and
never checks whether the *plant is actually obeying its own commands*.
Spoofing/replay/MITM attacks on ICS systems overwhelmingly manifest as
**inconsistency between command and physical response**, not as an
out-of-range absolute value — which is exactly the failure mode a plain
per-feature VAE is blind to. This is the single highest-leverage feature
addition for this dataset family and is the documented mechanism behind why
graph/residual-aware detectors (GDN, MTAD-GAT) beat plain reconstruction
baselines on HAI-like data.

---

### 1.3 Robust scaling instead of min-max

**What:** Replace `MinMaxScalerArtifact` (or extend it) with a robust
median/IQR scaler for continuous features, keeping min-max only for features
already naturally bounded to [0,1] (e.g. valve positions already in percent).

**Where:**
- `edge/preprocessing.py`: add `RobustScalerArtifact` class, same interface
  (`fit`, `transform`, `to_dict`, `save`, `load`) as `MinMaxScalerArtifact` so
  it's a drop-in.
- `scripts/prepare_data.py::_scale_hai()`: switch continuous feature scaling
  to `RobustScalerArtifact`; keep min-max specifically for the transition
  and already-normalized binary features.
- `scripts/prepare_hai_scalers.py`: update to fit and save the new artifact
  type per node, same CLI shape.

**How:**
```python
class RobustScalerArtifact:
    def __init__(self, feature_columns, median, iqr):
        self.feature_columns = feature_columns
        self.median = median
        self.iqr = np.where(iqr > 1e-9, iqr, 1.0)

    @classmethod
    def fit(cls, parquet_path, *, normal_only=True, quantile_range=(5, 95)):
        frame = pd.read_parquet(parquet_path)
        feature_columns = [c for c in frame.columns if c not in {"time", "label"}]
        if normal_only:
            frame = frame.loc[frame["label"] == 0]
        values = frame[feature_columns].to_numpy(dtype=np.float64)
        lower, upper = np.percentile(values, quantile_range, axis=0)
        return cls(feature_columns, np.median(values, axis=0), upper - lower)

    def transform(self, values, feature_columns):
        return ((values - self.median) / self.iqr).astype(np.float32)  # NOTE: no clipping to [0,1]
```
Because output is no longer bounded to [0,1], the VAE decoder's final
activation must change from implicit linear (fine) — but if you keep
`torch.sigmoid` anywhere in a decoder head (as `TemporalVAE` currently does),
switch that head to linear output for robust-scaled inputs.

**Why:** Min-max scaling is fit once on train-normal data and is extremely
sensitive to a handful of extreme-but-legitimate values in the training
window — a single spike recalibrates the entire [0,1] range for that sensor
for all future data, including test. This directly increases the false
positive rate as your table shows (0.026 → 0.042 as engineering was added,
because more features means more chances one is min-max-fragile). Robust
scaling (median/IQR, already validated in your own notebook's
`scaler_summary()` cell) is far less sensitive to this and keeps scale
stable under drift between train and test collection periods.

---

### 1.4 Multi-scale rolling statistics (replace single 15-row window)

**What:** Instead of one fixed `rolling_window=15`, compute rolling
mean/std/local-deviation at **three scales**: fast (5 rows), medium (15
rows, current default), slow (60 rows — matches the model's full window
size, capturing drift across the whole window).

**Where:**
- `scripts/prepare_data.py::_engineer_hai_features()`: change
  `HAI_ROLLING_WINDOW = 15` to `HAI_ROLLING_WINDOWS = (5, 15, 60)` and loop
  over scales when building `rolling_std` / `local_deviation` columns,
  naming them `{col}__rolling_std_w{window}` etc.
- Feed the resulting (larger) candidate set straight into the 1.1 screening
  gate so only the scales that actually help survive per feature — some
  sensors will be discriminative at fast scale, others only at slow scale.

**How:** Straightforward loop reuse of the existing rolling logic per window
size; no new algorithm, just parameterize what's currently hardcoded.

**Why:** A single window scale assumes all anomaly signatures unfold at the
same speed. Fast attacks (command injection spikes) need a short window to
stand out against a moving average; slow attacks (gradual sensor drift,
stealthy setpoint creep) are invisible at a 15-row scale and only show up
against a much longer baseline. Multi-scale features let the
separability gate (1.1) pick whichever scale actually carries signal per
sensor, instead of you guessing one number for the whole dataset.

---

### 1.5 Frequency-domain features for oscillatory/vibration-style channels

**What:** For sensors that are periodic/oscillatory in normal operation
(flow, pressure, valve position — the `_FT`, `_PIT`, `_FCV` families),
compute band-limited FFT energy over each window instead of only
time-domain statistics.

**Where:**
- New function `scripts/prepare_data.py::_spectral_features(frame, columns, window)`
  applied per rolling window (reuse the model's `hai_window_size=60`
  window, computed at partition-prep time is not feasible row-wise — instead
  compute this **inside** `edge/data_loader.py::HAIWindowDataset.__getitem__`
  as an optional augmentation of the returned window tensor, since FFT is a
  function of a fixed-length window, not a running row-wise column).
- Config flag: `Config.enable_spectral_features: bool = False` in
  `edge/config.py`, defaulted off until validated, then flipped on.

**How:**
```python
def spectral_energy_bands(window: np.ndarray, n_bands: int = 4) -> np.ndarray:
    # window: [window_size, feature_count]
    spectrum = np.abs(np.fft.rfft(window, axis=0))
    band_edges = np.linspace(0, spectrum.shape[0], n_bands + 1, dtype=int)
    bands = [spectrum[band_edges[i]:band_edges[i+1]].mean(axis=0) for i in range(n_bands)]
    return np.concatenate(bands)  # [n_bands * feature_count]
```
Concatenate this vector to the flattened/sequence window features before
feeding the model — or, cleaner architecturally, feed it as a **separate
auxiliary encoder branch** (see Part 2, sensor-graph encoder) rather than
mixing time-domain and frequency-domain scales in one vector.

**Why:** Oscillation-frequency changes (a pump running at the wrong RPM, a
valve hunting, a controller in an unstable feedback loop) are often
invisible in time-domain mean/std but very visible as a shift in dominant
frequency or band energy. This is a standard technique in industrial
condition monitoring and directly targets attack types your current
purely-time-domain feature set structurally cannot see.

---

### 1.6 Synthetic-anomaly validation set for feature/threshold calibration

**What:** Since HAI training partitions are 100% normal (no true attacks to
validate feature separability or calibrate thresholds against), generate a
small synthetic-anomaly validation split by injecting realistic
perturbations (bias shift, freeze/replay, ramp, noise-injection) into a
held-out slice of normal training data.

**Where:**
- New module: `scripts/synthetic_anomaly_injection.py`
  - `def inject_anomalies(frame, feature_columns, seed) -> tuple[pd.DataFrame, np.ndarray]`
    returns a perturbed copy plus a synthetic label vector.
- Used by `_engineer_hai_features()` (feeds the 1.1 screening gate) and by
  the new threshold-calibration step (Part 3) as an additional, disclosed
  synthetic-only validation signal — **never mixed into real test metrics**,
  and always reported separately and labeled "synthetic" in output JSON.

**How:** Implement at minimum these four injection types on randomly chosen
column/window combinations: (a) constant bias offset, (b) freeze/replay (hold
last value for N rows), (c) linear ramp drift, (d) Gaussian noise injection
at 3–5x normal std. Label injected rows as `1`, everything else `0`.

**Why:** Right now your feature screening and thresholding have **no
labeled signal at all** during the training-data stage (real labels only
exist in test, which must stay untouched for anything but final reporting).
Without this, "feature separability" and "threshold selection" are both
being done blind or, worse, accidentally tuned against the test set by
repeated evaluation — a subtle but serious leakage risk. A synthetic
validation set gives you a principled, leakage-free proxy signal for every
calibration decision upstream of the final test evaluation.

---

## PART 2 — Advanced VAE Architecture

### Recommended architecture: **Recurrent-Flow VAE with Graph-Attention Sensor Encoding ("RFG-VAE")**

This combines three published, individually-validated techniques into one
architecture matched to this exact problem (multivariate industrial
time-series, physically-structured sensors, need for well-calibrated
anomaly scores):

1. **Graph attention sensor encoder** (from GDN — Graph Deviation Networks)
2. **Recurrent stochastic latent per timestep** (from OmniAnomaly / VRNN)
3. **Normalizing-flow posterior** (from OmniAnomaly) to fix the posterior-collapse
   / Mahalanobis-scoring failure you already observed

### 2.1 Graph-attention sensor encoder

**What:** Before any temporal modeling, learn a per-timestep **cross-sensor
attention** embedding: each sensor's representation at time *t* is a
learned weighted combination of all sensors' raw values at time *t*, with
attention weights that reflect the plant's actual sensor dependency
structure.

**Where:** New file `edge/models/graph_encoder.py`, class `SensorGraphEncoder`.

**How:**
```python
class SensorGraphEncoder(nn.Module):
    """Learns a sensor dependency graph via attention, embeds each timestep."""
    def __init__(self, feature_count: int, embed_dim: int = 32, heads: int = 4):
        super().__init__()
        self.feature_count = feature_count
        self.sensor_embeddings = nn.Parameter(torch.randn(feature_count, embed_dim))
        self.value_projection = nn.Linear(1, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, time, feature_count] -> per-timestep sensor-graph embedding
        batch, time, features = x.shape
        values = self.value_projection(x.reshape(batch * time, features, 1))
        sensor_context = self.sensor_embeddings.unsqueeze(0) + values
        attended, attn_weights = self.attention(sensor_context, sensor_context, sensor_context)
        pooled = self.output_projection(attended).mean(dim=1)  # [batch*time, embed_dim]
        self.last_attention_weights = attn_weights.detach()  # for explainability/forensics
        return pooled.reshape(batch, time, -1)
```
Expose `last_attention_weights` so the **security agent's forensic
explanation** (`agents/security/security_agent.py`) can eventually cite
*which sensor relationships* drove an anomaly score — directly useful for
your existing MITRE-technique reasoning step.

**Why:** This is the architectural fix for the "no sensor-relationship
modeling" gap identified in the diagnosis. Instead of hoping the control-loop
residual features (1.2) manually capture every important relationship, the
attention layer *learns* which sensors should predict which others under
normal operation, and an anomaly naturally shows up as attention-weighted
prediction error rather than requiring you to hand-specify every pair.

### 2.2 Recurrent stochastic latent (VRNN-style, per timestep)

**What:** Replace the single global latent vector (the current bottleneck)
with **one stochastic latent per timestep**, conditioned on a recurrent
hidden state that carries information forward through the window — so the
model can represent *when* and *how* the process deviates, not just *that*
the whole window looks off.

**Where:** New file `edge/models/sequence_vae.py`, class `RecurrentFlowVAE`
(this becomes the third architecture option alongside `VAE` and
`TemporalVAE` in `edge/models/vae.py`'s module, imported the same way).

**How (core structure):**
```python
class RecurrentFlowVAE(nn.Module):
    architecture = "recurrent_flow"

    def __init__(self, feature_count, latent_dim=16, hidden_dim=64,
                 flow_layers=4, graph_embed_dim=32):
        super().__init__()
        self.graph_encoder = SensorGraphEncoder(feature_count, graph_embed_dim)
        self.rnn = nn.GRUCell(graph_embed_dim, hidden_dim)
        # Prior network: p(z_t | h_{t-1}) instead of fixed N(0, I)
        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        # Posterior network: q(z_t | h_{t-1}, x_t)
        self.posterior_net = nn.Linear(hidden_dim + graph_embed_dim, latent_dim * 2)
        self.flow = PlanarFlowStack(latent_dim, n_layers=flow_layers)
        # Decoder: reconstructs x_t from z_t and h_{t-1}
        self.decoder_rnn = nn.GRUCell(latent_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, feature_count)

    def forward(self, x):  # x: [batch, time, feature_count]
        batch, time, _ = x.shape
        embedded = self.graph_encoder(x)  # [batch, time, graph_embed_dim]
        h = torch.zeros(batch, self.rnn.hidden_size, device=x.device)
        outputs, mus, logvars, log_det_sum = [], [], [], 0.0
        for t in range(time):
            prior_mu, prior_logvar = self.prior_net(h).chunk(2, dim=-1)
            post_mu, post_logvar = self.posterior_net(
                torch.cat([h, embedded[:, t]], dim=-1)).chunk(2, dim=-1)
            z0 = post_mu + torch.exp(0.5 * post_logvar) * torch.randn_like(post_mu)
            z, log_det = self.flow(z0)          # non-Gaussian posterior via flow
            log_det_sum = log_det_sum + log_det
            h = self.rnn(embedded[:, t], h)
            h_dec = self.decoder_rnn(z, h)
            outputs.append(self.output_head(h_dec))
            mus.append((post_mu, prior_mu)); logvars.append((post_logvar, prior_logvar))
        return torch.stack(outputs, dim=1), mus, logvars, log_det_sum
```
`PlanarFlowStack` is a small stack of invertible planar transforms (standard
normalizing-flow building block; ~30 lines, can be lifted near-verbatim from
the OmniAnomaly reference implementation pattern).

**Why per-timestep + conditional prior:** A **fixed** `N(0, I)` prior (what
standard VAEs, including your current `VAE`/`TemporalVAE`, use) forces every
latent code toward the same single mode regardless of what normal operating
regime the plant is actually in. Industrial processes have multiple
legitimate operating modes (startup, steady-state, different production
recipes) — a fixed prior either collapses all of them together (posterior
collapse, which is what you're already seeing) or fights against
representing them faithfully. A **conditional prior** `p(z_t | h_{t-1})`
lets the model represent "what's normal *given* the recent trajectory,"
which is a much sharper, more useful baseline for anomaly scoring, and this
is precisely the mechanism that lets OmniAnomaly and similar architectures
achieve materially higher separability than plain reconstruction VAEs on
this class of dataset.

### 2.3 Normalizing-flow posterior — the direct fix for your latent-scoring failure

**What:** Instead of assuming the latent posterior `q(z|x)` is Gaussian
(which your Mahalanobis-distance scoring implicitly assumed, and which
failed), transform the Gaussian sample through a small stack of invertible
flow layers to get a flexible, non-Gaussian posterior, and score anomalies
using the **flow's own log-likelihood**, not a post-hoc Mahalanobis
distance on `mu`.

**Where:** `edge/models/sequence_vae.py`, class `PlanarFlowStack`; scoring
logic in a new `evaluation/latent_likelihood.py` replacing/augmenting
`evaluation.inference.latent_mahalanobis_scores`.

**How (scoring):**
```python
def flow_anomaly_score(model, window) -> float:
    _, mus, logvars, log_det = model(window)
    post_mu, prior_mu = mus[-1]
    post_logvar, prior_logvar = logvars[-1]
    # KL(q(z_t)||p(z_t|h_{t-1})) computed analytically pre-flow, adjusted by log_det post-flow
    kl = 0.5 * (prior_logvar - post_logvar +
                (post_logvar.exp() + (post_mu - prior_mu).pow(2)) / prior_logvar.exp() - 1).sum(-1)
    return (kl - log_det).mean().item()  # higher = more anomalous
```
This score is used **directly**, not blended 50/50 with reconstruction error
via a fixed `alpha` — see Part 3 for principled combination.

**Why:** Your diagnosis already proved that combining a naive Mahalanobis
latent score with reconstruction error made results *worse*
(ROC-AUC 0.668→0.603). That's not a tuning failure, it's a modeling
mismatch: Mahalanobis distance is only meaningful under a Gaussian
assumption, and posterior-collapsed or non-Gaussian latents violate that
assumption badly. The flow-based KL/log-likelihood score is the
theoretically correct anomaly score under a properly regularized VAE, and it
naturally stays sensible even when the true posterior isn't Gaussian.

### 2.4 Loss function — sequence ELBO with free-bits

**What:** Full training loss combining per-timestep reconstruction
likelihood, per-timestep KL against the conditional prior, and a
**free-bits floor** to hard-prevent posterior collapse.

**Where:** `edge/training/sequence_trainer.py` (new; parallel to existing
`trainer.py` and `dp_trainer.py`, following the same `TrainingResult`
dataclass contract so it's a drop-in for `edge/main.py`).

**How:**
```python
def sequence_vae_loss(reconstruction, targets, mus, logvars, log_det,
                       free_bits: float = 0.5, beta: float = 0.3):
    recon_loss = F.mse_loss(reconstruction, targets, reduction="mean")
    kl_terms = []
    for (post_mu, prior_mu), (post_logvar, prior_logvar) in zip(mus, logvars):
        kl = 0.5 * (prior_logvar - post_logvar +
                     (post_logvar.exp() + (post_mu - prior_mu).pow(2)) / prior_logvar.exp() - 1)
        kl_terms.append(torch.clamp(kl, min=free_bits))  # free-bits floor per dimension
    kl_loss = torch.stack(kl_terms).mean() - log_det.mean() / targets.shape[1]
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
```
`free_bits=0.5` guarantees each latent dimension carries at least that much
information regardless of what the optimizer would otherwise prefer —
this is the direct, mechanical fix for posterior collapse, stronger than
just lowering `beta`.

**Why:** Simply lowering `beta` (a softer fix mentioned in the earlier
diagnosis) reduces the *pressure* toward collapse but doesn't prevent it;
free-bits makes collapse **impossible below the floor** by construction,
which is the standard, proven technique (used in VQ-VAE-2, OmniAnomaly, and
most modern sequence VAEs) for exactly this failure mode.

### 2.5 Checkpoint/config integration

**Where:**
- `edge/models/vae.py` (or new `sequence_vae.py`, imported alongside):
  add `architecture = "recurrent_flow"` tag, matching the existing pattern
  used by `TemporalVAE`.
- `edge/training/sequence_trainer.py`: mirror `trainer.py`'s checkpoint
  dict, adding `graph_embed_dim`, `flow_layers`, `free_bits` fields.
- `evaluation/inference.py::load_checkpoint_model()`: add a third
  `elif architecture == "recurrent_flow":` branch reconstructing
  `RecurrentFlowVAE` from checkpoint metadata — same pattern already used
  for `TemporalVAE`, so this is a small, low-risk addition to existing code.
- `edge/config.py`: add `graph_embed_dim`, `flow_layers`, `free_bits`,
  `sequence_latent_dim` fields with sensible defaults, following the
  existing `Config` dataclass + `from_env()` pattern exactly.

**Why isolate as a new file rather than editing `TemporalVAE` in place:**
Keeps the three architectures (`VAE`, `TemporalVAE`, `RecurrentFlowVAE`)
independently selectable and comparable via `scripts/comparative_study.py`'s
existing `--architecture` flag (extend its `choices` list), preserving your
ability to run controlled ablations and prove each change's individual
contribution rather than only reporting one final number.

---

## PART 3 — Thresholding & Scoring (ties preprocessing + architecture together)

### 3.1 Extreme Value Theory (POT) thresholding

**What:** Replace the fixed `quantile(0.95)` threshold with a Peak-Over-
Threshold extreme value fit on validation scores, giving a threshold
calibrated to the actual tail behavior of each node's score distribution
instead of an arbitrary fixed percentile.

**Where:** New module `evaluation/threshold.py`,
`def pot_threshold(scores: np.ndarray, risk: float = 1e-3) -> float`
(standard Generalized Pareto Distribution fit over an initial percentile
threshold, using `scipy.stats.genpareto`). Called from
`evaluation/inference.py::evaluate_node()` in place of the current
`np.quantile(validation_errors, validation_quantile)` line, with the old
quantile kept as a fallback/comparison value in the output JSON.

**Why:** A fixed 95th percentile threshold implicitly assumes every node's
score distribution has the same tail shape and that 5% is the "right" false
positive budget everywhere — neither is true (your `node_03` compact
partition has 7 base features vs. `node_01`'s 38; their noise floors are
not comparable). POT thresholds adapt to each node's actual tail behavior
and is the standard approach in the anomaly-detection literature (used by
SPOT/DSPOT and cited directly in the OmniAnomaly paper) for exactly this
problem.

### 3.2 Principled score combination (quality-weighted, not fixed alpha)

**What:** When combining reconstruction score and flow-based latent score,
weight each by its own standalone validation ROC-AUC rather than a fixed
`alpha=0.5`.

**Where:** `evaluation/inference.py::combined_anomaly_scores()` — modify
signature to accept `reconstruction_auc` and `latent_auc` (computed on the
synthetic validation set from 1.6), and set
`alpha = latent_auc / (reconstruction_auc + latent_auc)` instead of a
constant.

**Why:** Directly prevents the regression you already observed — a weak or
broken signal (previously, an under-regularized latent score) can no longer
drag down a working signal, because it will earn near-zero combination
weight if its own validation AUROC is near 0.5 (random).

---

## File-Level Change Map (summary)

| File | Change |
|---|---|
| `scripts/feature_selection.py` | **new** — separability screening (1.1) |
| `scripts/synthetic_anomaly_injection.py` | **new** — injected-anomaly validation set (1.6) |
| `scripts/prepare_data.py` | add residual pairs (1.2), multi-scale windows (1.4), wire in screening gate |
| `edge/preprocessing.py` | add `RobustScalerArtifact` (1.3) |
| `edge/config.py` | add spectral/graph/flow config fields |
| `edge/data_loader.py` | optional spectral feature branch (1.5) |
| `edge/models/graph_encoder.py` | **new** — `SensorGraphEncoder` (2.1) |
| `edge/models/sequence_vae.py` | **new** — `RecurrentFlowVAE`, `PlanarFlowStack` (2.2–2.4) |
| `edge/training/sequence_trainer.py` | **new** — sequence ELBO training loop |
| `evaluation/inference.py` | add `recurrent_flow` load branch, quality-weighted combination (3.2) |
| `evaluation/threshold.py` | **new** — POT thresholding (3.1) |
| `evaluation/latent_likelihood.py` | **new** — flow-based scoring, replaces raw Mahalanobis |
| `scripts/comparative_study.py` | extend `--architecture` choices to include `recurrent_flow` |

---

## Suggested implementation order

1. **1.1 + 1.6** (screening gate + synthetic validation) — unlocks every
   other decision being made on evidence instead of guesswork.
2. **1.2 + 1.3** (residual features + robust scaling) — cheap, high-leverage
   preprocessing fixes, testable with your *existing* `VAE`/`TemporalVAE`
   before touching architecture, so you get a clean before/after signal.
3. **2.1–2.4** (RFG-VAE) — the architecture upgrade, benchmarked against
   the improved-preprocessing baseline from step 2.
4. **3.1 + 3.2** (POT thresholding, quality-weighted combination) — final
   calibration layer once the underlying scores are trustworthy.
5. **1.4 + 1.5** (multi-scale, spectral) — incremental refinements layered
   on top once the core architecture change is validated, since they add
   the most implementation complexity for comparatively smaller expected
   gains.

Each step should be run through `scripts/comparative_study.py` as its own
ablation with a fixed seed set, so the final report can show, line by line,
which change contributed how much — the same rigor your project's existing
evidence-gated evaluation model (`evaluation/check_success_criteria.py`)
already expects.
