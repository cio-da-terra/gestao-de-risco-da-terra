"""CAT-lite: a scaled-down Climate-Adaptive Transformer for conilon yield.

Replicates the architecture of Namdeo & Baraskar (2026), "Climate-Adaptive
Transformer for Crop Yield Prediction with Conformal Uncertainty
Quantification" (IJERT 15(04)), sized for our data:

| Paper (7,182 rows)                         | Here (~430 rows)                         |
|--------------------------------------------|------------------------------------------|
| 12 monthly steps × 4 ERA5 vars             | 12 months Mar(Y-1)–Feb(Y) × tmax, tmin, rain, RH (NASA POWER) |
| PAPE: sinusoidal + learned phenological    | same                                     |
|   embedding + seasonal harmonics           |                                          |
| Learnable phenology attention mask,        | same, initialised from conilon stages    |
|   initialised from crop physiology         |                                          |
| 4 layers, d=128, 8 heads                   | 1 layer, d=16, 2 heads                   |
| Cross-modal fusion: climate → vegetation   | climate → context tokens (ENSO, biennial |
|   → static, with sigmoid gating            |   lag, LSPA anomalies), sigmoid gating   |
| Huber loss, AdamW, OneCycleLR, early stop  | same; 3-seed ensemble                    |
| Split conformal + ACI                      | our shared online conformal + ACI        |

The target is the yield anomaly after the municipality level and common
trend, like the other PoC models: a Transformer cannot extrapolate the trend.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import nn

# PyTorch's fused eval-mode attention kernel returns NaN with a learned float
# attention mask (seen with torch 2.15 CPU); the standard path is correct.
torch.backends.mha.set_fastpath_enabled(False)

CLIMATE_VARS = ["tmax", "tmin", "rain", "rh"]
# Months of the sequence: March of Y-1 .. February of Y (forecast is end of Feb)
SEQ_MONTHS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
SEQ_COLS = [f"m{k:02d}_{v}" for k in range(12) for v in CLIMATE_VARS]
CONTEXT = ["oni", "lag2_anom", "lspa_prev", "lspa_cur"]

# Conilon phenology prior for the attention mask (relative importance per month):
# Mar–Jun branch growth (next crop's bearing wood), Jul–Oct flowering and fruit
# set (most water-sensitive), Nov–Feb grain filling.
PHENO_PRIOR = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8])
STAGES = ["growth"] * 4 + ["flowering"] * 4 + ["filling"] * 4


def add_monthly_sequences(feats: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """Add 48 columns: monthly climate for the 12 months before the forecast."""
    w = wx.assign(yr=wx.date.dt.year, m=wx.date.dt.month)
    rh = "RH2M" if "RH2M" in w.columns else None
    agg = {"tmax": ("tmax", "mean"), "tmin": ("tmin", "mean"), "rain": ("rain", "sum")}
    if rh:
        agg["rh"] = (rh, "mean")
    mon = w.groupby(["code", "yr", "m"]).agg(**agg)
    if not rh:
        mon["rh"] = 0.0
    out = feats.copy()
    for k, m in enumerate(SEQ_MONTHS):
        src_year = out.year - 1 if m >= 3 else out.year
        idx = pd.MultiIndex.from_arrays([out.code, src_year, np.full(len(out), m)])
        vals = mon.reindex(idx)
        for v in CLIMATE_VARS:
            out[f"m{k:02d}_{v}"] = vals[v].to_numpy()
    return out.dropna(subset=SEQ_COLS).reset_index(drop=True)


class PAPE(nn.Module):
    """Phenology-Aware Positional Encoding: sinusoidal + learned + harmonic."""

    def __init__(self, d: int, n: int = 12):
        super().__init__()
        pos = torch.arange(n).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(n, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("sinusoidal", pe)
        self.learned = nn.Parameter(torch.zeros(n, d))
        m = torch.tensor(SEQ_MONTHS, dtype=torch.float32)
        harm = torch.stack([torch.sin(2 * math.pi * m / 12), torch.cos(2 * math.pi * m / 12),
                            torch.sin(4 * math.pi * m / 12), torch.cos(4 * math.pi * m / 12)], 1)
        self.register_buffer("harmonics", harm)
        self.harm_proj = nn.Linear(4, d)

    def forward(self, x):
        return x + self.sinusoidal + self.learned + self.harm_proj(self.harmonics)


class CATLite(nn.Module):
    def __init__(self, n_vars: int, n_ctx: int, d: int = 16, heads: int = 2, layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_vars, d)
        self.pape = PAPE(d)
        # Learnable phenology mask: additive bias on attention to each month
        self.pheno_bias = nn.Parameter(torch.tensor(np.log(PHENO_PRIOR), dtype=torch.float32))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, activation="gelu",
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.pool = nn.Linear(d, 1)
        # Context tokens: one token per context feature (value × embedding + id)
        self.ctx_val = nn.Parameter(torch.randn(n_ctx, d) * 0.1)
        self.ctx_id = nn.Parameter(torch.randn(n_ctx, d) * 0.1)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(2 * d, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.last_pool_weights = None

    def forward(self, seq, ctx):
        h = self.pape(self.inp(seq))                              # (B, 12, d)
        mask = self.pheno_bias[None, :].expand(12, 12)            # bias on keys
        h = self.encoder(h, mask=mask)
        w = torch.softmax(self.pool(h).squeeze(-1) + self.pheno_bias, dim=1)   # (B, 12)
        self.last_pool_weights = w.detach()
        c = (w[..., None] * h).sum(1, keepdim=True)               # (B, 1, d)
        tokens = ctx[..., None] * self.ctx_val + self.ctx_id      # (B, n_ctx, d)
        c2, _ = self.cross(c, tokens, tokens)
        g = torch.sigmoid(self.gate(torch.cat([c, c2], -1)))
        fused = (g * c + (1 - g) * c2).squeeze(1)
        return self.head(fused).squeeze(-1)


class CATRegressor:
    """sklearn-style wrapper so CAT-lite runs inside the PoC backtest."""

    features = SEQ_COLS + CONTEXT + ["year"]

    def __init__(self, seeds=(0, 1, 2), epochs: int = 400, patience: int = 40,
                 lr: float = 3e-3, weight_decay: float = 1e-2, batch: int = 64):
        self.seeds, self.epochs, self.patience = seeds, epochs, patience
        self.lr, self.wd, self.batch = lr, weight_decay, batch
        torch.set_num_threads(4)

    def _arrays(self, X):
        seq = X[SEQ_COLS].to_numpy(np.float32).reshape(len(X), 12, len(CLIMATE_VARS))
        ctx = X[CONTEXT].to_numpy(np.float32)
        return seq, ctx

    def fit(self, X: pd.DataFrame, y):
        seq, ctx = self._arrays(X)

        def safe_sd(s):
            # A feature constant in training (e.g. LSPA before 2011) keeps scale 1,
            # like sklearn's StandardScaler; dividing by ~0 explodes at test time.
            return np.where(s < 1e-6, 1.0, s).astype(np.float32)

        self.seq_mu, self.seq_sd = seq.mean((0, 1)), safe_sd(seq.std((0, 1)))
        self.ctx_mu, self.ctx_sd = ctx.mean(0), safe_sd(ctx.std(0))
        y = np.asarray(y, np.float32)
        self.y_mu, self.y_sd = y.mean(), y.std() + 1e-6
        S = torch.tensor((seq - self.seq_mu) / self.seq_sd)
        C = torch.tensor((ctx - self.ctx_mu) / self.ctx_sd)
        Y = torch.tensor((y - self.y_mu) / self.y_sd)
        # Early-stopping validation = the most recent known year (time-respecting)
        years = X["year"].to_numpy()
        val = years == years.max()
        tr = ~val
        self.models = [self._fit_one(seed, S, C, Y, tr, val) for seed in self.seeds]
        return self

    def _fit_one(self, seed, S, C, Y, tr, val):
        torch.manual_seed(seed)
        model = CATLite(len(CLIMATE_VARS), len(CONTEXT))
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        n_tr = int(tr.sum())
        steps = self.epochs * max(1, math.ceil(n_tr / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps,
                                                    pct_start=0.2)
        loss_fn = nn.HuberLoss(delta=1.0)
        idx_tr = np.where(tr)[0]
        best, best_state, wait = float("inf"), None, 0
        g = torch.Generator().manual_seed(seed)
        for _ in range(self.epochs):
            model.train()
            perm = torch.tensor(idx_tr)[torch.randperm(n_tr, generator=g)]
            for i in range(0, n_tr, self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(model(S[b], C[b]), Y[b])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(S[val], C[val]), Y[val]).item()
            if v < best - 1e-4:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        seq, ctx = self._arrays(X)
        S = torch.tensor((seq - self.seq_mu) / self.seq_sd)
        C = torch.tensor((ctx - self.ctx_mu) / self.ctx_sd)
        with torch.no_grad():
            preds = [m(S, C).numpy() for m in self.models]
            self.pool_weights_ = np.mean([m.last_pool_weights.numpy() for m in self.models], 0)
        return np.mean(preds, 0) * self.y_sd + self.y_mu

    def learned_phenology(self) -> np.ndarray:
        """Learned phenology bias, as relative weights (softmax), averaged over seeds."""
        b = np.mean([m.pheno_bias.detach().numpy() for m in self.models], 0)
        return np.exp(b) / np.exp(b).sum()
