"""
pinn_model.py
--------------
A physics-informed neural network (PINN) for this project's CHF target
(log(CHF / physics_baseline), same target every other model in models.py
uses). Built after reading this repo's prior PINN work
(results/pinn/tier1-2b_summary.json, docs/CHF_Physics_Approaches_Explained.md)
on a related extrapolation problem, whose evidence directly shaped this
design:

- A collocation-based physics-PENALTY loss (monotonicity + Zuber-trend terms)
  was tried extensively there (tier1, tier2) and did NOT reliably beat a
  plain baseline; the best-found configuration in tier2b had BOTH penalty
  weights at 0.0. So this implementation treats the physics-penalty weight
  (`lam_mono`) as something to validate, not something to assume helps --
  see PINN_LAMBDA_GRID below, tested including lam_mono=0.
- A plain additive residual-vs-correlation model failed under extrapolation
  there because the correction learned inside the training range gets
  misapplied outside it. This PINN instead predicts the SAME physics-ratio
  target the rest of this pipeline already validated (see models.py's
  docstring), not a raw additive residual.
- Plain ANN (sklearn MLPRegressor) in THIS pipeline is catastrophically
  unstable under extrapolation (pooled LOSO R2 = -5.15, individual folds in
  the negative hundreds of thousands) because it's an unbounded function and
  log-space training turns any large wrong output into an astronomical CHF
  after exp(). This PINN's output layer is explicitly bounded
  (`OUTPUT_BOUND * tanh(raw / OUTPUT_BOUND)`) to remove that specific failure
  mode by construction, independent of whether the physics penalty helps.

Physics penalty (only meaningful, well-established, direction-only trends
are used -- not a pressure-monotonicity term, which is what made the prior
work's Zuber-trend penalty fragile, since CHF vs. pressure is genuinely
non-monotonic):
  - dCHF/d(mass_flux) >= 0   (higher flow -> better cooling -> higher CHF)
  - dCHF/d(quality)    <= 0  (more vapor -> earlier dryout -> lower CHF)
Both are evaluated via autograd on each training batch (flow-boiling rows
only -- mass_flux_kg_m2s > 0), not on separately-sampled collocation points,
to keep this first attempt simple; extending to true collocation points is a
natural next step if this shows promise.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from features import FEATURE_COLUMNS

RANDOM_STATE = 42
OUTPUT_BOUND = 15.0  # observed log-ratio target spans roughly -11 to +9

MASS_FLUX_IDX = FEATURE_COLUMNS.index("mass_flux_kg_m2s")
QUALITY_IDX = FEATURE_COLUMNS.index("quality")

# Swept during fit(); the prior work's evidence says lam_mono=0 may well win
# again here too -- that's a legitimate, honestly-reported outcome, not a
# failure of the implementation.
PINN_LAMBDA_GRID = [0.0, 0.05, 0.2]


class _PinnNet(nn.Module):
    def __init__(self, n_features: int, hidden=(32, 16)):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.net(x).squeeze(-1)
        return OUTPUT_BOUND * torch.tanh(raw / OUTPUT_BOUND)


class PINNRegressor:
    """sklearn-style estimator: fit(X_df, y_log_ratio, sample_weight=None) /
    predict(X_df) -> y_log_ratio. Plugs directly into
    models.fit_predict_sklearn_model exactly like RF/HistGB/ANN, since it
    consumes and produces the same log-ratio-space values."""

    def __init__(self, hidden=(32, 16), epochs=200, batch_size=1024, lr=1e-3,
                 lam_mono_grid=None, val_frac=0.1, patience=15, random_state=RANDOM_STATE):
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lam_mono_grid = lam_mono_grid or PINN_LAMBDA_GRID
        self.val_frac = val_frac
        self.patience = patience
        self.random_state = random_state
        self.best_lam_mono_ = None

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        X = X[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        X = np.where(np.isnan(X), self._medians_, X)
        X = (X - self._mean_) / self._std_
        return X.astype(np.float32)

    def _train_one(self, Xtr, ytr, wtr, Xval, yval, lam_mono, seed):
        torch.manual_seed(seed)
        net = _PinnNet(Xtr.shape[1], self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)

        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.float32)
        wtr_t = torch.tensor(wtr, dtype=torch.float32)
        flow_mask_t = torch.tensor(Xtr[:, MASS_FLUX_IDX] > self._mass_flux_zero_scaled_, dtype=torch.bool)
        Xval_t = torch.tensor(Xval, dtype=torch.float32)
        yval_t = torch.tensor(yval, dtype=torch.float32)

        n = Xtr_t.shape[0]
        best_val, best_state, bad_epochs = float("inf"), None, 0
        rng = np.random.default_rng(seed)

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            net.train()
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb = Xtr_t[idx].clone().requires_grad_(lam_mono > 0)
                yb, wb, flow_b = ytr_t[idx], wtr_t[idx], flow_mask_t[idx]

                pred = net(xb)
                data_loss = (wb * (pred - yb) ** 2).mean()

                phys_loss = torch.tensor(0.0)
                if lam_mono > 0 and flow_b.any():
                    grads = torch.autograd.grad(pred.sum(), xb, create_graph=True)[0]
                    d_mass_flux = grads[:, MASS_FLUX_IDX][flow_b]
                    d_quality = grads[:, QUALITY_IDX][flow_b]
                    phys_loss = (torch.relu(-d_mass_flux).mean() + torch.relu(d_quality).mean())

                loss = data_loss + lam_mono * phys_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

            net.eval()
            with torch.no_grad():
                val_pred = net(Xval_t)
                val_loss = ((val_pred - yval_t) ** 2).mean().item()
            if val_loss < best_val - 1e-5:
                best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        net.load_state_dict(best_state)
        return net, best_val

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight=None):
        X_arr = X[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        self._medians_ = np.nanmedian(X_arr, axis=0)
        X_filled = np.where(np.isnan(X_arr), self._medians_, X_arr)
        self._mean_ = X_filled.mean(axis=0)
        self._std_ = X_filled.std(axis=0)
        self._std_[self._std_ == 0] = 1.0
        self._mass_flux_zero_scaled_ = (0.0 - self._mean_[MASS_FLUX_IDX]) / self._std_[MASS_FLUX_IDX]

        X_scaled = ((X_filled - self._mean_) / self._std_).astype(np.float32)
        y = np.asarray(y, dtype=np.float32)
        w = np.ones(len(y), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)

        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(y))
        n_val = max(1, int(len(y) * self.val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        best_overall = (None, None, float("inf"))
        for lam_mono in self.lam_mono_grid:
            net, val_loss = self._train_one(
                X_scaled[tr_idx], y[tr_idx], w[tr_idx],
                X_scaled[val_idx], y[val_idx],
                lam_mono, seed=self.random_state)
            if val_loss < best_overall[2]:
                best_overall = (net, lam_mono, val_loss)

        self._net_, self.best_lam_mono_, _ = best_overall
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self._prep(X)
        self._net_.eval()
        with torch.no_grad():
            pred = self._net_(torch.tensor(X_scaled, dtype=torch.float32))
        return pred.numpy()


def build_pinn():
    return PINNRegressor()
