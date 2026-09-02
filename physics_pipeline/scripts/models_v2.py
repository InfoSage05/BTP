"""
models_v2.py
------------
The physics-corrected learner:

    CHF = Phi_physics * exp( g_theta(pi(x)) )

`g_theta` is a bounded, optionally monotone, optionally trust-decaying
correction learned in log space. Three structural guarantees, each replacing
a symptom-level patch in the original pipeline:

1. BOUNDED OUTPUT (idea 5). `g_theta` is clipped to +/- ln(BOUND), so the
   correction can never move the physics answer by more than a factor of
   BOUND. The original pipeline instead clipped the FINAL prediction at
   CHF_CLIP_MAX = 30,000 kW/m^2 -- a post-hoc guard added after an
   MLPRegressor blow-up turned one fold's R^2 into -1,600,000. Bounding the
   correction rather than the answer means the guarantee holds in every
   regime, including ones where 30,000 kW/m^2 is not a meaningful ceiling.

2. MONOTONICITY (constraints C3, C4). Enforced natively by
   HistGradientBoostingRegressor's `monotonic_cst`. Because Phi_physics is
   itself monotone in the same direction, a monotone correction is a
   SUFFICIENT condition for the product to be monotone -- stronger than
   strictly necessary, but exactly checkable, which is the point.

3. TRUST DECAY. `g_theta -> 0` as a query moves away from the training
   manifold, so out-of-domain predictions fall back to pure physics instead
   of extrapolating a learned correction. This is the direct fix for the
   documented Approach-2 failure ("the formula's error changed shape
   completely" outside the training range).
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

#: Maximum multiplicative correction. ln(3) means the learner may scale the
#: physics answer by at most 3x in either direction.
DEFAULT_CORRECTION_BOUND = 3.0

#: Fallback absolute clip, kept only as a last-resort sanity guard for the
#: unbounded arms of the ablation (A0-A4). Observed data max is 28,800 kW/m^2.
CHF_CLIP_MAX = 30_000.0

# Features whose correction must be monotone, by feature space. Sign is the
# required direction of d(log correction)/d(feature).
MONOTONE_BY_SPACE = {
    "raw": {
        "quality": -1,            # C3: CHF decreases with quality
        "mass_flux_kg_m2s": +1,   # C4: CHF increases with mass flux
    },
    "pi": {
        "pi_quality": -1,         # C3
        "pi_We_D": +1,            # C4: We_D ~ G^2, so increasing in G
        "pi_katto": -1,           # C4: Katto number ~ 1/G^2, so decreasing in G
    },
}
# The correlation-stacked spaces reuse their parent space's constraints; the
# stacked log-Bo features are left unconstrained on purpose, because the
# correction's dependence on "what Katto-Ohno thinks" has no required sign.
MONOTONE_BY_SPACE["pi_corr"] = MONOTONE_BY_SPACE["pi"]
MONOTONE_BY_SPACE["raw_corr"] = MONOTONE_BY_SPACE["raw"]


def monotonic_vector(feature_names, space):
    """sklearn `monotonic_cst` vector for the given feature space."""
    wanted = MONOTONE_BY_SPACE.get(space, {})
    return [wanted.get(name, 0) for name in feature_names]


class PhysicsCorrectedModel:
    """Learner for log(CHF / Phi_physics) with optional structural guarantees.

    Parameters
    ----------
    base : {"histgb", "ann"}
    bound : float or None
        Maximum multiplicative correction. None disables bounding.
    monotone : bool
        Apply monotonic constraints (HistGB only; ignored for the ANN, which
        has no native support).
    trust_decay : bool
        Shrink the correction toward zero away from the training manifold.
    space : {"raw", "pi"}
        Which feature space -- determines the monotonic constraint mapping.
    """

    def __init__(self, base="histgb", bound=None, monotone=False,
                 trust_decay=False, space="pi", trust_gamma=1.0):
        self.base = base
        self.bound = bound
        self.monotone = monotone
        self.trust_decay = trust_decay
        self.space = space
        self.trust_gamma = trust_gamma
        self.model_ = None
        self._nn = None
        self._nn_pipe = None
        self._d_ref = None
        self._monotone_applied = False

    # -- internals ---------------------------------------------------------

    def _build(self, feature_names):
        if self.base == "ann":
            return make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                             early_stopping=True, random_state=RANDOM_STATE),
            )
        kwargs = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
                      random_state=RANDOM_STATE)
        if self.monotone:
            cst = monotonic_vector(feature_names, self.space)
            if any(c != 0 for c in cst):
                kwargs["monotonic_cst"] = cst
                self._monotone_applied = True
        return HistGradientBoostingRegressor(**kwargs)

    def _fit_trust(self, X):
        """Fit the training-manifold distance model used for trust decay."""
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        Z = pipe.fit_transform(X)
        # 2 neighbours: the first is the point itself for training rows.
        nn = NearestNeighbors(n_neighbors=min(2, len(Z))).fit(Z)
        d, _ = nn.kneighbors(Z)
        d_self = d[:, -1]
        self._nn_pipe, self._nn = pipe, nn
        # Reference scale: a distance this large is "at the edge of" the data.
        self._d_ref = float(np.percentile(d_self[np.isfinite(d_self)], 95)) or 1.0
        if not np.isfinite(self._d_ref) or self._d_ref <= 0:
            self._d_ref = 1.0

    def _trust_weight(self, X):
        if not self.trust_decay or self._nn is None:
            return 1.0
        Z = self._nn_pipe.transform(X)
        d, _ = self._nn.kneighbors(Z)
        d_near = d[:, 0]
        return np.exp(-self.trust_gamma * np.clip(d_near / self._d_ref - 1.0, 0.0, None))

    # -- public API --------------------------------------------------------

    def fit(self, X, y, baseline, sample_weight=None):
        X = pd.DataFrame(X)
        target = np.log(np.maximum(np.asarray(y, float), 1e-9)
                        / np.maximum(np.asarray(baseline, float), 1e-9))

        # --- baseline calibration ------------------------------------------
        # The bound is a statement about how far the model may move the
        # PHYSICS ANSWER, so it is only meaningful if the baseline is itself a
        # calibrated CHF estimate. Katto-Ohno and the gated baseline are
        # (median CHF/Phi = 0.97 and 0.98); the latent-heat scale G*h_fg is
        # not -- it sits about 1400x above actual CHF, because it is a flux
        # scale rather than a CHF prediction.
        #
        # Without this correction, bounding a latent-baseline model to 3x
        # pinned every prediction near G*h_fg and produced R2 of -2.3e6. The
        # fix is to absorb the constant offset first: the bound then applies
        # to the residual AFTER calibration, which is what was intended all
        # along, and it makes the bound well-defined for any baseline.
        #
        # The offset is estimated on TRAINING rows only, so it carries no
        # information about the held-out source.
        #
        # It is applied ONLY when the baseline actually needs it, i.e. when the
        # median log-ratio already exceeds the bound. Applying it
        # unconditionally measurably hurts: with the calibrated Katto-Ohno
        # baseline it moved pooled LOSO from 0.710 to 0.658, because the
        # per-rig scale offsets are large and inconsistent (median CHF/Phi runs
        # 0.97 on NRC, 1.49 on kaeri_uniform, 1.81 on zhao2020, 2.59 on
        # pin-fin), so an offset fitted on the other six sources transfers the
        # WRONG rig's calibration onto the held-out one. Where the physics is
        # already calibrated, leaving it alone is strictly better.
        finite = np.isfinite(target)
        median_log_ratio = float(np.median(target[finite])) if finite.any() else 0.0
        needs_calibration = (self.bound is not None
                             and abs(median_log_ratio) > np.log(self.bound))
        self.calibration_ = median_log_ratio if needs_calibration else 0.0
        self.calibration_applied_ = bool(needs_calibration)

        self.model_ = self._build(list(X.columns))

        t0 = time.time()
        try:
            self.model_.fit(X, target, sample_weight=sample_weight)
        except (TypeError, ValueError):
            # MLPRegressor has no sample_weight; a monotonic_cst rejection
            # also lands here on older sklearn, so retry unconstrained.
            try:
                self.model_.fit(X, target)
            except ValueError:
                self._monotone_applied = False
                self.monotone = False
                self.model_ = self._build(list(X.columns))
                self.model_.fit(X, target)
        self.train_seconds_ = time.time() - t0

        if self.trust_decay:
            self._fit_trust(X)
        return self

    def predict(self, X, baseline):
        X = pd.DataFrame(X)
        g = np.asarray(self.model_.predict(X), dtype=float)

        # Split the prediction into the constant calibration offset and the
        # row-dependent correction. Only the correction is bounded and decayed;
        # the offset is a property of the baseline, not of the learner, so
        # switching the learner off must not discard it.
        c = getattr(self, "calibration_", 0.0)
        g = g - c

        if self.bound is not None:
            lim = np.log(self.bound)
            g = np.clip(g, -lim, lim)

        g = c + g * self._trust_weight(X)

        pred = np.exp(g) * np.asarray(baseline, dtype=float)
        pred = np.where(np.isfinite(pred), pred, np.nan)
        if self.bound is None:
            # Unbounded arms still get the legacy absolute clip so a single
            # blow-up cannot make a whole fold's metrics meaningless.
            pred = np.clip(pred, 0.0, CHF_CLIP_MAX)
        else:
            pred = np.maximum(pred, 0.0)
        return pred
