"""The CHF pipeline: physics backbone + trust-scaled bounded correction.

Design follows what the measurements support.

  Layer 0  fluid properties (CoolProp) and dimensionless groups. The only reason an
           unseen fluid is predictable at all: the model never sees the name
           "R-134a", it sees that fluid's density, surface tension, latent heat.

  Layer 1  a physics backbone with ZERO fitted parameters. Which correlation serves
           as the backbone (Katto-Ohno / Biasi / 2006 LUT) is selected on TRAINING
           rows only, by median absolute log error among the correlations valid for
           that fluid. That is one discrete choice made on training data -- it never
           sees a test row.

  Layer 2  one pooled ExtraTrees correction predicting log(CHF / backbone). Pooled,
           not per-dataset: per-dataset routing failed twice on this corpus.

  Layer 3  trust scales the BOUND on that correction; it never picks a model.
           Routing is a switch and every switch error costs a factor of 1000 (the
           router hit -1276 on an unseen fluid). Bounding is a dial with a measured
           monotonic trade-off. Low trust => tight clip => the prediction decays into
           the correlation rather than falling off a cliff.

           A trust-weighted blend toward a free (unanchored) pooled model was
           tried as a deployable replacement for per-dataset fallback and is OFF by
           default because it MEASURABLY HARMS generalisation: held-out D7 fell from
           R2 0.792 to -206.9. The cause is that trust measures distance in
           dimensionless space, and a new laboratory's rows sit *inside* that
           envelope (trust 0.72) while still being unseen. Trust detects unfamiliar
           physics, not an unfamiliar source -- so it must not be used to hand rows
           to an unanchored model. Keep `blend_free=False` unless re-measured.

  Layer 4  split-conformal intervals, quantiled separately per trust band
           (Mondrian). Required, not optional: this project's deep ensembles produce
           nominal 95% intervals with 9-22% actual coverage.

Ablation result worth knowing: the three trust signals are near-interchangeable
(median R2 0.736-0.753 whichever is used alone). The bounded-correction structure
does the work, not the gate -- which is why the pipeline is robust where the router
was brittle.
"""

from __future__ import annotations

import gzip
import pickle

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
from sklearn.ensemble import ExtraTreesRegressor

import physics_arms as P

DIM = P.DIMLESS
BOUND_MIN = 1.5      # tightest clip on the correction, at trust = 0
BOUND_MAX = 10.0     # loosest clip, at trust = 1
TRUST_BANDS = (0.33, 0.66)


# ----------------------------------------------------------------- layer 0
def prepare(ds) -> pd.DataFrame:
    """Attach properties, dimensionless groups and every physics arm."""
    df = P.enrich(ds).reset_index(drop=True)
    df = df[(df["CHF_kW_m2"] > 0) & np.isfinite(df["CHF_kW_m2"])].reset_index(drop=True)
    return _attach_physics(df, ds.fluid)


def _attach_physics(df: pd.DataFrame, fluid: str) -> pd.DataFrame:

    df["phys_katto"] = np.clip(P.phys_katto(df), 1e-3, None)
    if fluid == "water" and "X" in df.columns:
        df["phys_biasi"] = np.clip(P.phys_biasi(df), 1e-3, None)
        df["phys_lut"] = np.clip(P.phys_lut(df), 1e-3, None)
    else:
        # Biasi is water-only; the LUT is water-only and indexed on local quality
        df["phys_biasi"] = np.nan
        df["phys_lut"] = np.nan
    arms = df[["phys_katto", "phys_biasi", "phys_lut"]].to_numpy(float)
    with np.errstate(all="ignore"):
        la = np.log(np.where(arms > 0, arms, np.nan))
    df["corr_spread"] = pd.Series(np.nanstd(la, axis=1)).fillna(0.0).to_numpy()
    return df


def mahalanobis(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    t = np.log10(np.clip(train, 1e-12, None))
    e = np.log10(np.clip(test, 1e-12, None))
    mu = np.nanmedian(t, axis=0)
    cov = np.cov(np.nan_to_num(t - mu, nan=0.0), rowvar=False) + np.eye(t.shape[1]) * 1e-6
    inv = np.linalg.pinv(cov)
    d = np.nan_to_num(e - mu, nan=0.0)
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", d, inv, d), 0, None))



# Aliases for fluids the corpus uses; anything else is passed to CoolProp directly
# so a user is not silently limited to three fluids -- but an unrecognised name
# RAISES rather than falling back to water, which would return a confident wrong
# number computed from the wrong substance.
FLUID_ALIAS = {"water": "Water", "h2o": "Water", "steam": "Water",
               "r-123": "R123", "r123": "R123",
               "r-134a": "R134a", "r134a": "R134a"}


def resolve_fluid(name: str) -> str:
    key = str(name).strip().lower()
    cp = FLUID_ALIAS.get(key, str(name).strip())
    try:
        PropsSI("Pcrit", cp)
    except Exception as exc:
        raise ValueError(
            f"unknown fluid {name!r}: not a CoolProp substance. "
            f"Known aliases: {sorted(set(FLUID_ALIAS))}"
        ) from exc
    return cp


def validate_inputs(fluid_cp: str, P_kPa: float, G_kg_m2s: float,
                    D_mm: float, L_mm: float) -> list:
    """Reject physically impossible requests. Returns non-fatal warnings."""
    if not np.isfinite([P_kPa, G_kg_m2s, D_mm, L_mm]).all():
        raise ValueError("P, G, D and L must all be finite numbers")
    if D_mm <= 0 or L_mm <= 0:
        raise ValueError(f"diameter and heated length must be positive "
                         f"(got D={D_mm} mm, L={L_mm} mm)")
    if P_kPa <= 0:
        raise ValueError(f"pressure must be positive (got {P_kPa} kPa)")
    pc = PropsSI("Pcrit", fluid_cp) / 1e3
    if P_kPa >= pc:
        raise ValueError(
            f"pressure {P_kPa} kPa is at or above the critical pressure of "
            f"{fluid_cp} ({pc:.0f} kPa). CHF is not defined there.")
    if G_kg_m2s <= 0:
        raise ValueError(
            "mass flux must be positive. This pipeline covers FLOW boiling only; "
            "G = 0 is pool boiling and needs a pool-boiling backbone (Zuber), "
            "which is not fitted here.")
    warn = []
    if G_kg_m2s > 10000:
        warn.append(f"mass flux {G_kg_m2s:g} kg/m2s is far above the training "
                    f"range (max ~8000); treat the result as an extrapolation")
    if not (1.0 <= D_mm <= 50.0):
        warn.append(f"diameter {D_mm:g} mm is outside the training range 1-40 mm")
    return warn


ARMS = ["phys_katto", "phys_biasi", "phys_lut"]


class CHFPipeline:
    """Fit on pooled rows; predict CHF with a calibrated interval and a trust label."""

    def __init__(self, alpha: float = 0.10, seed: int = 0, use_spread: bool = True,
                 blend_free: bool = False, backbone: str | None = None,
                 conformal: str = 'global'):
        self.alpha = alpha            # target coverage is 1 - alpha
        self.seed = seed
        self.use_spread = use_spread
        self.blend_free = blend_free
        self.backbone = backbone      # None => select on training rows
        # 'global' one quantile for all rows; 'band' a Mondrian quantile per
        # trust band. 'band' is tighter in-distribution but WORSE on a held-out
        # laboratory, because trust rates a new lab as high-trust and hands it
        # the tightest interval. Measured, see pipeline_coverage.csv.
        self.conformal = conformal

    # ------------------------------------------------------- layer 1 select
    def _select_backbone(self, tr: pd.DataFrame) -> str:
        if self.backbone:
            return self.backbone
        y = tr["CHF_kW_m2"].to_numpy(float)
        best, best_err = "phys_katto", np.inf
        for arm in ARMS:
            if arm not in tr:
                continue
            p = tr[arm].to_numpy(float)
            m = np.isfinite(p) & (p > 0)
            if m.sum() < 0.9 * len(tr):        # must be valid on almost every row
                continue
            err = float(np.median(np.abs(np.log(p[m] / y[m]))))
            if err < best_err:
                best, best_err = arm, err
        return best

    # ------------------------------------------------------------- fit
    def fit(self, tr: pd.DataFrame):
        rng = np.random.RandomState(self.seed)
        y = tr["CHF_kW_m2"].to_numpy(float)
        self.backbone_ = self._select_backbone(tr)
        base = np.clip(tr[self.backbone_].to_numpy(float), 1e-3, None)
        X = np.nan_to_num(tr[DIM].to_numpy(float), nan=0.0)
        target = np.log(np.clip(y / base, 1e-6, None))
        ok = np.isfinite(target)

        idx = np.where(ok)[0]
        rng.shuffle(idx)
        n_cal = max(30, int(0.2 * len(idx)))
        cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]

        # ---- layer 2: anchored correction, and optionally a free model
        self.model_ = ExtraTreesRegressor(n_estimators=300, n_jobs=-1,
                                          random_state=self.seed)
        self.model_.fit(X[fit_idx], target[fit_idx])
        if self.blend_free:
            self.free_ = ExtraTreesRegressor(n_estimators=300, n_jobs=-1,
                                             random_state=self.seed)
            self.free_.fit(X[fit_idx], np.log(np.clip(y[fit_idx], 1e-6, None)))

        # ---- layer 3: trust calibration, from the FIT rows only.
        # Building the Mahalanobis cloud from all training rows would include the
        # conformal calibration slice, so a calibration row's novelty would be
        # measured against a cloud containing itself -- trust comes out optimistic
        # and the conformal quantile too tight.
        self.train_dim_ = tr[DIM].to_numpy(float)[fit_idx]
        d_tr = mahalanobis(self.train_dim_, self.train_dim_)
        self.d_lo_, self.d_hi_ = np.percentile(d_tr, [50, 95])
        s_tr = tr["corr_spread"].to_numpy(float)
        self.s_lo_, self.s_hi_ = np.percentile(s_tr, [50, 95])
        self.fluids_ = set(tr["fluid_name"]) if "fluid_name" in tr else set()

        # ---- layer 4: Mondrian conformal -- one quantile per trust band
        cal = tr.iloc[cal_idx]
        resid = np.abs(np.log(np.clip(cal["CHF_kW_m2"].to_numpy(float), 1e-9, None))
                       - np.log(np.clip(self._point(cal), 1e-9, None)))
        t_cal = self.trust(cal)
        q = min(1 - self.alpha, 0.999)
        self.q_global_ = float(np.quantile(resid, q))
        self.q_band_ = {}
        for b, m in self._bands(t_cal).items():
            self.q_band_[b] = (float(np.quantile(resid[m], q)) if m.sum() >= 20
                               else self.q_global_)
        return self

    @staticmethod
    def _bands(t: np.ndarray) -> dict:
        lo, hi = TRUST_BANDS
        return {"low": t < lo, "mid": (t >= lo) & (t < hi), "high": t >= hi}

    # ------------------------------------------------------- trust signal
    def trust(self, te: pd.DataFrame) -> np.ndarray:
        def norm(v, lo, hi):
            return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)

        sig = [norm(mahalanobis(self.train_dim_, te[DIM].to_numpy(float)),
                    self.d_lo_, self.d_hi_)]
        if self.use_spread:
            sig.append(norm(te["corr_spread"].to_numpy(float), self.s_lo_, self.s_hi_))
        if self.fluids_ and "fluid_name" in te:
            sig.append((~te["fluid_name"].isin(self.fluids_)).to_numpy(float))
        return np.clip(1.0 - np.maximum.reduce(sig), 0.0, 1.0)

    # ------------------------------------------------------- prediction
    def _point(self, te: pd.DataFrame) -> np.ndarray:
        base = np.clip(te[self.backbone_].to_numpy(float), 1e-3, None)
        X = np.nan_to_num(te[DIM].to_numpy(float), nan=0.0)
        t = self.trust(te)
        cap = np.log(BOUND_MIN) + t * (np.log(BOUND_MAX) - np.log(BOUND_MIN))
        anchored = np.log(base) + np.clip(self.model_.predict(X), -cap, cap)
        if not self.blend_free:
            return np.exp(anchored)
        # high trust -> lean on the free model, low trust -> lean on the anchor
        free = self.free_.predict(X)
        return np.exp(t * free + (1.0 - t) * anchored)

    def predict(self, te: pd.DataFrame) -> pd.DataFrame:
        p = self._point(te)
        t = self.trust(te)
        if self.conformal == "band":
            w = np.empty(len(te))
            for b, m in self._bands(t).items():
                w[m] = self.q_band_.get(b, self.q_global_)
        else:
            # widen as trust falls, from a single global quantile
            w = self.q_global_ * (1.0 + 2.0 * (1.0 - t))
        label = np.where(t >= TRUST_BANDS[1], "interpolating",
                 np.where(t >= TRUST_BANDS[0], "extrapolating", "correlation fallback"))
        return pd.DataFrame({
            "chf": p, "lo": p * np.exp(-w), "hi": p * np.exp(w),
            "trust": t, "regime": label,
            "physics_only": te[self.backbone_].to_numpy(float),
        }, index=te.index)

    # ------------------------------------------------------- deployment
    def save(self, path: str):
        """Gzip the pickle: 300 trees over 28k rows is ~900 MB raw.

        The reference cloud is subsampled for the file only. An earlier version
        truncated `self.train_dim_` in place, so calling save() silently shrank the
        live object -- predictions happened to match here, but a method that mutates
        the thing it is serialising is not something to hand over.
        """
        import copy
        shrunk = copy.copy(self)
        if len(self.train_dim_) > 5000:
            rng = np.random.RandomState(0)
            shrunk.train_dim_ = self.train_dim_[
                rng.choice(len(self.train_dim_), 5000, replace=False)]
        with gzip.open(path, "wb", compresslevel=6) as f:
            pickle.dump(shrunk, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "CHFPipeline":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    def predict_one(self, fluid: str, P_kPa: float, G_kg_m2s: float, D_mm: float,
                    L_mm: float, Tin_C: float | None = None,
                    dHin_sub_kJkg: float | None = None) -> dict:
        """Predict from the quantities an experimenter knows BEFORE the test.

        No outlet quality is asked for, by design: it does not exist until CHF has
        already happened. Returns the prediction, a calibrated interval and a plain
        statement of how far outside the training data the request sits.
        """
        fluid_cp = resolve_fluid(fluid)
        warns = validate_inputs(fluid_cp, P_kPa, G_kg_m2s, D_mm, L_mm)
        known = fluid_cp in {resolve_fluid(f) for f in self.fluids_} if self.fluids_ else False
        if not known:
            warns.append(f"no training data for {fluid}; the prediction rests on the "
                         f"physics correlation, which uses that fluid's real properties")
        rec = {"P_kPa": P_kPa, "G_kg_m2s": G_kg_m2s, "D_mm": D_mm, "L_mm": L_mm,
               "CHF_kW_m2": 1.0}   # placeholder; never read by prediction
        # only create the inlet columns that were actually supplied -- an all-None
        # column still satisfies `in df` and poisons enrich() with NaN
        if dHin_sub_kJkg is not None:
            rec["dHin_sub_kJkg"] = dHin_sub_kJkg
        elif Tin_C is not None:
            rec["Tin_C"] = Tin_C
        row = pd.DataFrame([rec])
        canon = {"Water": "water", "R123": "R-123", "R134a": "R-134a"}.get(fluid_cp, fluid_cp)
        ds = type("D", (), {"df": row, "fluid": canon, "name": "query"})()
        f = _attach_physics(P.enrich(ds).reset_index(drop=True), canon)
        f["fluid_name"] = canon
        r = self.predict(f).iloc[0]
        chf = float(r.chf)
        if not np.isfinite(chf) or chf <= 0:
            raise ValueError("the physics backbone returned no usable value for these "
                             "inputs; they are outside its validity range")
        return {"CHF_kW_m2": round(chf, 1),
                "interval_kW_m2": (round(float(r.lo), 1), round(float(r.hi), 1)),
                "confidence": str(r.regime), "trust": round(float(r.trust), 3),
                "physics_only_kW_m2": round(float(r.physics_only), 1),
                "backbone": self.backbone_, "warnings": warns}
