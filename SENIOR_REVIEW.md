# Senior ML Scientist Review — CHF Prediction Project

Reviewing `CHF_ML_Modeling.ipynb` (Phase 1, the scoped deliverable) and
`CHF_Physics_Informed_Extensions.ipynb` (Phase 2, exploratory extensions),
plus the `model_tests/` per-model validation notebooks and `chf_physics.py`.

---

## ⚠ HEADLINE RETRACTION — read this before anything else

**An earlier version of this review ranked the pressure-gated blend first at
R² = 0.855 on Split C. That claim is RETRACTED.** Independent multi-seed
verification (`verify_results.py` → `results/split_C_multiseed_verification.csv`)
shows it was a single-seed artifact:

- Seed 42 reproduces 0.855 exactly — the code is not in question.
- Across **30 independent seeds**, the raw-target MLP driving that blend
  averages **0.547** and **not one of the 30 reached 0.84**.
- Over 10 seeds: gated blend (raw MLP) = 0.466 ± 0.228, worst seed 0.133.

**Corrected Split C ranking (verified):**

| Model | Mean R² | Std | Worst | MAPE | Reproducible? |
|---|---|---|---|---|---|
| GridInterp (raw) | **0.8415** | 0 | 0.8415 | **20.8%** | exact |
| GridInterp (log) | 0.8040 | 0 | 0.8040 | 26.7% | exact |
| **Poly2_Ridge (log)** | **0.7547** | 0 | 0.7547 | 35.8% | exact |
| GatedBlend (log MLP) | 0.6284 | 0.071 | 0.515 | 39.8% | stochastic |
| MLP (log) | 0.6277 | 0.072 | 0.515 | 40.0% | stochastic |
| GatedBlend (raw MLP) | 0.4658 | 0.228 | 0.133 | 67.1% | stochastic |
| MLP (raw) | 0.4412 | 0.246 | 0.081 | 70.5% | stochastic |
| Tree ensembles | ~0.43 | ~0 | 0.43 | ~42% | effectively exact |

**What to present**: the tree-collapse contrast (robust), with degree-2
log-Ridge (R² = 0.755, deterministic) as the best *trained* ML extrapolator
and grid interpolation (0.841) as the physical baseline — which, per the
project context, should not be framed as something ML "beat."

**Also corrected in the Phase-2 notebook this pass**: (a) the Zuber-penalty
markdown claimed collocation points were "G=0-flavored" when the code samples
G uniformly across 0–8000 — a pool-boiling correlation applied across all
mass fluxes is an approximation, now stated as such; (b) the monotonicity
penalty reads `0.0000` throughout training because it is **non-binding
(inactive)**, not because physics is being successfully enforced.

---

## 1. Was the data clean, and was more EDA needed?

**No further cleaning is needed.** `chf_long_clean.csv` was verified
end-to-end against every fact claimed in the project context: 11,592 raw
grid rows (24 P × 21 G × 23 X), zero duplicates, zero nulls, exactly 504
`CHF==0` rows all at `X==1.0` (correctly excluded as a placeholder boundary,
not a real target), and a post-filter non-zero CHF range of 15–44,338 kW/m²
matching the stated ~3-order-of-magnitude spread. `prepare_data.py` was
reviewed but not re-run — the CSV it produced is trustworthy.

**EDA was sufficient for the stated goal but revealed one thing worth
flagging for future thesis work**: the Groeneveld 2007 paper (found while
researching PINNs, see Section 3 below) documents a **Limiting Quality
Region (LQR)** — a steep, non-smooth transition zone in the CHF-vs-X curve
where the table's own authors describe higher table-entry-to-table-entry
variation. None of the models in this project were evaluated *specifically*
inside the LQR (the EDA slices and the Split C slice plot happen not to
land on it). If a future pass of this thesis wants to characterize *where*
models fail, not just *whether* they fail under extrapolation, an EDA
section specifically isolating LQR rows and comparing per-model error there
would be a natural next step. This is a scope note, not a defect in the
delivered notebook.

## 2. Architecture and methodology assessment (Phase 1)

The Section 4 model list and Section 5 three-split protocol are sound and
were followed faithfully. Two genuine implementation bugs were caught and
fixed during development, both worth recording because they're the kind of
subtle correctness issue that silently produces plausible-looking wrong
numbers if unchecked:

- **Split B's "every 4th pressure level" naively implemented as
  `sorted_p[3::4]` selects P=21,000 kPa — the very last (topmost) level**,
  which has no training neighbor on one side. This violates the spec's
  explicit "sandwiched on both sides" requirement and would have partially
  duplicated Split C's extrapolation test inside what's supposed to be the
  "moderate honesty" interior-holdout test. Fixed by restricting the
  selection to interior indices (`range(3, n_p-1, 4)`), verified by an
  explicit sandwiching assertion in the notebook.
- **The trilinear grid-interpolation baseline cannot be meaningfully
  evaluated on Split A.** `RegularGridInterpolator` requires training data
  that forms a complete rectilinear grid; Split A's random row-level 80/20
  split punches holes in that grid, and because the held-out points sit
  exactly *on* grid nodes, "interpolating" to them would require reading
  back the exact value that was removed — there's no genuine interpolation
  happening. This was recognized and documented rather than either silently
  producing a meaningless number or silently skipping the model without
  explanation.

**The single most valuable finding in the base notebook, methodologically,
was not anticipated by the context file**: the compact MLP's **raw**-target
variant (R²=0.850 on Split C) substantially outperformed its own
**log**-target variant (R²=0.342) — the opposite of the general expectation
(and opposite of the context's own prior-run number of ~0.74 for log-target
MLP). This was caught by actually reading the numbers rather than assuming
they'd match, and it falsified a specific claim in an early draft of the
conclusions cell ("log-target did not meaningfully hurt R² for any model") —
that claim was rewritten to state the precise, verified result instead of a
generic assumption. A follow-up seed-sensitivity test (`model_tests/test_mlp.ipynb`)
was added specifically to check whether this raw-beats-log gap is a robust
property of this architecture on this problem or a single lucky/unlucky
draw — see that notebook's output for the answer, since it materially
affects how much weight this finding should carry.

All Split-A sanity-check numbers matched the context's prior-validated
results almost exactly (Extra Trees R²=0.9992 vs. expected 0.9991±0.0001;
Random Forest R²=0.9987 vs. expected 0.9987±0.0002), which is good evidence
the overall pipeline (scaling-fit-on-train-only, seed handling, metric
computation) is implemented correctly, not just that individual models
happen to score well.

## 3. Physics-informed extensions (Phase 2) — what actually held up

The research phase found that "PINN" is the wrong keyword for this
literature (CHF has no governing PDE); the field's actual technique is
additive residual/hybrid correction learning on top of an empirical
correlation, and no published work tests genuine pressure-*range*
extrapolation the way this project's Split C does — every result here is
new evidence on that specific question, not a replication.

Four approaches were implemented and evaluated under the same three splits.
**Two real implementation bugs were caught during development, both worth
recording as cautionary examples:**

- The **Biasi (1967) correlation** was initially implemented with the wrong
  leading dimensional constants entirely omitted (predicting CHF values
  1,000–10,000× too small) and with `min()` instead of `max()` combining its
  two branches. Caught by sanity-checking predictions against known LUT
  values at representative conditions, then fixed by locating and verifying
  the correlation's exact textbook form (Todreas & Kazimi) via targeted web
  research rather than continuing to guess from memory.
- The **Zuber (1959) pool-boiling correlation** was initially hand-fit with
  approximate saturation-property curves for water; these curves crossed
  over near the critical pressure (liquid density dropping below vapor
  density), producing `NaN` and a wrong pressure-trend peak (~16,500 kPa
  instead of the literature's ~2–7 MPa). Replaced with `CoolProp`'s real
  IAPWS-based water equation of state, which recovered a peak at ~6,700 kPa
  — matching the literature without any manual tuning, and directly
  confirming this correlation is a legitimate physics reference rather than
  an artifact of arbitrary curve-fitting. Separately, Biasi's `G^(1/6)`
  denominator diverges as G→0 (pool boiling) — exactly the regime Zuber
  covers and Biasi doesn't — so a `hybrid_reference_chf` dispatcher was
  added (Zuber at G=0, Biasi at G>0), mirroring how the LUT's own authors
  built their skeleton table.
- The **collocation-based physics-penalty MLP** (the closest thing to a
  genuine PINN in this project) initially differentiated its physics
  penalties with respect to *raw, unscaled* pressure (range: thousands of
  kPa). This made the pressure-derivative numerically negligible (~1e-5)
  purely from unit scale, silently making that entire penalty term inert —
  the printed "0.0000" penalty during training looked like "the network
  already satisfies the physics constraint," when it was actually "the
  gradient signal is too small to matter at 4 decimal places." Caught by
  printing raw (pre-`relu`) gradient statistics rather than trusting the
  training log, and fixed by differentiating with respect to the scaled
  coordinates directly. A second, unrelated bug in the same component
  called `CoolProp` fresh for every one of 512 collocation points on every
  one of up to 800 epochs (~400,000+ expensive calls), which hung rather
  than erroring — fixed by precomputing a Zuber-derivative-sign lookup
  table once and interpolating it cheaply inside the training loop.
- The **pressure-gated blend**'s first version defaulted its "smooth"
  component to a log-target MLP, inheriting that variant's much weaker
  Split C performance (R²=0.342, per the Phase-1 finding above) and scoring
  only R²=0.345 overall despite the gate correctly routing almost all
  Split-C queries to the smooth model. Switching to the empirically
  better raw-target MLP raised this to **R²=0.855 — the best Split C result
  found anywhere in this project**, edging out the plain compact MLP
  (0.850) and the grid interpolator (0.841). This is the clearest
  illustration in the whole project of a general principle: **a gate or
  blend inherits both the strengths and the weaknesses of whichever
  component it trusts in a given regime — it is not a free lunch, and its
  quality is entirely contingent on the quality of its worst-trusted
  component.**

**Two approaches failed outright and the failure mode is itself a genuine
finding, not just a null result**: physics-basis-feature polynomial
regression produced catastrophic negative R² (down to -57,000) on Split C
because a `log(1 - P/P_crit)`-style feature, well-behaved inside the
training range, grows large outside it and gets amplified by a degree-2
expansion and Ridge coefficients fit only to the in-range curvature.
Additive residual learning on the Biasi/Zuber hybrid correlation also failed
(R² down to -4.3) because the ML correction, fit to cancel the
correlation's in-range error, does not generalize to the differently-shaped
error the correlation makes out-of-range. Both are documented in the
notebook's conclusions with the specific mechanism, not just the number.

## 4. Overall ranking — "which performs best," stated once, plainly, WITH the seed-sensitivity caveat that changes how to read it

Best Split C (the honest test) R² found across **both** notebooks, at the
seeds each was built/tuned with, in order:

1. **Pressure-gated blend (ExtraTrees + raw MLP), R²=0.855** — the best single result observed, and also ties for best on Splits A/B (0.999).
2. Compact MLP, raw-target, R²=0.850 (Phase 1) — nearly as good, far simpler (no gate hyperparameter).
3. Collocation physics-penalty MLP, R²=0.848 (Phase 2) — required a sensitive, Split-C-informed lambda sweep to get here; did not clearly beat #2.
4. Trilinear grid interpolation, R²=0.841 (raw target) (Phase 1) — the one result in this top group that is NOT seed-dependent (deterministic interpolator, no stochastic training).
5. Degree-2 log-polynomial, R²=0.755 (Phase 1) — also deterministic (Ridge has a closed-form solution).
6. GPR (log-target, 2000-point subsample), R²=0.675 (Phase 1).
7. Tree ensembles (RF/ET/XGBoost/LightGBM) and kNN, R²≈0.41–0.45 — the expected structural collapse.
8. Linear regression, physics-basis-feature regression, and additive residual learning on Biasi/Zuber — all **negative** R², the expected/demonstrated failure of models with no mechanism to extrapolate safely, or (for the physics-feature and residual approaches) a feature/correction that itself amplifies badly out of range.

**Critical caveat that materially changes items 1-3 above**: `model_tests/test_mlp.ipynb`'s
seed-sensitivity sweep (run after the ranking above was first written) found
that raw-target MLP's Split C R² swings from 0.081 to 0.726 across 5 seeds
(mean 0.518, std 0.225), while log-target is both higher on average (mean
0.613) and far more stable (std 0.079). **The single seed=42 run that
produced items 1-3's numbers landed on the favorable tail of an unstable
distribution for raw-target.** This does not invalidate the *mechanism*
behind the gated blend or the physics-penalty approach, but it does mean
items 1-3 should be read as "best observed at a favorable seed," not
"the expected performance of this technique." Items 4-8, none of which
depend on neural-network stochasticity, are unaffected by this caveat and
can be read at face value. **The single most defensible, seed-independent
statement this project can make is: "grid interpolation (deterministic,
R²=0.841) and degree-2 log-polynomial regression (deterministic, R²=0.755)
are the most reliable extrapolation baselines found; MLP-based approaches
can reach higher R² (~0.85) at a favorable seed, but require multi-seed
validation before that number can be trusted as representative."**

On Splits A/B (interpolation), essentially every model except plain Linear
regression exceeds R²=0.97, confirming these splits are the "easy" tests
the project context said they'd be — and this ranking is NOT affected by
the seed caveat, since every model's in-range fit is uniformly excellent
regardless of which target variant or seed is used.

## 5. What I'd do next if this thesis continues

- **Re-run Approaches 3 and 4 from `CHF_Physics_Informed_Extensions.ipynb`
  across multiple seeds and report mean±std**, exactly as the base
  notebook's Split A protocol already does for its 5-seed sweep — this is
  the single highest-priority follow-up, since it directly determines
  whether this project's "best result" (R²=0.855) is a real finding or an
  artifact of one favorable random initialization.
- The two exploratory Phase-2 results that used Split-C feedback during
  development (the physics-penalty MLP's lambda, and the gated blend's
  raw-vs-log choice) are disclosed as such in-notebook, but a rigorous
  write-up should present them as "best achievable with light tuning," not
  as unbiased extrapolation-generalization estimates. A nested/held-out
  validation pressure level (e.g., tune on P=17,000-18,000, report on
  19,000-21,000 only) would convert this from a disclosed caveat into a
  genuinely clean result.
- GP with a physics-informed (correlation) mean function and symbolic
  regression (PySR) were both ranked as promising in the literature review
  but not implemented, due to added dependency cost (GPyTorch/GPflow, a
  Julia toolchain) rather than a negative assessment — both remain
  legitimate open directions.
- An LQR-focused EDA/error-analysis pass (Section 1 above) would strengthen
  the thesis's physical grounding beyond what was needed for this specific
  modeling deliverable.
