# Independent Adversarial Audit of the Critical Heat Flux Project

**Audit date:** 14 September 2026
**Scope:** complete forensic re-examination, feature-space reconstruction, literature novelty search, and attempted refutation of the project's central claims.
**Standard applied:** would this survive a hostile referee at *Nuclear Engineering and Design*?

---

## EXECUTIVE SUMMARY — READ THIS FIRST

This audit was conducted under instruction to treat the existing pipeline as a candidate prototype rather than a finished result, and to actively try to disprove it. Six findings materially change the project.

| # | Finding | Consequence |
|---|---|---|
| 1 | **A 2026 paper independently published three of our four core claims.** | The novelty claim must be rewritten. This is the most important finding in the audit. |
| 2 | **The look-up-table backbone was missing its published diameter correction.** A real implementation bug. | Fixed. Lifts that arm on KAERI from R² 0.768 → 0.894 and ±10% from 22.0% → 49.2%. |
| 3 | **The project attributed its success to the wrong component.** | The *anchor* is significant (p = 0.016, median ΔR² +2.10). The *bound* is not (p = 0.237, median ΔR² +0.0001). |
| 4 | **The bound is nonetheless justified — as tail insurance.** | Improves worst-case held-out R² in **8 of 8** independent feature configurations. |
| 5 | **The feature set was never justified; now it is.** Eight configurations tested, including Reynolds, Prandtl, Bond, Froude. | The shipped 5 features win. Every addition hurts the anchored model. |
| 6 | **The out-of-distribution coverage failure is largely fixable.** | Cross-dataset conformal calibration moves median coverage from 0.564 to 0.885 against a 0.90 target. **This is now the project's strongest novel contribution.** |

**Net assessment.** The pipeline is sound and survives adversarial testing, but it is **less novel and differently justified** than previously claimed. One real bug was found and fixed. The strongest remaining contribution is not the architecture — it is the out-of-distribution uncertainty result and the multi-fluid transfer analysis.

---

# PART 1 — FORENSIC AUDIT: WHAT WAS ACTUALLY BUILT

## 1.1 Reconstructed pipeline logic

| Layer | Operation | Justified? |
|---|---|---|
| 0 | CoolProp saturation properties → 5 dimensionless groups | **Now yes** — was asserted, now tested against 7 alternatives (§3) |
| 1 | Select among Katto–Ohno / Biasi / 2006 LUT by median abs log error on training rows | Yes, but see §2.2 — one arm was implemented incorrectly |
| 2 | ExtraTrees(300) on `log(CHF / backbone)` | **Yes — and this is the component that matters** (§4) |
| 3 | `trust = 1 − max(mahalanobis, corr_spread, fluid_new)`; scales a clip from 1.5× to 10× | Partially — see §4.3, the justification changes |
| 4 | Split conformal, 20% calibration slice | **No — fails out-of-distribution.** Replaced in §6 |

## 1.2 Verified inventory

- **Corpus:** 28,582 flow-boiling rows; 8 datasets; 3 fluids (water, R-123, R-134a); 5 geometries (vertical tube, horizontal tube, annulus, flat plate, helical coil); uniform and non-uniform axial heating.
- **Reference object:** Groeneveld 2006 look-up table, 11,088 non-zero entries of 11,592.
- **Prior benchmark:** 961 runs, 22 approaches (804 machine-learning runs over 13 families + 157 physics/hybrid runs over 9 arms).
- **Leakage controls:** hard whitelist at load time, 4 documented leaks banned.
- **Splits:** 80/20, 70/30, grouped, leave-one-dataset-out.

## 1.3 Items confirmed correct and left unchanged

Checked and found sound — no change made:

1. The **energy-balance circularity demonstration** (R² = 0.99977, 100% within ±10%, zero fitted parameters). Correctly derived, correctly attributed to Hall & Mudawar (2000b) Eq. 8.
2. The **leakage whitelist**. Power/perimeter/length reconstruction at R² = 0.9999934 is real; the bans are correct.
3. **Katto–Ohno equations 8–18.** Verified line by line against NASA NTRS 20230009827, extracted directly. Coefficients 0.1, 0.098, 0.0384, 0.234, 1.043, 5/6, 0.0124, 1.12, 1.52, 0.0031, 0.28 and the ρg/ρf = 0.15 regime gate **all match**.
4. **Biasi's per-diameter exponent loop.** The scalar-branch fix is correct and necessary.
5. **The duplicate-aware splitting** and the `GROUP_MEANING` labelling.
6. **The refusal behaviour** (pool boiling, supercritical, unknown fluid, bad geometry).

---

# PART 2 — BUGS FOUND

## 2.1 The look-up table was missing its diameter correction [FIXED]

**Severity: high. This is a genuine implementation error that degraded every result using that arm.**

The 2006 look-up table is normalised to an **8 mm** tube. `phys_lut()` interpolated the table and returned the value **directly**, with no diameter correction applied.

What it was actually being asked about:

| Dataset | Diameter range | Fraction actually at 8 mm |
|---|---|---|
| D1 NRC | 2.00 – 16.00 mm | 17.3% |
| D2 Zhao | 1.00 – 37.50 mm | **0.3%** |
| D3 KAERI uniform | 5.44 – 28.30 mm | **0.0%** |
| D6 Hardik helical | 6.00 – 10.00 mm | 25.6% |

**The published correction**, from the table's own documentation (Groeneveld et al., *Nucl. Eng. Des.* 237:1909–1922, §3), extracted verbatim from the local PDF:

> "The effect of tube diameter on CHF is accounted for using the diameter correction factor: CHF_D / CHF_{D=8mm} = (D/8)^{−1/2} for the range 3 < D < 25 mm. **Outside this range the diameter effect appears to be absent** (Wong, 1994)."

So the factor is `(8/D)^0.5`, and **D must be clamped to [3, 25] mm** — the authors explicitly say the correction does not apply outside that band.

**Measured effect of the fix:**

| Dataset | R² shipped | R² unclamped | **R² clamped (correct)** | ±10% shipped | **±10% clamped** |
|---|---|---|---|---|---|
| D1 NRC | 0.8991 | 0.9415 | **0.9411** | 37.9% | **44.8%** |
| D2 Zhao | 0.5221 | 0.4526 | **0.5821** | 29.4% | **45.4%** |
| D3 KAERI uniform | 0.7682 | 0.8920 | **0.8944** | 22.0% | **49.2%** |
| D6 Hardik helical | −0.3845 | −0.0827 | **−0.0827** | 14.7% | **16.0%** |

Every dataset improves. Note that the **clamp is load-bearing**: on Zhao, applying the correction without it makes things *worse* (0.522 → 0.453), because Zhao spans 1–37.5 mm and the naive correction extrapolates a factor its own authors disclaim. With the clamp: 0.582.

**This is not a tuned result.** The exponent, the reference diameter and the validity band all come from the source document. Nothing was fitted.

**Implemented** in `physics_arms.py` as `lut_diameter_factor()`, tagged `[V]` with the quotation in the docstring.

## 2.2 The `pure_physics` baseline was a confounded comparison [ASSESSED — no correction needed]

`run_pipeline.py` hardcodes `p_phys = te["phys_katto"]`, while the pipeline *selects* its backbone. So "pipeline beats physics" could have been measuring backbone selection rather than the machine-learning correction.

**Tested.** In leave-one-dataset-out, `backbone_alone` and `katto_alone` are **numerically identical** (median R² 0.3355, worst −0.3443) — the selector always picks Katto there, because the look-up-table and Biasi arms are undefined on most rows of a merged multi-fluid corpus and fail the 90%-validity requirement.

**Verdict: the headline leave-one-dataset-out comparison was NOT confounded.** The confound exists only in the per-dataset random splits, where Zhao does select the look-up table (§2.3).

## 2.3 The Zhao backbone consumes outlet quality [DISCLOSED — not fixed]

On the per-dataset 80/20 split, D2 Zhao selects `phys_lut`, which is indexed on **local vapour quality** — the circular variable the whole project exists to eliminate.

Re-measured with the **fixed** look-up table, 5 seeds:

| D2 Zhao, 80/20 | Uses quality? | R² | ±10% |
|---|---|---|---|
| Backbone freely selected (`phys_lut`) | **Yes** | 0.8658 | 67.9% |
| Backbone forced to `phys_katto` | No | **0.5931** | **40.8%** |

**The honest Zhao number is 0.593, not 0.866.** The gap is 0.27 in R² — larger than before the fix, because fixing the look-up table made the circular arm *better*.

**Recommendation:** any published per-dataset table must either force `backbone='phys_katto'` or report both rows with the quality dependence stated. Reporting 0.866 alone would be indefensible.

## 2.4 A process error of my own

The earlier background launches used `nohup … &` inside an already-backgrounded call, which produced a "task completed" notification while the Python was still running. Same mistake the project record already documents. Noted so it is not mistaken for a finished run.

---

# PART 3 — FEATURE SPACE REBUILT FROM FIRST PRINCIPLES

## 3.1 The gap identified

The shipped feature set is five dimensionless groups built from **density, surface tension, and latent heat**:

`Katto_We_inv`, `rho_ratio_gl`, `LtoD_dim`, `dh_sub_star`, `P_red`

It contains **no viscosity, no thermal conductivity, no specific heat, and no gravity**. Therefore:

- **no Reynolds number** — no measure of turbulence
- **no Prandtl number** — no measure of the momentum-to-thermal boundary-layer ratio
- **no Bond or confinement number** — no measure of buoyancy versus surface tension
- **no Froude number** — no measure of inertia versus gravity

For a *convective* boiling problem that is a conspicuous omission, and it had never been tested. All required properties are available from CoolProp for all three fluids (verified).

## 3.2 The groups added

| Group | Formula | Physical meaning |
|---|---|---|
| `Re_l` | `G·D/μ_l` | Inertia vs viscous force — turbulence level |
| `Pr_l` | `μ_l·c_p,l/k_l` | Momentum vs thermal diffusivity |
| `We_D` | `G²·D/(ρ_l·σ)` | Weber number on **diameter** (existing one is on **length**) |
| `Bond` | `g·(ρ_l−ρ_g)·D²/σ` | Buoyancy vs surface tension — the confinement scale |
| `Fr` | `G²/(ρ_l²·g·D)` | Inertia vs gravity |
| `mu_ratio` | `μ_g/μ_l` | Viscosity ratio |

All finite on 100% of rows.

## 3.3 Result — eight configurations, leave-one-dataset-out, 3 seeds

| Feature set | n | pooled ML (no anchor) | **Pipeline** | Pipeline worst |
|---|---|---|---|---|
| **F0 base 5 (shipped)** | 5 | −2.061 | **+0.7084** | −0.4357 |
| F6 base + Re | 6 | −1.781 | +0.6944 | −0.4654 |
| F1 base + Re, Pr | 7 | +0.255 | +0.6812 | −0.5254 |
| F2 base + We_D, Bond, Fr | 8 | −0.653 | +0.6718 | −0.4035 |
| F7 base + Bond | 6 | −0.941 | +0.6701 | −0.3693 |
| F3 all 11 | 11 | **+0.559** | +0.6630 | −0.5005 |
| F4 base − L/D | 4 | −18.269 | +0.5006 | −0.3846 |
| F5 no geometry | 3 | −37.889 | +0.4563 | −0.4243 |

## 3.4 Interpretation — a genuinely interesting result

Read the two middle columns against each other. There is a clean, monotone, **opposite** trend:

- **The unanchored pooled model improves dramatically with more physics features.** Median R² goes from −2.06 (5 features) to +0.56 (11 features); its worst case from −755.6 to −38.4. Reynolds and Prandtl alone lift it from −2.06 to +0.25.
- **The anchored pipeline gets monotonically worse.** 0.708 → 0.663.

**[OUR INTERPRETATION]** The transport properties carry real, usable physics — that is why the unanchored model needs them so badly. But **the physics backbone already encodes that information**, because Katto–Ohno is built from the same property set. Supplying it again to the correction model is redundant, and redundant features in a 28,582-row problem buy variance, not signal.

This yields a sharper statement of what anchoring is *for*:

> **The value of a physics backbone is precisely that it supplies, for free and exactly, the information a data-driven model would otherwise need many more features and far more data to approximate.**

That is a defensible, quantified claim and it is supported by an 8-point monotone trend on an independent axis.

## 3.5 The deferred "clean feature re-run" — resolved

The project had deferred testing whether `LtoD_dim` carries residual leakage from the energy balance (which contains both L and D).

**Result: it does not, and it is load-bearing.** Removing it costs **0.708 → 0.501**, and removing geometry entirely costs **0.708 → 0.456**. The unanchored model collapses to −18.3 and −37.9 respectively.

**Verdict: keep `LtoD_dim`.** The deferred concern is closed in favour of the existing design.

---

# PART 4 — THE PROJECT ATTRIBUTED ITS SUCCESS TO THE WRONG COMPONENT

This is the most scientifically significant correction in the audit.

## 4.1 What was claimed

The project's narrative, stated in the pipeline source and in every report, is:

> *"Routing is a switch and every switch error costs a factor of 1000. Bounding is a dial with a measured monotonic trade-off."* — and therefore the **trust-scaled bound** is the core design decision, evidenced by pooled machine learning reaching R² −868.6 while the pipeline's worst case is −0.429.

The problem with that evidence is that it compares the **full pipeline** against a model with **no anchor at all**, and then attributes the entire difference to the bound. Those are two different changes bundled together.

## 4.2 The decomposition

Four arms were evaluated on identical leave-one-dataset-out folds, F0 features, 3 seeds:

| Held-out | Katto alone | pooled ML | **anchored, unbounded** | **anchored + bound** |
|---|---|---|---|---|
| D1 NRC | 0.9261 | 0.3832 | 0.9434 | 0.9443 |
| D2 Zhao | −0.3443 | −1.5556 | −0.6226 | −0.4357 |
| D3 KAERI uniform | 0.6200 | 0.8169 | 0.7332 | 0.7331 |
| D4 KAERI non-uniform | 0.7970 | 0.8403 | 0.9667 | 0.9668 |
| D5 Helical R-123 | 0.0510 | −369.65 | 0.4351 | 0.4351 |
| D6 Hardik helical | −0.1613 | −2.5666 | 0.6938 | 0.6838 |
| D7 Hardik straight | 0.6362 | −755.58 | 0.7935 | 0.7935 |
| D8 Pioro R-134a | −0.0974 | −598.72 | −0.2281 | −0.0606 |
| **Median** | **0.3355** | **−2.061** | **0.7135** | **0.7084** |
| **Worst** | −0.3443 | −755.58 | −0.6226 | **−0.4357** |

## 4.3 Paired significance tests (Wilcoxon signed-rank, n = 8)

| Comparison | Wins | p | Median ΔR² |
|---|---|---|---|
| **Anchoring alone vs no anchor** | 7/8 | **0.0156** | **+2.0967** |
| Pipeline vs unanchored pooled ML | 7/8 | 0.0156 | +2.1851 |
| Pipeline vs pure physics | 7/8 | **0.0391** | +0.1352 |
| **Bounded vs unbounded** | 5/8 | **0.2367** | **+0.0001** |

## 4.4 What this means

**The anchor does the work.** Changing the target from `log(CHF)` to `log(CHF / backbone)` moves the median held-out R² by **+2.10** and is significant at p = 0.016. This single change takes the method from unusable (−2.06, worst −755.6) to useful (+0.71).

**The bound contributes nothing to median accuracy.** Median ΔR² = **+0.0001**, winning 5 of 8, p = 0.237. It is indistinguishable from no bound at all on central performance.

**Two further corrections to the record:**

1. **"Beats pure physics on all eight datasets" is no longer true.** It is **7 of 8**. Physics beats the pipeline on D2 Zhao (−0.3443 vs −0.4357). The earlier 8/8 came from a run predating the Zhao diameter fix, in which the fabricated 8 mm geometry made physics look worse than it is. The corrected comparison is 7/8 at p = 0.039 — still significant, but it is no longer a clean sweep and must not be reported as one.
2. **The −868.6 figure belongs to the anchor, not the bound.** It is the pooled-ML worst case. Attributing it to bounding overstates what bounding does by roughly three orders of magnitude.

## 4.5 The bound is still justified — but on different grounds

Having demoted the bound, the audit then tested whether it earns its place at all. It does, on **tail risk**, and the evidence is stronger than the median evidence was weak.

Across the **eight independent feature configurations** of §3.3 — an axis entirely separate from the dataset axis:

| Criterion | Result |
|---|---|
| Bound improves the **median** | 3 of 8 configurations — a wash |
| Bound improves the **worst case** | **8 of 8 configurations** |
| Worst case over all configs, bounded | **−0.5254** |
| Worst case over all configs, unbounded | **−0.7299** |

Eight of eight on an independent axis is a binomial p ≈ 0.008 against a fair-coin null.

**The defensible claim, which replaces the previous one:**

> **The bound is not an accuracy mechanism; it is a tail-risk mechanism. It leaves median held-out accuracy unchanged (3 of 8 configurations, median ΔR² +0.0001, p = 0.24) while improving the worst-case held-out R² in 8 of 8 independent feature configurations. For a safety-adjacent application, buying a bounded tail at zero median cost is the correct trade — but it must be argued on tail risk, not on accuracy.**

This is a weaker claim than the project previously made. It is also one that will survive a referee.

---

# PART 5 — LITERATURE SEARCH: THE NOVELTY CLAIM DOES NOT SURVIVE

The audit instruction was explicit: *actively search for papers that could invalidate our claims.* One was found, and it is close enough that it must change how this work is positioned.

## 5.1 The competing paper

> **"Campaign-disjoint validation of machine-learned critical heat flux models: leakage quantification and physics-anchored residual transfer."**
> ScienceDirect, article S2666546826002156. Published approximately August 2026.
> Companion preprint: *"Campaign-resolved evidence for adaptive critical-heat-flux surrogates under experimental shift,"* Research Square rs-10789551.

**Access caveat, stated plainly:** the full texts returned **HTTP 403** on both ScienceDirect and Research Square. Everything below is from abstract-level sources. **The full text must be obtained before any submission.** In particular, whether they bound the correction is **unverified**.

## 5.2 What they establish

| Their contribution | Our corresponding claim | Status |
|---|---|---|
| Derive the outlet-quality energy-balance identity and **quantify the apparent gain from that leakage** | Our Claim 1 — the circularity demonstration | **PRE-EMPTED** |
| **Campaign-disjoint validation** on 24,579 records from 60 campaigns — *the same NRC database as our D1* | Our Claim 2 — grouped/leave-one-source-out validation | **PRE-EMPTED** |
| Hybrids that learn the **multiplicative logarithmic residual** of a correlation (Bowring; we use Katto–Ohno) | Our Claim 3 — the anchored `log(CHF/backbone)` target | **PRE-EMPTED** |
| Companion paper: when evidence transfers, **when uncertainty should be widened**, sparse adaptation | Our Claim 4 — trust-scaled interval widening | **PARTIALLY PRE-EMPTED** |

Their reported headline: the direct learner degrades from **4.69% random-split error to 11.91% campaign-disjoint**, a 2.54× increase, while non-trainable correlations stay near unity.

**That is the same phenomenon we measured, on the same database, with the same interpretation.**

## 5.3 Other relevant prior work found

- **Furlong, Zhao, Salko & Wu** — *Physics-Based Hybrid Machine Learning for CHF Prediction with Uncertainty Quantification* (arXiv 2502.19357). Anchors to **Biasi and Bowring**; uncertainty via deep ensembles, Bayesian neural networks and deep Gaussian processes. All hybrids beat pure machine learning. No evidence of a bounded correction.
- **Cazzola, Ghione, Sargentini, Nespoulous & Finotello** — *Learning Complex Physical Regimes via Coverage-oriented Uncertainty Quantification: An application to the Critical Heat Flux* (arXiv 2602.21701, February 2026). Coverage-oriented uncertainty quantification on the NRC dataset. **Does not appear to address distribution-shift coverage failure** — which leaves our §6 result standing.
- Multiple look-up-table-guided grey-box and transfer-learning papers, already known to the project.

## 5.4 Honest revised novelty assessment

**What we must NOT claim:**

1. ❌ That the energy-balance circularity of outlet quality is our finding.
2. ❌ That campaign/source-disjoint validation of CHF machine learning is our idea.
3. ❌ That anchoring a learner to a correlation via a multiplicative log-residual is our architecture.
4. ❌ That the published literature systematically over-reports — we still have not audited it, **and now at least one paper has already made the correction we would be recommending.**

**What plausibly survives — each flagged for verification:**

| Candidate | Strength | Why it may survive |
|---|---|---|
| **Out-of-distribution conformal calibration** (§6) | **Strongest** | The UQ papers found do not address coverage collapse under dataset shift. Our cross-dataset calibration moves median coverage 0.564 → 0.885. |
| **Multi-fluid, multi-geometry corpus** | Moderate | Prior work is water-only, uniformly heated vertical tubes. We span 3 fluids, 5 geometries, uniform and non-uniform heating. Leave-one-*dataset*-out across fluids and geometries is a strictly harder shift than leave-one-campaign-out inside one homogeneous database. |
| **The transfer/detection tension** | Moderate–strong | Requires ≥2 fluids to state, so it is structurally outside a single-fluid study's reach. |
| **The anchor/bound decomposition** (§4) | Moderate | A methodological result: separating which component of a hybrid actually generalises, with significance tests on two independent axes. |
| Trust-scaled bound | **Weak** | Our own data says it does not improve median accuracy. Defensible only as tail control, and we cannot confirm the competing paper does not do something equivalent. |

**Bottom line.** This is no longer a paper that can be sold as *"we discovered that CHF machine learning is validated wrongly."* Someone else published that. It can be sold as *"we extend campaign-disjoint analysis to a multi-fluid, multi-geometry corpus, decompose which part of physics anchoring actually generalises, and fix the uncertainty-calibration failure that anchoring alone does not solve."* That is a narrower but genuine contribution.

---

# PART 6 — THE OUT-OF-DISTRIBUTION COVERAGE FAILURE IS LARGELY FIXABLE

**This is the audit's most valuable positive result.**

## 6.1 The problem restated

The shipped pipeline calibrates its conformal interval on a 20% slice held out of the **training** data. That slice is exchangeable with the training rows, so the resulting quantile answers *"how wrong am I on rows like the ones I trained on?"* — which is the wrong question when the deployment case is a new laboratory.

Measured coverage against a 90% target on a genuinely unseen dataset ranged from **12.4% to 99.5%**. The conformal guarantee is mathematically void under distribution shift, and this is exactly the shift that matters.

## 6.2 The proposed fix — cross-dataset (jackknife-over-datasets) conformal

Do not ask how wrong the model is on familiar rows. Ask **how wrong it has historically been on entire unseen datasets**, and use that distribution.

To set the interval for held-out dataset *d*: pool the residuals the pipeline produced on **every other dataset when that dataset was itself held out**, and take their (1−α) quantile. Dataset *d*'s own residuals are never used.

This swaps an exchangeability assumption that is **false** (a new dataset resembles the training rows) for one that is weaker and much closer to true (**new datasets resemble other new datasets**).

## 6.3 Closing the leakage hole in the fix itself

A referee would immediately object: in the fast version, the models producing those residuals were trained on data that *included d*. So *d* influenced the sizing of its own interval.

The **strict** protocol removes this: for every pair (*d*, *e*), train on all datasets except **both**, predict *e*, keep those residuals. By symmetry this is 28 fits, not 56. Dataset *d* never touches any model whose residuals size its interval.

## 6.4 Results

| Held-out dataset | Shipped | **Strict cross-dataset** | Strict + trust widening |
|---|---|---|---|
| D1 NRC | 0.995 | 0.996 | 0.999 |
| D2 Zhao | 0.137 | 0.203 | 0.437 |
| D3 KAERI uniform | 0.648 | **0.911** | 0.937 |
| D4 KAERI non-uniform | 0.528 | **0.977** | 0.983 |
| D5 Helical R-123 | 0.424 | 0.696 | 0.992 |
| D6 Hardik helical | 0.712 | 0.859 | 0.981 |
| D7 Hardik straight | 0.600 | **0.927** | 0.964 |
| D8 Pioro R-134a | 0.354 | 0.470 | 0.754 |
| **Median** | **0.564** | **0.885** | 0.972 |
| **Mean \|coverage − 0.90\|** | **0.374** | **0.198** | **0.133** |

**The strict, leakage-free result (0.885) is identical to the fast approximation (0.885)** — the leakage concern was real in principle and immaterial in practice, and now that is measured rather than argued.

## 6.5 Honest assessment of the fix

**What it achieves.** Median coverage moves from 0.564 to **0.885** against a 0.90 target — essentially nominal. Mean absolute deviation from target roughly halves. Coverage improves on **8 of 8** datasets. Three datasets that were badly under-covered (D3 0.648, D4 0.528, D7 0.600) are brought to 0.911, 0.977 and 0.927.

**What it costs.** The intervals are roughly **4× wider**. That is not a defect — it is the honest width. The model genuinely is that uncertain on an unseen dataset, and the old intervals were lying about it.

**Where it still fails.** D2 Zhao (0.203) and D8 Pioro (0.470) remain badly under-covered. These are the same two datasets that are unpredictable in point-estimate terms. **No calibration method can rescue a prediction that is simply wrong**; a correctly-sized interval around a badly biased prediction still misses.

**Why this may be a genuine contribution.** The CHF uncertainty-quantification papers located in the literature search (Furlong et al. 2025; Cazzola et al. 2026) do not appear to address coverage collapse under dataset shift. **[OPEN QUESTION — requires full-text verification.]**

---

# PART 7 — CROSS-FLUID TRANSFER, AND A CLAIM I HAVE TO WITHDRAW

## 7.1 Leave-one-fluid-out

Every row of a working fluid held out, across every geometry it appears in. This is a strictly harder shift than leave-one-campaign-out inside one homogeneous database, and it is a test a single-fluid study structurally cannot run.

| Held-out fluid | n test | n train | Physics | pooled ML | **Pipeline** | Pipeline ±10% |
|---|---|---|---|---|---|---|
| R-123 | 312 | 28,270 | 0.1456 | **−1245.93** | **0.4637** | 16.5% |
| R-134a | 268 | 28,314 | −0.0974 | **−598.72** | **−0.0606** | 15.9% |

**Interpretation.** On R-123 the pipeline turns 0.146 into 0.464 — anchoring plus correction transfers usefully to an unseen fluid. On R-134a it manages only −0.061 versus physics's −0.097; barely better than the correlation, but it does not collapse, whereas the unanchored model reaches −598.7.

The headline for cross-fluid transfer: **the anchored method is the difference between "mediocre" and "catastrophic"** — three orders of magnitude — but it does not make an unseen fluid *well* predicted.

## 7.2 CORRECTION: the "transfer/detection tension" claim was overstated

I previously described this as the project's strongest research gap, and stated it as:

> *"A novelty gate cannot detect an unseen fluid, because the dimensionless representation that makes transfer possible is exactly what erases fluid identity. Not one of Pioro's 268 R-134a rows was flagged."*

**I tested that claim directly, and as a general statement it is false.**

Decomposing the trust score with each fluid held out:

| Held-out fluid | Full trust (all 3 signals) | % flagged | **Distance-only trust** | **% flagged by distance alone** |
|---|---|---|---|---|
| R-123 | 0.000 | 100.0% | 0.167 | **82.4%** |
| R-134a | 0.000 | 100.0% | 0.195 | **100.0%** |

**Mahalanobis distance in dimensionless space flags 82–100% of unseen-fluid rows.** The representation does *not* make unseen fluids undetectable.

**Where the original number came from.** `router_results.csv` genuinely records `frac_flagged_unseen = 0.000000` for D8. But that is the **router's own gate** — a sigmoid on raw Mahalanobis distance with a threshold set at a high percentile of training distance. The pipeline's gate normalises the same distance between the 50th and 95th percentiles of training distance, and R-134a rows sit above the 95th, so they flag.

**So the router's failure to detect was a threshold-calibration failure of one specific gate, not a structural impossibility.** The honest version of the finding is much narrower:

> *A distance gate's threshold must be calibrated against the training distance distribution, not set as an absolute. The router's was not, and it consequently flagged none of the unseen-fluid rows; the same distance measure, percentile-normalised, flags 82–100% of them.*

That is a useful engineering lesson. **It is not a deep research gap, and it must not be presented as one.**

## 7.3 Two further numerical corrections to the project record

1. **"The router hit R² −1276 on an unseen fluid."** This appears in `pipeline.py`'s own module docstring and has been repeated throughout the project's documents, including by me. **It is wrong.** In `ANSWER_Q3_router.txt`, −1276.44 on D8 belongs to `global_model_no_routing`. The routers scored **−42.46** (v1, dimensionless) and **−0.097** (v2, identity). The −1276 figure argues *for* anchoring, not *against* routing.

2. **"An oracle router would score 0.831; ours scored 0.099."** The oracle median of 0.831 checks out. **The 0.099 does not reproduce** from any stored column: router v1 median R² is 0.478, router v2 is 0.136; median ±10% is 0.132 for v1. **Treat 0.099 as unverified and do not quote it.**

The underlying conclusion — *routing did not earn its complexity, 5/8 against the global model and 2/8 against physics, verdict FAIL* — is unaffected and is correctly recorded in the results file. Only the supporting numbers were wrong.

---

# PART 8 — THREE HOSTILE REVIEWERS

## Reviewer 1 — CHF / thermal-hydraulics expert

**"Your look-up-table baseline was wrong, so every comparison against it is worthless."**
*Accepted and fixed.* The missing `(8/D)^0.5` correction is documented in §2.1 with the source quotation and the validity clamp. All affected results regenerated. The leave-one-dataset-out headline is unchanged, because Katto is the selected backbone there.

**"Your Katto constant is unverified, so your backbone is not reproducible."**
*Partially accepted.* Equations 8–18 are now verified line-by-line against NASA NTRS 20230009827. `C_Kc` is genuinely absent from that source. The sweep (§ below) shows physics-alone R² is flat across 0.75–1.5× but the **pipeline is sharply sensitive**, collapsing outside 0.75–1.25×, with the shipped value at the peak. That is corroboration, not verification. **Remains an open item.**

| C scale | Physics median R² | Pipeline LODO median |
|---|---|---|
| 0.50× | 0.233 | −0.013 |
| 0.75× | 0.315 | 0.672 |
| **1.00× (shipped)** | 0.336 | **0.705** |
| 1.10× | 0.337 | **0.727** |
| 1.25× | 0.335 | 0.598 |
| 1.50× | 0.333 | −0.040 |

The C_Kc branch is selected on 14.2% of corpus rows overall, but 63.5% on helical coils and 0% on Pioro — so sensitivity is concentrated in the helical datasets.

**"You apply a tube correlation to annuli and flat plates via hydraulic diameter. That is not justified."**
*Accepted, unresolved.* D2 Zhao mixes tubes, annuli and plates, and it is the single dataset where physics beats the pipeline and where both point accuracy (−0.429) and coverage (0.203) fail worst. **A per-geometry treatment was not implemented. This is the most likely explanation for Zhao and it is untested.**

**"Your best per-dataset number for Zhao uses outlet quality through the look-up-table backbone."**
*Accepted.* Disclosed in §2.3. Honest value 0.593, not 0.866.

## Reviewer 2 — Machine-learning expert

**"Random splits on a compilation database are meaningless."**
*Agreed — that is the project's own thesis.* Four split types reported; the honest ones are grouped and leave-one-dataset-out.

**"Your uncertainty is not calibrated out-of-distribution."**
*Was true; now substantially addressed.* §6: median coverage 0.564 → 0.885, with the leakage-free strict protocol giving the same answer. Two datasets still fail.

**"Your significance claims rest on n = 8."**
*Accepted, and it is a real limitation.* Wilcoxon on 8 paired datasets has minimum achievable two-sided p ≈ 0.008. The anchoring result (p = 0.016) and the physics comparison (p = 0.039) are near the resolution limit of the test. **The 8-of-8 tail result in §4.5 is stronger precisely because it uses a second, independent axis (feature configurations).**

**"You never tuned hyperparameters."**
*Accepted, and deliberately not pursued.* Eleven structural variants were already rejected, and an eight-point feature sweep shows the method is limited by data coverage and source confounding, not model capacity. Tuning ExtraTrees is unlikely to move a −0.43 on Zhao. **This is a judgement, not a measurement.**

**"You chose ExtraTrees arbitrarily."**
*Partially accepted.* It was selected from a 961-run standalone grid but never re-compared **inside** the anchored architecture. Given that the anchor supplies the physics and the correction model only learns a small residual, model choice is likely second-order — but **untested**.

**"Your feature set was asserted, not justified."**
*Was true; now resolved.* §3: eight configurations, including all the standard convective groups. The shipped set wins.

## Reviewer 3 — Journal editor

**"What is actually new? A 2026 paper already published your leakage identity, your campaign-disjoint validation, and your anchored log-residual."**
*Accepted. This is the hardest question and the honest answer is that the original framing is dead.* §5. What remains: the multi-fluid/multi-geometry extension, the anchor/bound decomposition, and the out-of-distribution conformal result.

**"Your headline claim shrank during the audit."**
*Accepted.* "Beats physics on all eight datasets" is now **7 of 8**. "The bound saves us from −868" is now "the **anchor** saves us from −755; the bound improves the worst case in 8 of 8 configurations at no median cost." "A novelty gate cannot detect an unseen fluid" is **withdrawn**.

**"Is this incremental?"**
*Partly yes.* As an architecture paper it is incremental. As a **negative-results and methodology paper** it is not: an independently-found implementation bug in the international reference table's application, a decomposition showing which half of a hybrid actually generalises, a demonstration that adding textbook dimensionless groups *hurts* an anchored model, and a working fix for distribution-shift coverage — together with a documented record of eleven rejected variants, two failed routing designs, and three of the project's own numerical claims corrected. That is a publishable contribution in a venue that values reproducibility, but it is **not** a high-impact methods paper.

---

# PART 9 — FINAL VERDICT

## A. Current scientific level *before* this audit

A carefully-built, honestly-documented engineering prototype with an unusually good negative-results record — but with:
- one undetected implementation bug in a reference-standard correlation,
- an unjustified feature set,
- its central success attributed to the wrong component,
- a known, unaddressed uncertainty failure,
- at least three numerical claims that do not reproduce from its own stored results,
- and a novelty claim that had not been checked against the literature.

**Level: strong undergraduate project; not yet submittable.**

## B. Final scientific level *after* the audit

The same prototype, now with:
- the bug found and fixed from the source document,
- the feature set justified against seven alternatives,
- the success correctly attributed, with significance tests on two independent axes,
- the uncertainty failure largely repaired and verified leakage-free,
- the false claims withdrawn,
- and the novelty honestly repositioned against prior art.

**Level: a defensible methodology/negative-results contribution. Submittable to a reproducibility-valuing venue after the items in §D and §G are closed. Not a high-impact methods paper.**

## C. What I changed

1. **Fixed the look-up-table diameter correction** — `(8/D)^0.5`, clamped to 3–25 mm per the source. KAERI ±10%: 22.0% → 49.2%.
2. **Rebuilt and tested the feature space** — added Reynolds, Prandtl, Weber-on-diameter, Bond, Froude, viscosity ratio; 8 configurations evaluated; shipped set confirmed optimal.
3. **Decomposed the architecture** — anchor vs bound vs selection, with paired significance tests.
4. **Re-justified the bound** on tail risk (8/8 configurations) rather than accuracy.
5. **Fixed the out-of-distribution calibration** — cross-dataset conformal, median coverage 0.564 → 0.885, verified with a strict 28-fit leakage-free protocol.
6. **Ran leave-one-fluid-out** — a test prior single-fluid work cannot perform.
7. **Measured Katto C_Kc sensitivity** and verified Eqs. 8–18 against the NASA source.
8. **Found the competing 2026 paper** and rewrote the novelty assessment.
9. **Withdrew the transfer/detection claim** and corrected the −1276 and 0.099 figures.
10. **Resolved the deferred clean-feature re-run** — L/D is load-bearing, keep it.
11. **Quantified the Zhao backbone honesty gap** with the fixed table: 0.866 → 0.593.

## D. What I deliberately did NOT change, and why

| Left alone | Why |
|---|---|
| The ExtraTrees(300) correction model | Eleven structural variants and an eight-point feature sweep both indicate the bottleneck is data coverage, not model capacity. Tuning would be computational theatre. **This is a judgement.** |
| The three trust signals | An existing ablation shows they are near-interchangeable (0.736–0.753 alone). Since the bound's value is tail control and any reasonable signal delivers it, refining the gate optimises the wrong thing. |
| The `blend_free` option (left off) | Measured failure documented in-source (D7: 0.792 → −206.9). Deleting a failed experiment invites its repetition. |
| The Zhao backbone (not forced to Katto) | Forcing it would hide the issue in code rather than disclose it. The honest number is now published alongside. A publication should force it; the research record should show both. |
| Pool boiling | Explicitly out of scope by supervisor direction. |
| The 80/20 and 70/30 splits | Required by the supervisor, and they are the correct control for the argument — showing what the honest splits are being contrasted *against*. |

## E. Strongest demonstrated result

**Leave-one-dataset-out, 8 datasets, 5 seeds — entirely unseen experimental campaigns:**

| | Pure physics | **Pipeline** | Pooled ML |
|---|---|---|---|
| Median R² | 0.336 | **0.711** | −1.323 |
| Worst case | −0.344 | **−0.429** | **−755.6** |

with, decomposed: **anchoring** contributing median ΔR² **+2.10** (p = 0.016) and **bounding** contributing median ΔR² **+0.0001** (p = 0.24) but improving the worst case in **8 of 8** feature configurations.

And: **cross-dataset conformal calibration raising median out-of-distribution coverage from 0.564 to 0.885** against a 0.90 target, verified leakage-free.

## F. Genuine scientific contribution

1. **A working fix for conformal coverage collapse under dataset shift** in CHF prediction, verified with a leakage-free protocol. *Strongest candidate; requires full-text check of Cazzola et al. 2026.*
2. **The anchor/bound decomposition** — evidence that the generalisation benefit of physics-anchored hybrids comes almost entirely from the residual target, not from constraining the correction, with the constraint justified separately on tail risk.
3. **The redundancy result** — adding textbook convective dimensionless groups improves an unanchored model by >2.6 R² while *degrading* an anchored one, quantifying what the physics backbone is actually supplying.
4. **Multi-fluid, multi-geometry campaign-disjoint evaluation**, including leave-one-fluid-out.
5. **A corrected application of the 2006 look-up table**, with the magnitude of the error quantified.

**Explicitly NOT claimed:** the energy-balance circularity, campaign-disjoint validation, or anchored log-residual learning. All pre-empted.

## G. Remaining weaknesses — no sugar-coating

1. **No external validation set.** Every dataset informed development. Leave-one-dataset-out is a proxy, and the architecture was chosen partly in response to it.
2. **Zhao (−0.429) and Pioro (−0.065) remain unsolved**, in both point accuracy and coverage. The mixed-geometry hypothesis for Zhao is **untested**.
3. **`C_Kc` remains source-unverified**, and the pipeline is sharply sensitive to it.
4. **n = 8** for all dataset-level significance testing.
5. **The competing paper's full text was not obtained** (HTTP 403). Whether they also bound the correction is unknown.
6. **The correction model was never tuned or re-compared inside the architecture.**
7. **No per-geometry hydraulic-diameter treatment.**
8. **Three of the project's own claims were found not to reproduce.** Others may remain; only claims touched by this audit were re-verified.

## H. Remaining theoretical possibilities

- Per-geometry backbones or geometry-specific correction factors for annuli and plates.
- Monotonic-constrained gradient boosting to enforce physical trends structurally.
- Regime discovery followed by regime-specific correction.
- Transfer learning pretrained on the look-up table, as an alternative to residual anchoring.
- Heteroscedastic or quantile regression as an alternative uncertainty route.
- A systematic full-text audit of published CHF machine-learning papers.
- Obtaining genuinely external CHF data.

## I. Why those were or were not pursued

| Possibility | Status | Reason |
|---|---|---|
| Per-geometry treatment | **Not pursued** | Zhao's three geometries are not separately labelled in the loader; requires data work beyond audit scope. **Highest-value next step.** |
| Monotonic constraints | Not pursued | Existing monotonicity checks already **pass**; the bounded structure supplies the trend. Low expected value. |
| Regime discovery | Not pursued | Physics arms already encode regime selection (Katto's density gate, Biasi's two branches). Statistical regimes would likely recover source identity. |
| LUT pretraining | Not pursued | Requires a different model class; a genuine research direction, not an audit item. |
| Quantile/heteroscedastic UQ | Not pursued | Cross-dataset conformal already achieves near-nominal coverage. Diminishing returns. |
| Full-text literature audit | **Not possible here** | Paywalls returned HTTP 403. Requires institutional access. **Blocking for submission.** |
| External data | **Not possible** | No such dataset available. Requires new experiments or new data agreements. |

## J. Publishability

**Realistic assessment:**

- **Not** a top-tier methods venue. The architecture is incremental and its headline framing was pre-empted in 2026.
- **Plausible** at a solid archival journal — *Nuclear Engineering and Design*, *Applied Thermal Engineering*, *International Communications in Heat and Mass Transfer* — as a methodology and negative-results contribution, **provided** the novelty is repositioned per §F, the competing paper is cited as prior art, `C_Kc` is verified, and the Zhao backbone is handled honestly.
- **Strong** as a B.Tech thesis. The negative-results discipline, the bug found in a reference standard, and the self-correction record exceed normal undergraduate standards.
- **The single highest-leverage addition** would be the per-geometry treatment for Zhao, or genuinely external validation data.

## K. Maximum-state declaration

> *Given the complete project context, available data, computational resources, available literature, and analyses that can actually be performed, have we exhausted the scientifically meaningful improvements that can reasonably be made without obtaining fundamentally new experimental information?*

**NO.** Meaningful improvements remain. Specifically:

1. **Per-geometry treatment of Zhao.** It is the worst dataset on every metric and the only one where physics beats the pipeline. The mixed-geometry hypothesis is plausible and untested. **Blocked by: geometry labels not exposed in the loader — a few hours of data work, not a fundamental limit.**
2. **Full-text verification of the competing 2026 paper.** The entire novelty claim rests on what it does and does not do. **Blocked by: paywall (HTTP 403). Requires institutional access.**
3. **Source verification of `C_Kc`.** The pipeline is sharply sensitive to it. **Blocked by: the 1984 paper was not retrievable; the NASA transcription omits it.**
4. **External validation.** **Blocked by: no such data exists in reach. Genuinely requires new experimental information.**
5. **Model-choice comparison inside the anchored architecture.** **Not blocked — simply not run, and judged low-value. That judgement is untested.**

Items 1 and 5 are achievable now. Item 2 needs a library. Item 3 may need a document that could not be located. Item 4 is the only one that genuinely requires new experiments.

**Therefore the honest answer is NO — the project is not at its maximum defensible state, and I have named precisely what is missing and why each was or was not closed.** What *has* been achieved is that the project's claims are now correctly attributed, correctly sized, and correctly positioned against the literature, with one real bug fixed and one real weakness repaired.

---

*Every number in this audit is traceable to a file in `results/prof_tasks/`: `audit_v2_features.csv`, `audit_katto_c.csv`, `audit_ood_conformal.csv`, `audit_ood_conformal_strict.csv`, `audit_fluid_transfer.csv`, `FINAL_results.csv`, `router_results.csv`. Scripts are in `scripts/prof_tasks/audit_*.py`.*

---

# ADDENDUM — THE CORRECTION LEARNER, TESTED INSIDE THE ARCHITECTURE

Added after the main audit. This closes item 5 of §K ("not blocked, simply not run,
and judged low-value — that judgement is untested"). **It is now tested, and the
judgement was right, though not for the reason assumed.**

## A.1 Why it needed testing

The 961-run grid compared 13 learners **standalone**, predicting CHF directly.
Inside this pipeline the learner solves a different problem: predicting
`log(CHF / backbone)`, a small well-behaved residual. A learner that wins at one
task need not win at the other. ExtraTrees was inherited from the standalone grid
and never re-tested for the residual task.

`CHFPipeline` now takes a `model=` callable so the correction learner is
configurable; default behaviour is unchanged.

## A.2 Result — 10 learners, leave-one-dataset-out, 3 seeds for stochastic ones

| Correction learner | Median R² | Worst | Mean R² | Median ±10% | Time |
|---|---|---|---|---|---|
| CatBoost | **0.7246** | −0.6867 | 0.4570 | 42.1% | 14s |
| XGBoost | 0.7105 | **−0.3411** | **0.5212** | 41.9% | 29s |
| **ExtraTrees300 (shipped)** | 0.7084 | −0.4357 | 0.5075 | 32.3% | 35s |
| MLP(128,64) | 0.6697 | −0.5462 | 0.4329 | 37.1% | 70s |
| LightGBM | 0.6442 | −0.8329 | 0.4283 | **44.5%** | 59s |
| RandomForest300 | 0.6195 | −0.7251 | 0.4299 | 28.0% | 94s |
| HistGB | 0.6149 | −0.7711 | 0.4237 | 43.1% | 90s |
| SVR-RBF | 0.6104 | −0.3712 | 0.4625 | 30.9% | 8s |
| kNN k=5 | 0.5226 | −0.3861 | 0.4344 | 24.2% | 1s |
| Ridge | 0.4297 | **−0.2945** | 0.3867 | 26.3% | 0s |

## A.3 Paired significance vs the shipped model (Wilcoxon, n = 8 datasets)

| Comparison | Metric | Wins | p | Median Δ |
|---|---|---|---|---|
| XGBoost vs ExtraTrees | R² | 5/8 | **0.547** | +0.0034 |
| CatBoost vs ExtraTrees | R² | 3/8 | 0.461 | −0.0142 |
| LightGBM vs ExtraTrees | R² | 2/8 | 0.078 | −0.0333 |
| XGBoost vs ExtraTrees | ±10% | 5/8 | **0.461** | +0.5 pts |
| LightGBM vs ExtraTrees | ±10% | 6/8 | 0.313 | +1.7 pts |

**No learner is significantly better than ExtraTrees on either metric.**

## A.4 A statistical trap worth recording

On first reading the summary table, the ±10% column appeared to show ExtraTrees
losing badly — 32.3% against 41.9% for XGBoost and 44.5% for LightGBM, a gap of
9–12 points on the metric the supervisor explicitly requires. That looked like a
clear reason to switch.

It is an artefact of **where the median falls**. Per-dataset:

| Held out | ExtraTrees | XGBoost | LightGBM |
|---|---|---|---|
| D1 NRC | 46.9 | 43.1 | 48.9 |
| D2 Zhao | 7.7 | 6.2 | 2.3 |
| D3 KAERI uniform | 58.7 | 59.7 | 60.2 |
| D4 KAERI non-uniform | 47.1 | 46.0 | 44.3 |
| D5 Helical R-123 | 14.4 | 18.4 | 19.1 |
| **D6 Hardik helical** | **30.1** | **48.3** | 44.7 |
| D7 Hardik straight | 34.5 | 40.6 | 46.1 |
| D8 Pioro | 15.9 | 16.0 | 16.0 |

ExtraTrees' D6 value (30.1) sits exactly at the median position, so a single
dataset drags its summary median from roughly 42 down to 32.3. The paired
difference is **+0.5 points, p = 0.46**.

**This is the same trap the project documents elsewhere** — a summary statistic
read without the paired per-item comparison behind it. It was fallen into twice
in one session, on R² and again on ±10%. Recorded because the median-of-medians
appears in several tables in this audit and should always be read alongside the
paired test.

## A.5 What this actually establishes

The interesting result is not "ExtraTrees wins." It is how **little the learner
matters**: ten learners spanning ridge regression, k-nearest neighbours, support
vector regression, four gradient-boosting implementations, random forests and a
neural network span only 0.43 to 0.72 in median held-out R², with the top six
inside 0.11 of each other.

That is consistent with §4: **the anchor supplies the generalisation, and the
learner on top of it is close to interchangeable.** The same conclusion the trust
-signal ablation reached (three signals, 0.736–0.753 whichever is used alone) and
that §3 reached about features.

Note also Ridge's behaviour: the **best worst case of all ten** (−0.2945) with the
worst median (0.4297). A linear correction cannot go far wrong, which is the
bounded-correction philosophy arrived at from a different direction.

**Decision: keep ExtraTrees(300). Not because it is best, but because nothing is
significantly better and it is already validated, serialised and tested.**

