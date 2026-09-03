# physics_pipeline

Physics-first CHF prediction: closed-form correlations as the predictive
backbone, with machine learning restricted to a bounded correction on top.

Self-contained. All code lives in `scripts/`, all outputs in `results/`.
It **reads** the merged dataset and the four split definitions from
`../unified_chf_pipeline/` and never writes there.

Theory and provenance for every equation used here:
[`../physics_foundation/CHF_Physics_Foundation.md`](../physics_foundation/CHF_Physics_Foundation.md).

## The idea

Instead of predicting CHF directly, predict a bounded dimensionless correction
to a physical scale:

```
CHF = Phi_physics(x) * exp( g_theta(pi(x)) )
```

- `Phi_physics` — closed-form, regime-dispatched, **zero fitted parameters**.
  Zuber for pool boiling, Katto-Ohno or Hall-Mudawar for flow boiling.
- `pi(x)` — dimensionless groups (Weber, Katto, density ratio, reduced
  pressure, quality, L/D, Jakob, Bond), not raw engineering units.
- `g_theta` — a learned correction, bounded to at most 3x, optionally monotone
  in quality and mass flux, and decaying to zero away from the training data.

## Layout

```
physics_pipeline/
├── scripts/
│   ├── paths.py              where inputs are read from, outputs written to
│   ├── physics/
│   │   ├── properties.py     saturation properties (CoolProp + FC-72 table)
│   │   ├── correlations.py   Zuber, Kutateladze, Kandlikar, Kim, Katto-Ohno,
│   │   │                     Hall-Mudawar, Biasi, Tanase n, K1, coil factor
│   │   ├── groups.py         the dimensionless feature map pi(x)
│   │   ├── baseline.py       Phi_physics: none / latent / katto / gated
│   │   ├── constraints.py    the C1-C7 / S1-S10 physics scorecard
│   │   └── repair.py         Stage-0 data repair
│   ├── features_v2.py        feature matrices, "raw" and "pi" spaces
│   ├── models_v2.py          PhysicsCorrectedModel (bounded/monotone/trust)
│   ├── metrics_utils.py      R2 / RMSE / MAE / MAPE  (copy of the shared one)
│   ├── run_ablation.py       the A0-A5 ladder on all four splits
│   ├── run_improvements.py   follow-up experiments E1-E3 (splits 3 and 4)
│   ├── report_ablation.py    builds results/ablation_report.md
│   ├── per_source_r2.py      R2 matrix: every model x source x split
│   ├── validate_physics.py   correlation sanity checks, run this first
│   └── attribution_test.py   separates data fixes from physics ideas
└── results/
    ├── ablation_report.md    MAIN REPORT
    ├── improvement_report.md follow-up experiments E1-E3
    ├── per_source_r2.md      R2 by source dataset for every model
    ├── ablation_metrics.csv  raw metrics, one row per arm x split x fold
    └── predictions/          per-row predictions for every split
```

The `_v2` suffixes are kept so `attribution_test.py` can import the original
`features.py` / `models.py` from the shared pipeline without a module-name
collision. That script is the only place this folder imports shared *code*;
everywhere else it only reads shared *data*.

## Run

```bash
cd physics_pipeline/scripts
python validate_physics.py    # correlation checks -- run this first
python run_ablation.py        # ~10 min, writes ablation_metrics.csv
python report_ablation.py     # writes ablation_report.md
python per_source_r2.py       # writes per_source_r2.md
python run_improvements.py    # follow-up experiments E1-E3
python attribution_test.py    # data-fix vs physics-idea attribution
```

Scripts must be run from inside `scripts/` (they import `paths`, `physics`,
`features_v2` as top-level modules). If the shared pipeline ever moves, set
`CHF_DATA_DIR` and `CHF_SPLITS_DIR`.

## Data

Everything is scored on `../unified_chf_pipeline/data/master_chf_dataset.csv`
— 28,470 rows, 7 experimental sources, using the **same four split
definitions** as the original bake-off so the numbers are directly comparable.

| Split | Train / test | Whole source held out? |
|---|---|---|
| 1 random, stratified | 22,776 / 5,694 | no — every source on both sides |
| 2 condition-wise (top 20% pressure) | 21,851 / 6,619 | no |
| 3 surface-wise | 28,038 / 432 | **yes** — helical coil + pin-fin |
| 4 leave-one-source-out | 7 folds | **yes** — each source in turn |

Splits 1 and 2 keep the same rig on both sides, so they cannot distinguish
between models (a dozen models all score 0.96-0.98 on split 1, which is inside
the ~10% inter-laboratory reproducibility floor). **Splits 3 and 4 are the
ones that carry information.**

## What the results say

| | S1 random | S2 pressure | S3 surface | S4 LOSO |
|---|---:|---:|---:|---:|
| Published pipeline | 0.968 | 0.890 | 0.173 | 0.784 |
| A0 no physics | 0.979 | 0.903 | **-18.0** | 0.719 |
| A1 physics baseline | 0.970 | 0.877 | **0.711** | 0.800 |
| A2 + Katto-Ohno | 0.978 | 0.812 | 0.606 | -0.797 |
| A3 + pi features | 0.982 | 0.790 | 0.252 | -0.467 |
| A4 + gating | 0.979 | 0.827 | 0.315 | 0.620 |
| A5 + constraints | 0.957 | 0.854 | 0.506 | 0.710 |
| Physics only, no training | 0.779 | 0.508 | 0.505 | 0.785 |

Three findings:

1. **Physics vs no physics is a large win.** A0 and A1 differ only in whether a
   physics baseline exists; on the surface-wise split that is -18.0 vs 0.711.
2. **More elaborate physics did not beat simple physics.** Katto-Ohno, the
   dimensionless feature space and mechanism gating all lost to plain
   Zuber / G*h_fg. On the NRC fold, physics alone scores 0.926 and an
   unbounded learned correction on top drags it to -1.286.
3. **Constraints buy safety.** Every catastrophic result in the whole matrix
   belongs to an unconstrained model (ANN reaches -142,030 on the helical
   fold; the same architecture with a bound and trust decay reaches +0.093).

## Follow-up experiments (results/improvement_report.md)

Three targeted attempts to improve on A1. **None of them beat it** -- best new
arm reached 0.686 pooled LOSO against A1's 0.800. They did establish two
design rules and one robustness result:

- **Bound tightness must match baseline quality.** A 3x bound is fine on
  Katto-Ohno, which already explains most of the variation, and crippling on
  the latent-heat scale, which explains none of it and needs the correction to
  span ~600x (R2 = -0.82 on the surface split).
- **A global calibration offset transfers rig bias.** Fitting one median
  offset on the other six sources and applying it to the held-out one moved
  pooled LOSO from 0.710 to 0.658. Per-rig medians of CHF/Phi run 0.97 (NRC)
  to 2.59 (pin-fin) and are not transferable. `models_v2` therefore applies
  calibration only when the baseline actually needs it.
- **More trust decay is better, not less.** Sweeping trust_gamma over
  0.25 / 0.5 / 1.0 / 2.0 gave pooled LOSO 0.154 / 0.351 / 0.489 / 0.565 --
  monotone. The learned correction is actively harmful out of domain, which is
  the opposite of the hypothesis the sweep was built to test.
- **Correlation stacking is the most robust learned arm.** Feeding every
  correlation's predicted log(Bo) as a feature instead of dividing by one
  reaches 0.718 pooled LOSO and, unlike A2's -0.797, never collapses on any
  fold -- but it still does not beat A1.

## Known caveats

Recorded here rather than buried, because they affect how the numbers may be
quoted.

- **Single seed.** Every number above is one seed. This repository has already
  retracted a headline result after multi-seed verification. The wide gaps
  (-18.0 vs 0.711) will hold; the narrow ones (A1 0.800 vs physics-only 0.785)
  should not be quoted until bootstrapped.
- **Weak leakage in the helical repair.** `physics/repair.py` recovers the six
  coil tube diameters from `d = Q/(pi L CHF)`, which uses the target. Six
  per-rig constants from 257 rows, and the measured effect is small
  (strategy 3: 0.173 -> 0.186), but it should be re-derived from training rows
  only, or cited from the source apparatus table, before publication.
- **`C_Kc` is unverified.** The Katto-Ohno length constant could not be
  obtained from a primary source. The median baseline is unaffected, but 24.8%
  of rows move, some by ~3x. The ablation is internally valid (every arm uses
  the same value); absolute CHF numbers are provisional.
- **A5's out-of-domain numbers are the physics, not the model.** Trust decay
  zeroes the correction for 100% of strategy-3 test rows, so A5 there reduces
  exactly to `PHYS_gated`. Softening `trust_gamma` is the obvious next step.
- **Two corrections were implemented, tested, and deliberately not applied** —
  a Vishnev orientation factor and a Lienhard-Dhir finite-size factor. The
  55 rows of orientation data contradict the literature trend, and the
  finite-size correction has no transcribed sub-threshold form. Both are
  documented in `physics/baseline.py`.
- **The surface-characteristics story rests on 230 rows** (0.8% of the data)
  with no contact-angle column at all. `correlations.kim_roughness_K` and
  `kandlikar_K` are implemented and unused for that reason.
