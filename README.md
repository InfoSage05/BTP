# Critical Heat Flux (CHF) prediction benchmark

This BTP repository studies machine-learning prediction of critical heat flux (CHF) from the 2006 Groeneveld CHF Look-Up Table (LUT), with emphasis on honest pressure-range extrapolation rather than only random-split accuracy.

## Current scope and status

This is currently a **LUT-based benchmark and reproducibility audit**. It is not yet the engineered-surface CHF study described in `docs/manuscript/paper_outline.docx`: this repository has no surface roughness, wettability, contact angle, CHF-detection criteria, or leave-one-surface-out surface validation. That scope must be confirmed with the advisor before publication claims are finalized.

The strongest defensible contribution is the leakage-aware, multi-seed audit showing how single-seed selection and test-set tuning can mislead CHF extrapolation claims.

## Data and physical problem

CHF is the heat-flux threshold where nucleate boiling transitions to a vapor-blanketed regime, potentially causing a sharp wall-temperature increase. Inputs are `P` (pressure, kPa, 100–21,000), `G` (mass flux, kg m⁻² s⁻¹, 0–8,000), and `X` (thermodynamic quality, −0.50–1.00). The target `CHF` is in kW m⁻².

The LUT has `24 × 21 × 23 = 11,592` grid rows, with no duplicates or nulls. Exactly 504 rows have `CHF == 0`, all at `X == 1.0`; these all-steam placeholders are excluded, leaving **11,088 usable rows**. Non-zero CHF spans 15–44,338 kW m⁻², so raw-target and `log(CHF)` variants are compared and metrics are computed on the raw CHF scale.

Canonical data: `data/chf_long_clean.csv`. Regenerate it with `scripts/prepare_data.py`; `data/chf_long_with_gridbase.csv` is derived data.

## Validation protocol

1. **Split A:** random 80/20 interpolation test, repeated for seeds 0–4 and reported as mean ± standard deviation. It is optimistic because nearby grid points remain in training.
2. **Split B:** interior pressure holdout with training pressure levels on both sides. The earlier bug that selected the topmost pressure level was corrected.
3. **Split C:** train on `P <= 16,000 kPa`, test on 17,000–21,000 kPa (2,310 rows). This is the primary honest extrapolation test.

The grid interpolator is not meaningfully scored on random Split A because removed nodes create holes; on Split C its linear extrapolation is a valid deterministic baseline.

## Verified Split-C results

| Approach | Deterministic? | R² | MAPE | Meaning |
|---|---:|---:|---:|---|
| Grid interpolation, raw | Yes | **0.8415** | 20.8% | strongest reliable table baseline |
| Grid interpolation, log | Yes | 0.8040 | 26.7% | deterministic alternative |
| Degree-2 Ridge, log | Yes | **0.7547** | 35.8% | strongest reliable trained ML baseline |
| MLP, log | No | 0.6277 ± 0.072 | 40.0% | more stable neural baseline |
| MLP, raw | No | 0.4412 ± 0.246 | 70.5% | highly seed-sensitive |
| Tree ensembles | Effectively | ≈0.43 | ≈42% | structural extrapolation collapse |

In-domain Split A/B results are generally above `R²=0.97` for flexible models, but that indicates interpolation—not safe out-of-domain use. The favorable pressure-gated `R²=0.855` result was retracted after multi-seed verification and must not be presented as representative.

## Physics-informed and hybrid work

Implemented extensions include physics-basis Ridge features, Biasi/Zuber residual learning, a PyTorch collocation network with quality-monotonicity and Zuber pressure-trend penalties, a pressure-gated mixture of experts, and Tier-1/Tier-2/Tier-2b proxy-band searches.

Biasi and Zuber implementations were sanity-checked and corrected: Biasi uses the dimensional constants and correct branch combination; Zuber uses CoolProp/IAPWS properties; the dispatcher uses Zuber at `G=0` and Biasi for `G>0`. Collocation training uses scaled coordinates and caches the Zuber derivative-sign lookup.

Negative results are important. Near-critical logarithmic features can explode during extrapolation, and residual corrections can fail when correlation error changes shape outside the training range. `results/pinn/tier2b_summary.json` reports raw `R²=0.8474`, but the best penalty weights are `lam_mono=0` and `lam_zuber=0`; this is not evidence that active physics constraints improve performance.

## Open blockers before publication

See [CHF Extrapolation Audit.html](<CHF Extrapolation Audit.html>) and [docs/SENIOR_REVIEW.md](docs/SENIOR_REVIEW.md).

- Re-run physics-informed selection only with a training-only proxy validation band; evaluate Split C once at the end.
- Retire the earlier test-selected PINN headline `R²=0.8123` everywhere.
- Run bootstrap/statistical comparisons against the grid baseline and freeze one canonical results table.
- Verify or remove provisional literature citations and metrics.
- Add LQR-focused error analysis and write Methods, Results, Discussion, and Conclusion.
- Resolve whether the final paper is the LUT/rigor study, mentor/coil-geometry extension, or engineered-surface study.

The separate external-data track, `scripts/plan2_pipeline.py`, audits mentor/PDF data. Its summary reports 55 mentor rows, 468 PDF rows, exact identity/unit checks, and grouped-CV log-degree-2 Ridge at `R²=0.9139`; it is not a replacement for the main LUT benchmark.

## Repository map

```text
data/raw/                         supplied workbooks and PDFs
data/chf_long_clean.csv           canonical usable LUT data
cad_models/                       CAD generation and STEP/STL exports
notebooks/                        modelling, physics, PINN, and diagnostics
scripts/prepare_data.py           data cleaning and CSV generation
scripts/verify_results.py         audit and multi-seed verification
scripts/plan2_pipeline.py         mentor/PDF audit and external evaluation
scripts/chf_physics.py            Biasi, Zuber, and hybrid utilities
scripts/build_notebook*.py        notebook builders
results/                          CSV/JSON summaries, figures, and outputs
docs/                             context, review, manuscript, and references
CHF Extrapolation Audit.html      publication-readiness audit
requirements.txt                  Python dependencies
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/build_notebook.py
python scripts/build_notebook_physics.py
python scripts/build_notebook_pinn.py
python scripts/build_model_test_notebooks.py
python scripts/verify_results.py
python scripts/plan2_pipeline.py
```

Longer PINN/full-training runs are available through `modal_btp_gpu_pipeline.py` and `modal_pinn_grid_search.py`. Modal is not required for deterministic baselines.

## References

The primary source is the 2006 Groeneveld CHF Look-Up Table, published in *Nuclear Engineering and Design* (2007). Supporting papers and PDFs are in `docs/references/`; manuscript drafts are in `docs/manuscript/`. Reported numbers should always identify the split, seed protocol, target transform, configuration, and result file.
