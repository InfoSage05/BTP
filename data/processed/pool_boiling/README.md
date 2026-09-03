# Pool-boiling schema mapping, technique comparison, and ensemble

## Task 1: dataset preparation (G=0 mapping)

**Strip dataset** (`strip_pool_boiling_water.csv`, 55 rows, water, from
`data/Master file Strip.xlsx`): mapped into the existing flow-boiling
(P,G,X,D) schema per explicit instruction -- G_kg_m2s=0 for all rows (no
forced flow). P_kPa = sheet's P[bar]*100 (verified ~99.84 kPa, near-
atmospheric). D_mm = "Apparent dia (mm)" directly. X = -Cp*(Tsat-Tpool)/hfg,
the standard subcooling-to-equivalent-quality conversion (Cp~4181 J/kg-K,
hfg~2.258e6 J/kg both verified consistent with water at ~1 atm -- not
fabricated units). CHF_kW_m2 = sheet's CHF[W/m^2]/1000, cross-checked
exactly against the sheet's own "CHF(MW/m^2)" column before use. All 55
rows retained (no imputation). Angle/Orientation kept as metadata columns,
NOT fed to the model -- the (P,G,X,D) schema has no way to represent heater
inclination, a real and important limitation (this dataset sweeps 0-180
degrees; the model is blind to that entirely).

**pin-fin dataset EXCLUDED from this mapping** (kept in
`data/raw/fine_tuning/pinfin_chf_water_fc72.csv`, untouched): it has no
bulk-diameter concept, only fin Width/Height/Spacing at O(10-100 um) --
2-3 orders of magnitude below the model's training diameter range
(3-20mm). Substituting fin dimensions for D_mm would silently force the
model to extrapolate into a wildly unphysical regime, worse than not
testing it. Also worth noting for future work: FC-72 (one of pinfin's two
fluids) is chemically n-perfluorohexane, which IS available in CoolProp
under that name -- `physics_features.py`'s FLUID_NAME_MAP currently has
`"fc72": None`; this could be fixed to `"n-Perfluorohexane"` if pinfin is
revisited with a better diameter proxy later.

## Task 2+3: technique comparison (from-scratch / full fine-tune / LoRA / MoE)

| Arch | Technique | Test set | R2 | MAPE% | Notes |
|---|---|---|---|---|---|
| MLP | from_scratch | pool-only | 0.012 | 55.0% | fails without pretraining -- 44 rows isn't enough alone |
| MLP | full_finetune | pool-only | 0.696 | 27.1% | best single MLP technique |
| MLP | lora | pool-only | 0.685 | 24.1% | **using only 8.8% of the parameters** |
| MLP | moe | pool-only | 0.474 | 43.4% | weakest -- see gate failure below |
| Transformer | from_scratch | pool-only | 0.774 | 21.1% | works fine even without pretraining here |
| Transformer | full_finetune | pool-only | 0.775 | 20.4% | best, but barely beats from-scratch |
| Transformer | lora | pool-only | 0.734 | 16.9% | **using only 1.65% of the parameters** |
| Transformer | moe | pool-only | 0.755 | 21.8% | close to full_finetune |

**LoRA is the standout practical result**: on both architectures it comes
within ~0.01-0.04 R2 of full fine-tuning while training 1.65-8.8% of the
parameters. For a 55-row domain this is a genuinely useful technique, not
just a cheaper approximation -- fewer trainable parameters relative to a
tiny dataset is itself a form of regularization against overfitting.

**Transformer's from-scratch result is notable**: unlike every flow-boiling
domain in Stage 2 (where from-scratch was consistently weak or broken),
the Transformer trained from scratch on just 44 pool-boiling rows reaches
R2=0.774, almost matching the pretrained+fine-tuned version. This suggests
either (a) this particular domain is easier to fit than the R123 flow
domains were, or (b) the flow-boiling pretraining provides little relevant
prior for pool boiling, since the governing physics is different enough
(no mass flux, no flow quality) that the "prior knowledge" barely
transfers. Both are plausible; distinguishing them would need more
pool-boiling data than is available here.

### MoE: honest finding -- the gate did not learn to route by regime

Gate check (average P(pool_expert) on true pool rows, average
P(flow_expert) on true flow rows -- both should approach 1.0 for correct
routing):

| Arch | P(pool_expert)\|pool_row | P(flow_expert)\|flow_row |
|---|---|---|
| MLP | 0.108 | 0.609 |
| Transformer | 0.073 | 0.665 |

**Both far from ideal (1.0), and MLP's is essentially inverted** -- the
gate assigns *low* weight to the pool expert even on pool rows. Root
cause, not a training bug: the gate and pool expert were trained ONLY on
pool-boiling rows (the MoE's training set was the pool train split alone).
The gate therefore never saw a single flow-regime (G>0) example during
training and had no data-driven reason to learn "route away from
pool_expert when G>0." A meaningful MoE gate needs to be trained on a
dataset spanning BOTH regimes simultaneously, not fine-tuned on one regime
alone with the other regime appearing only at test time. This was not
re-run with joint training due to time scope, but is the clear, documented
next step if MoE is pursued further.

Evaluated on a MIXED test set (11 pool + 11 flow-boiling rows from Stage
1's held-out core_interp test split, never used in any pool-boiling
training):
- MLP: R2=0.638 (n=22) -- better than pool-only alone (0.474), because the
  frozen flow expert still does reasonably on flow rows regardless of the
  gate's imperfect weighting.
- Transformer: R2=-0.073 (n=22) -- **breaks down** on the mixed set despite
  scoring 0.755 on pool-only. The combination of an uncalibrated gate and
  the frozen pretrained-only Transformer (not per-domain fine-tuned, since
  Stage 2's domain-specific fine-tuning wasn't reapplied here) performing
  unevenly across mixed regimes compounds into a clearly worse result than
  either single-regime evaluation suggested.

## Task 4: deep ensemble (5 members, full_finetune technique)

| Arch | Ensemble R2 | Single-model R2 (mean +/- std) | Coverage 95% |
|---|---|---|---|
| MLP | 0.730 | 0.728 +/- 0.014 | 18.2% |
| Transformer | 0.772 | 0.770 +/- 0.004 | 9.1% |

Same pattern as every other ensemble in this project: members converge to
near-identical solutions (low std), ensembling gives a small but real
accuracy bump for MLP (+0.034) and essentially nothing for Transformer
(+0.003), and calibration remains poor (9-18% coverage vs. the 95% target)
-- consistent with the systemic under-diversity issue already documented
in `data/processed/stage3/README.md`.

## Files
- `strip_pool_boiling_water.csv` -- prepared dataset (Task 1)
- `pool_boiling_technique_comparison.csv` -- Task 2+3 full results
- `pool_boiling_ensemble_results.csv` -- Task 4 results
- Generated by `scripts/chf_pipeline/prepare_pool_boiling.py`,
  `pool_boiling_techniques.py`, `ensemble_pool_boiling.py`. New model
  classes (`LoRAMLP`, `LoRAFTTransformer`, `MoEModel`, `MoEGate`) added to
  `scripts/chf_pipeline/models.py`.
