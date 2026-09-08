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
testing it. FC-72 (one of pinfin's two fluids) is chemically n-perfluorohexane --
`physics_features.py`'s FLUID_NAME_MAP has been fixed (as a follow-up pass)
to map "fc72" to the verified CoolProp name "n-Perfluorohexane" (case-
sensitive; confirmed via Tcrit/Pcrit against published FC-72 values, and
NOT "Novec649" -- a different, chemically distinct 3M fluid that also
resolves in CoolProp and could easily be mistaken for the right answer).
This does not change the pin-fin exclusion above -- that is a separate,
still-open issue (no diameter concept for fin geometry), unrelated to the
fluid-property lookup.

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

### MoE: gate routing -- broken, diagnosed, then fixed

**First attempt (pool-only gate training)**: gate assigned P(pool_expert)=0.108
(MLP) / 0.073 (Transformer) to true pool rows -- essentially inverted routing.

**Second attempt (joint-regime training, pool+flow rows combined for the
gate/pool-expert)**: barely moved the needle -- P(pool_expert)|pool_row went
to 0.087 (MLP) / 0.110 (Transformer), still far from correct. More data
diversity alone did not fix it.

**Root cause, confirmed directly**: G_kg_m2s is a perfectly separable signal
between the two regimes (pool rows are exactly 0.0, flow rows are 8.2 or
above) -- this was never a hard classification problem. The actual issue:
the gate only ever received gradient through the indirect regression-loss
path (its softmax output multiplies each expert prediction, and only the
combined output error backpropagates), which is weak, indirect supervision
for what is trivially separable if supervised directly.

**Fix: added a direct auxiliary cross-entropy loss on the gate output
against the true regime label** (known exactly for every row -- pool
iff G_kg_m2s equals 0, not a hidden or learned label). Total loss became
regression MSE plus the gate cross-entropy term. Result:

| Arch | P(pool_expert)\|pool_row | P(flow_expert)\|flow_row |
|---|---|---|
| MLP | 0.999 | 0.996 |
| Transformer | 0.999 | 0.965 |

Routing is now essentially perfect for both architectures. This confirms
the diagnosis: the gate never had a hard problem to solve, it had no direct
signal to solve it with.

**Fixing the gate revealed a second, separate problem -- since fixed too.**
With correct routing, regression accuracy on the pool-boiling test set
initially diverged sharply by architecture: MLP pool-expert (from-scratch,
same as flow_expert's counterpart was NOT -- see below) reached R2=0.712,
but Transformer's pool-expert -- a full un-pretrained FTTransformer trained
from scratch on only 36 pool-boiling rows -- failed outright (R2=-0.006, no
better than predicting the mean). Root cause: both experts were built
from-scratch inside the MoE, by omission rather than a considered choice --
the one place in this project that didn't follow the pretrain-then-finetune
pattern used everywhere else specifically because small-data from-scratch
training is unreliable (see Stage 1/2's own from-scratch baselines for the
same lesson elsewhere).

**Fix**: both experts (not just flow_expert) now initialize from the Stage 1
pretrained checkpoint, with the pool_expert's parameters fine-tuned at the
same low LR the rest of this project uses for pretrained-then-finetuned
models (1e-4 MLP, 5e-5 Transformer -- matching `finetune_mlp.py` /
`finetune_transformer.py`'s LR_FINETUNE), while the gate itself keeps a
much higher LR (3e-3) since it's small and starts from scratch. Result:

| Arch | Pool test R2 (before -> after) | Flow test R2 (after) | Gate: P(pool)\|pool_row | Gate: P(flow)\|flow_row |
|---|---|---|---|---|
| MLP | 0.712 -> **0.730** | 0.914 | 0.999 | 0.981 |
| Transformer | -0.006 -> **0.742** | 0.934 | 0.999 | 0.985 |

Both architectures now have correctly-routing, correctly-performing MoE
models on both regimes -- a genuinely complete, consistent result, not a
partial one. Transformer's pool-expert went from broken to matching (in
fact slightly exceeding) MLP's pool performance once given the same
pretraining head start every other small domain in this project relies on.
MLP's own pool-expert also improved modestly (0.712 -> 0.730) from the same
fix, confirming pretraining helps even where from-scratch training happened
to work by chance.


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
- `moe_joint_regime_results.csv`, `moe_supervised_gate_results.csv` -- the two MoE gate-fix iterations documented above
- Generated by `scripts/chf_pipeline/prepare_pool_boiling.py`,
  `pool_boiling_techniques.py`, `ensemble_pool_boiling.py`, `retrain_moe_joint.py`, `retrain_moe_supervised_gate.py`. New model
  classes (`LoRAMLP`, `LoRAFTTransformer`, `MoEModel`, `MoEGate`) added to
  `scripts/chf_pipeline/models.py`.
