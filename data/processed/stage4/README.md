# Stage 4: final comparison across every real-data regime, both architectures

Everything in `stage4_final_comparison.csv` uses the same deep-ensemble
(full-fine-tune) recipe, evaluated on genuinely held-out test splits, for
both MLP and Transformer, across every domain built in this project:
5 flow-boiling fine-tuning domains + 2 flow-boiling core splits (Stage 3)
+ 1 new pool-boiling domain (this task). See `data/processed/stage2/`,
`stage3/`, and `pool_boiling/` for the full per-stage detail this
summarizes.

## The comparison table

| Regime | Domain | Fluid | n | MLP R2 | Transformer R2 |
|---|---|---|---|---|---|
| flow | core_interp | water | 2,459 | 0.963 | **0.968** |
| flow | core_extrap | water | 5,994 | 0.916 | **0.953** |
| flow | hardik2016 coils | R123 | 31 | 0.560 | **0.736** |
| flow | hardik2017 tubes | R123 | 11 | **0.481** | 0.378 |
| flow | kaeri uniform | water | 130 | **0.907** | 0.903 |
| flow | kaeri nonuniform | water | 177 | **0.864** | 0.845 |
| flow | zhao2020 tubes | water | 287 | 0.890 | **0.892** |
| pool | strip (this task) | water | 11 | 0.730 | **0.772** |

## Does either model actually understand the physics?

**Partially, and unevenly -- not a clean yes.** Three separate pieces of
evidence, each pointing the same direction:

1. **Within flow boiling, both models generalize reasonably well across
   fluid and geometry shifts they were pretrained toward** (water tubes,
   R123 coils/tubes, non-uniform heating) -- R2 in the 0.48-0.97 range,
   worse on the smallest/noisiest domains (hardik2017, n=11) but never
   catastrophically broken once fine-tuned. This says the pretraining
   captured something genuinely transferable about how P, G, X, and D
   jointly govern CHF within the flow-boiling regime -- not perfect, but
   real.

2. **On pool boiling -- a regime the pretraining never saw governing
   variables for (no mass flux, no flow quality; only G was forced to
   zero as a schema trick) -- both models still reach R2~0.73-0.77 after
   fine-tuning, which is comparable to their WORST flow-boiling domains,
   not a collapse.** But the from-scratch Transformer result
   (R2=0.774, matching the pretrained+fine-tuned version almost exactly)
   is the tell: the flow-boiling pretraining provided little to no
   measurable advantage here. The model isn't understanding pool-boiling
   physics via transferred flow-boiling knowledge -- it's re-learning a
   compact regression from the ~44 pool-boiling training rows directly,
   largely independent of what it learned about flow boiling. G=0 is a
   schema trick that lets the same input slots accept pool-boiling data;
   it is not evidence the model learned that "G=0" means "a different
   physical mechanism governs this row."

3. **The Mixture-of-Experts result makes this concrete and quantifiable.**
   A gate trained only on pool-boiling rows assigned P(pool_expert)=0.11
   (MLP) / 0.07 (Transformer) to true pool rows -- essentially the
   opposite of correct routing. The models have no innate signal that
   distinguishes "this row needs pool-boiling reasoning" from "this row
   needs flow-boiling reasoning" beyond whatever a human-engineered
   feature (G=0) hands them -- and even with that explicit hint sitting
   right in the input, an undertrained gate failed to use it reliably.
   That is the most honest available answer to "do these models
   understand the underlying physics, or are they pattern-matching within
   the distribution they were shown": on evidence gathered here, closer
   to the latter, with real but limited transfer within a single physical
   regime (flow boiling) and much weaker, harder-to-attribute transfer
   across regimes (flow to pool boiling).

## Practical technique takeaway (separate from the physics question)
**LoRA is the standout practical result from this stage** -- see
`data/processed/pool_boiling/README.md` for the full numbers: within
~0.01-0.04 R2 of full fine-tuning while training only 1.65-8.8% of the
parameters, on a domain with just 55 rows. For small real-experimental
domains like this, LoRA-style adaptation is a genuinely better default
than full fine-tuning, not merely a cheaper approximation of it.

## Known limitations, stated plainly
- Pool-boiling test set is n=11 -- every number above from that domain is
  from a very small sample; treat as directional.
- MoE was evaluated with an admittedly undertrained gate (pool-only
  training data); a fair MoE evaluation needs joint-regime training,
  documented as follow-up work, not done here due to time scope.
- pin-fin pool-boiling data remains entirely unused (excluded, not
  fabricated around) -- a real, still-open dataset for future work if a
  defensible diameter proxy or a genuinely separate pool-boiling schema
  is built later.

## Files
- `stage4_final_comparison.csv` -- the table above, machine-readable,
  every number traceable to its source CSV in `stage2/`, `stage3/`, or
  `pool_boiling/`.
