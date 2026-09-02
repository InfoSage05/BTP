# physics_foundation/

The physics half of the project, kept deliberately separate from the modelling code.

Everything in `unified_chf_pipeline/`, `scripts/` and `notebooks/` predicts CHF by
fitting. This folder answers the prior question: **what actually governs CHF, and what
must be true of any model that claims to predict it?**

## Contents

| File | What it is |
|---|---|
| `CHF_Physics_Foundation.md` | The master document. Every law, mechanism, closed-form correlation and constraint gathered from the reference papers and the open literature. Read this one. |
| `CHF_Physics_Foundation.docx` | Same content, formatted Times New Roman 12 / 1.5 spacing per the house style declared at the end of `outline/paper outline.docx`. For sharing with the advisor. |
| `make_docx.py` | Regenerates the `.docx` from the `.md`. Edit the markdown, never the Word file. |

```bash
python physics_foundation/make_docx.py
```

## How the document is organised

| § | Contents |
|---|---|
| 1 | The phenomenon — boiling curve, **DNB vs. dryout**, the four CHF detection criteria, primary vs. secondary parameters, the exact energy balance, the local conditions hypothesis |
| 2 | **Dimensional analysis** — Kutateladze's derivation worked through, and the full dimensionless group inventory. The backbone of the generalisation argument |
| 3 | Pool boiling — all five competing trigger mechanisms with full equations: bubble interference, **Zuber hydrodynamic instability (complete derivation)**, macrolayer dryout, hot/dry spot, interfacial lift-off |
| 4 | **Surface characteristics** — contact angle, roughness, wickability, orientation, wall thermal activity, pressure/viscosity. The physics behind the paper outline's core claim |
| 5 | Flow boiling — why 500+ correlations exist, the LUT method, **K1–K8 correction factors**, Biasi, Katto–Ohno, Shah, Hall–Mudawar, the diameter-exponent controversy |
| 6 | Geometry-specific — helical coils (centrifugal/secondary flow), mini/micro-channels, pin fins |
| 7 | What ML has and hasn't done, sourced from the primary literature rather than asserted |
| 8 | **Physical constraints any admissible model must satisfy** — 7 hard, 10 soft, 6 data-integrity. This is the operational payload |
| 9 | Proposed physics-first algorithm (discussion draft, not implemented) |
| 10 | Source inventory, including an explicit list of what still needs obtaining |

## Reading conventions

Every equation is tagged for provenance:

- **[V]** — transcribed verbatim from a primary source read during this research.
- **[S]** — structure verified, but one or more constants need checking against the original.
- **[X]** — known to be relevant but **not** transcribed, because no clean source was obtained.

Nothing was written from recollection. §10.3 lists the ten items that must be obtained
from primary sources before publication — including the W-3 and Bowring correlations,
Katto–Ohno's `C_Kc` constant, and the Rahman wickability coefficient.

## The three findings that matter most

1. **§2.2 — the correct feature space is dimensionless.** NUREG/KM-0011's own general form
   says a tube CHF correlation is a function of `(ρ_f/ρ_g, Weber-like group, x, D/D_ref)`.
   Katto–Ohno, Hall–Mudawar and Merilo are all written in exactly that space. Water at
   15 MPa and R123 at 1 MPa map to the same point there, and to wildly different points in
   raw engineering units. This is why the pipeline's `log(CHF/physics_baseline)` target
   moved strategy 3 from R² = −16.4 to +0.22, and it argues for going further.

2. **§8 — the constraint list.** Seven hard constraints (energy balance, monotonicity in
   quality, the `G → 0` pool-boiling limit, `CHF → 0` at critical pressure) and ten soft
   ones (the `P/P_c ≈ 0.35` maximum, `K ∈ [0.119, 0.157]`, the sign reversal of the
   diameter effect at low mass flux). All are testable today with no new data. The
   `ln(1 − P/P_crit)` feature that blew up Split C was ruled inadmissible by constraint C6
   *before* it was ever fitted.

3. **§8.3 D6 — the noise floor.** Inter-laboratory reproducibility on nominally identical
   test sections is about **10%** (two independent multi-lab studies, 1970 and 1984/85).
   Our random-split results sit inside that band, which means they are measuring
   source-specific bias, not physics. The unflattering surface-wise and LOSO numbers are
   the meaningful ones.

## Known gaps between the outline and the data

Recorded in §8.3 rather than buried. The paper outline builds its novelty on engineered
surface characteristics, but `unified_chf_pipeline/data/master_chf_dataset.csv` currently
has **no contact-angle column at all** and roughness at **0.6% coverage** — and Kim et
al.'s roughness correlation needs two descriptors (`R_a` *and* peak spacing `S_m`), not
one. There is also no CHF-detection-criterion column, no `D_h`/`D_e` distinction, and no
wall thickness. These should be resolved with the advisor before the results section is
written.
