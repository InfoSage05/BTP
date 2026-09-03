# data/raw/fine_tuning/

**Stage 2: real, distinct-domain datasets — fine-tune here.**

Every file here is genuinely tabulated/measured data (not digitized), but
each one differs from the pretraining core in fluid, geometry, or heating
regime — exactly what transfer-learning fine-tuning is for. Size varies a
lot (55 to 1,865 rows); that's fine, homogeneity with the core is what
would have excluded a file from this folder, not row count.

| File | Rows | What makes it a distinct domain |
|---|---|---|
| `hardik2016_helical_coils_r123_lowpressure_chf.csv` | 156 | R123 fluid, helical coil geometry. From Hardik & Prabhu, Appl. Thermal Eng. 112 (2017) 1223-1239; row count verified exactly against the paper's own per-coil totals. |
| `hardik2017_straight_tubes_r123_chf.csv` | 55 | R123 fluid, straight tubes, different pressure/geometry range than the core. From Hardik, Kumar & Prabhu, IJHMT 113 (2017) 466-481; verified against the paper's stated total. |
| `helical_coil_r123_appendixCD.csv` (+ source PDF) | 257 | R123 fluid, helical coil — same fluid/geometry family as the two Hardik sets above. |
| `pinfin_chf_water_fc72.csv` | 175 | Different fluid (FC-72) *and* different geometry (pin-fin surfaces); only dataset with surface-condition features. |
| `kaeri_tr1665_uniform_chf.csv` (+ `.xml`) | 651 | Water/tubes, but a distinct test campaign — uniform heating. |
| `kaeri_tr1665_nonuniform_chf.csv` (+ `.xml`) | 888 | Water/tubes, but non-uniform axial heating profile — a physical regime the core LUT doesn't cover. |
| `zhao2020_chf_flowboiling_tubes.csv` | 1,865 | Multi-source tube compilation, not a single homogeneous campaign like the pretraining core. |

Use these to fine-tune (or evaluate transfer learning) after pretraining
on `data/raw/pretraining/`, one target domain at a time.
