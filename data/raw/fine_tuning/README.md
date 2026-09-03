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

---

## Engineered-surface (pool boiling) extractions

Added on the `Cleaned_Main` line. These are ATF / engineered-surface pool-boiling
datasets extracted by hand from 11 source papers — the subject of
`docs/manuscript/paper_outline.docx`. They differ from everything above in that
they carry **surface characteristics** (roughness, static contact angle,
orientation, material) rather than flow conditions: pool boiling has no mass
flux, so `G` and `X` are genuinely absent, not missing.

Every extraction was validated against the independently published parameter
ranges in Serrao et al. (2025), *Nucl. Eng. Des.* 435, 113924, before being
trusted. Seven of eight matched exactly; the one that did not is flagged.

| File | Rows | Source | Range check vs Serrao 2025 |
|---|---|---|---|
| `he2022_horizontal_tube_pool_boiling_chf.csv` | 40 | He, Ali & Chen, IJHMT 196 (2022) 123270 | Ra 1.55–25.21 ✅ · CA 54–82 ✅ · CHF 381–922 ✅ |
| `ahn2010_zircaloy4_anodized_pool_boiling_chf.csv` | 10 | Ahn et al., NED 240 (2010) 3350 | Ra 0.05–0.32 ✅ · CA 0–49 ✅ · CHF 1004–1924 ✅ |
| `yeom2020_atf_cladding_pool_boiling_chf.csv` | 9 | Yeom et al., NED 370 (2020) 110919 | Ra 0.07–5.72 ✅ · CA 33–107 ✅ · CHF 454–931 ✅ |
| `ali2020_cr_coated_zirc4_ion_irradiation_chf.csv` | 8 | Ali et al., NED 362 (2020) 110581 | CA 68–83 ✅ · CHF lower bound 677 ✅ |
| `kim2019_cr_coated_oxidized_roughness.csv` | 7 | Kim, Son & Kim, IJHMT 144 (2019) 118655 | Ra 0.093–0.107 ✅ (per-specimen CHF is figure-only) |
| `kam2015_sic_cr_coated_plates_chf.csv` | 6 | Kam et al., Ann. Nucl. Energy 76 (2015) 335 | Ra ✅ · CA 19–81 ✅ · **CHF ⚠️ disagrees with Serrao — see below** |
| `zhang2023_boiling_crisis_sapphire_surfaces_chf.csv` | 6 | Zhang et al., *Nat. Commun.* 14 (2023) 2321 (open access) | CA 0–85 ✅ · CHF 980–2220 ✅ (Ra not published) |
| `jo2019_atf_coating_pool_boiling_chf.csv` | 4 | Jo et al., NED 354 (2019) 110166 | Ra 0.067–0.12 ✅ · CA 64–89 ✅ · CHF 571–709 ✅ |
| `cheollee2019_zrsi2_coated_zircaloy_chf.csv` | 4 | Cheol Lee et al., Ann. Nucl. Energy 126 (2019) 350 | Ra 0.154–0.207 ✅ · CA 0–65 ✅ · CHF 804–1004 ✅ |
| `ali2018_fecral_atf_pool_boiling_chf.csv` | 4 | Ali et al., NED 338 (2018) 218 | CHF 784–1089 ✅ |
| `seo2015_zircaloy_sic_cladding_chf.csv` | 2 | Seo, Jeun & Kim, Exp. Therm. Fluid Sci. 64 (2015) 42 | Ra 0.088/0.107 ✅ · CA 85/93 ✅ · CHF 635/1037 ✅ |

Consolidated into `data/processed/surface_master.csv` (100 rows, 11 publications;
65 have roughness + contact angle + CHF all present).

**Kam 2015 discrepancy — do not silently trust either number.** The paper's own
text gives CHF of 1020 (bare SS), 940 (Zircaloy-4), 1230 (SiC 400 nm) and
1470 kW/m² (SiC 1 µm). Serrao's Table 2.3 quotes 660–1223 for the same study.
The values here are verbatim from the paper, so the disagreement is upstream.
Resolving it needs Fig. 7 digitised.

### Reference-only files in this folder (no point-level CHF)

`serrao2025_atf_surface_chf_source_summary.csv` (14 per-study ranges),
`liang2020_nanofluid_chf_pool_boiling_source_summary.csv` (62 studies, row counts
sum to exactly 431 as the paper states), `son2020_cr_cral_*` (surface
characterisation; the paper reports only relative CHF enhancement, never
absolute kW/m²), `ramakrishnan2026_copper_microchannel_ageing_chf.csv` (10 rows,
flow boiling, no per-row roughness). These are provenance/target lists, not
training data.
