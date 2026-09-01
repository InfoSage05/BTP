# Acquisition worklist — the 14 studies behind the Serrao et al. (2025) surface-CHF database

**Why this list exists.** `paper_outline.docx` is framed around CHF over *engineered surfaces* (roughness, wettability, contact angle, material, orientation). Verified on 2026-09-01: **no dataset currently in this repository contains both a mass flux/operating condition and a surface characteristic in the same row.** The large flow-boiling sets (LUT 11,592; NRC 24,443; KAERI 1,539; Zhao 1,865) have no surface features at all; the surface sets (pin-fin 175; Ali 2018 4 rows) have no flow conditions. That gap is what this worklist closes.

Serrao et al. 2025 (`1-s2.0-S0029549325001013-main.pdf`, *Nucl. Eng. Des.* 435, 113924) built a **101-point** ATF pool-boiling CHF database from these 14 studies, with exactly the feature set the outline asks for: pressure, dimensional feature, average roughness, static contact angle, surface orientation, heater thermal effusivity. Their raw 101 rows are not published ("Data will be made available on request"). Re-extracting the 14 source papers reconstructs that database independently.

**Per-study point counts are from Serrao's own Table 2.1, so the yield of each paper is known in advance.**

## Status

| # | Study | DOI | Points | Status |
|---|---|---|---|---|
| 1 | Ali et al. 2018, *Nucl. Eng. Des.* 338, 218–231 | `10.1016/j.nucengdes.2018.08.024` | 4 | ✅ **Done** → `data/raw/external/ali2018_fecral_atf_pool_boiling_chf.csv` |
| 2 | Zhang et al. 2023, *Nat. Commun.* 14, 2321 | `10.1038/s41467-023-37899-7` | 6 | ✅ **Done** (open access via PMC10122678) → `data/raw/external/zhang2023_boiling_crisis_sapphire_surfaces_chf.csv`. Roughness per surface still missing — not given numerically in the paper text. |
| 3 | He, Ali & Chen 2022, *Int. J. Heat Mass Transf.* 196, 123270 | `10.1016/j.ijheatmasstransfer.2022.123270` | **26** | ✅ **Done — 40 rows extracted (27 fully characterised), more than Serrao used** → `he2022_horizontal_tube_pool_boiling_chf.csv`. Ra/CA/CHF all validate exactly. |
| 4 | Ahn et al. 2010, *Nucl. Eng. Des.* 240(10), 3350–3360 | `10.1016/j.nucengdes.2010.07.006` | 10 | ✅ **Done — 10/10 rows, fully characterised** → `ahn2010_zircaloy4_anodized_pool_boiling_chf.csv`. All three ranges validate exactly. |
| 5 | Yeom et al. 2020, *Nucl. Eng. Des.* 370, 110919 | `10.1016/j.nucengdes.2020.110919` | 9 | ✅ **Done — 9/9 rows, fully characterised** → `yeom2020_atf_cladding_pool_boiling_chf.csv`. All three ranges validate exactly. |
| 6 | Kim, Son & Kim 2019, *Int. J. Heat Mass Transf.* 144, 118655 | `10.1016/j.ijheatmasstransfer.2019.118655` | 7 | ⚠️ **Partial** → `kim2019_cr_coated_oxidized_roughness.csv`. Roughness table (7 rows) + bare reference (CHF 705.0, CA 80.8°) are text-exact; **per-specimen CHF and contact angle exist only in Figs. 8–9**, so 1 of 7 rows is fully characterised. Needs figure digitisation to complete. |
| 7 | Fong et al. 2001, *J. Enh. Heat Transf.* 8(2), 137–146 | `10.1615/jenhheattransf.v8.i2.60` | 7 | ⬜ Need PDF (Begell House, not Elsevier — may need a different route) |
| 8 | Son, Cho & Kim 2019, *Int. J. Heat Mass Transf.* 128, 418–430 | `10.1016/j.ijheatmasstransfer.2018.09.013` | 6 | ❌ **No extractable table.** Paper reports only relative enhancements (NBHTC +24%, CHF +34% FeCrAl; CHF +27% Cr); absolute CHF, Ra and contact angle are all figure-only (Figs. 3, 6). Same pattern as Son et al. 2020. Needs figure digitisation. |
| 9 | Ali et al. 2020, *Nucl. Eng. Des.* 362, 110581 | `10.1016/j.nucengdes.2020.110581` | 6 | ✅ **Done — 8 rows (4 fully characterised)** → `ali2020_cr_coated_zirc4_ion_irradiation_chf.csv`. CHF measured on only 2 of 4 surfaces (paper states limited sample availability). |
| 10 | Kam et al. 2015, *Ann. Nucl. Energy* 76, 335–342 | `10.1016/j.anucene.2014.09.046` | 5 | ⚠️ **Partial — 6 rows (3 fully characterised)** → `kam2015_sic_cr_coated_plates_chf.csv`. Cr-coated CHF is figure-only. **Range discrepancy:** extracted CHF 940–1470 vs Serrao's quoted 660–1223 for this study — see note below. |
| 11 | He & Chen 2023, *Nucl. Eng. Des.* 410, 112374 | `10.2139/ssrn.4385755` (SSRN preprint; published version in NED 410) | 5 | ⬜ Need PDF (SSRN returned 403 to automated fetch) |
| 12 | Jo et al. 2019, *Nucl. Eng. Des.* 354, 110166 | `10.1016/j.nucengdes.2019.110166` | 4 | ⬜ Need PDF |
| 13 | Cheol Lee et al. 2019, *Ann. Nucl. Energy* 126, 350–358 | `10.1016/j.anucene.2018.11.019` | 4 | ⬜ Need PDF |
| 14 | Seo, Jeun & Kim 2015, *Exp. Therm. Fluid Sci.* 64, 42–53 | `10.1016/j.expthermflusci.2015.01.017` | 2 | ⬜ Need PDF |

**Progress: 90 rows extracted across 8 of the 14 studies, 55 of them with roughness + contact angle + CHF all present.** That already exceeds Serrao's 101-point database in usable rows for several studies, because we kept rows they appear to have filtered out.

Remaining to acquire: Fong 2001 (7), He & Chen 2023 (5), Jo 2019 (4), Cheol Lee 2019 (4), Seo 2015 (2) — 22 points across 5 papers.

### ⚠️ One validation flag — Kam et al. 2015

Every other extraction landed inside Serrao's published range. Kam 2015 did not: the paper's own text gives CHF of 1020 (bare SS), 940 (Zircaloy-4), 1230 (SiC 400 nm) and 1470 kW/m² (SiC 1 µm), but Serrao's Table 2.3 quotes 660–1223 for this study. The extracted values come verbatim from the paper's results section, so the discrepancy is upstream — either Serrao digitised Fig. 7/12 rather than using the text, or they included the Cr-coated cases (whose CHF is figure-only and described merely as "lower") and excluded the SiC 1 µm case. **Do not silently trust either number for this study**; if Kam 2015 rows matter to a final result, digitise Fig. 7 and reconcile.

## How to download

Every DOI resolves via `https://doi.org/<DOI>` — paste that and your institutional access does the rest. These are the same publishers you already pulled the previous five PDFs from, so the route is known to work. Automated fetching is *not* an option here: ScienceDirect, SSRN, and Nature's own article page all block programmatic access (403 / auth redirect), which is why this is a manual step.

Drop the PDFs anywhere in `docs/references/` under any filename — they get identified by content, not filename.

## What gets extracted from each

Per surface tested, the target columns (matching Serrao's feature set so the reconstruction is comparable):

`material` · `surface_treatment` · `roughness_Ra_um` · `static_contact_angle_deg` · `surface_orientation_deg` · `pressure_MPa` · `CHF_kW_m2` · `heater_geometry` · `dimensional_feature` · `thermal_effusivity`

## Validation method (already proven on 2 papers)

Serrao's Table 2.3 publishes the min–max range of every variable for every one of the 14 studies. So each re-extraction is checkable against an independent published range before it is trusted:

- **Ali et al. 2018** — extracted CHF 784.1–1088.7 kW/m²; Serrao's row says 784–1089. ✅
- **Zhang et al. 2023** — extracted CHF 980–2220 (Serrao: 980–2220 ✅), contact angle 0–85 (Serrao: 0–85 ✅), and the heated area 10×10 mm ÷ capillary length 2.5 mm = dimensional feature 4 (Serrao: 4 ✅). Three independent matches.

Any future extraction that does *not* land inside its Serrao range is a transcription error and must be re-checked before use.

## ⚠️ Highest-value lead — ask the advisor first, before downloading anything

**Serrao et al. 2025 cites Dr. Pattanayak's own experimental work — twice.**

> Pattanayak, B., Deswal, H., Saxena, V., Kothadia, H., 2021. "Effect of Strip Orientations and Geometry on the Critical Heat Flux in Pool Boiling." *Advances in Fluid and Thermal Engineering*, 293–303.

Serrao cites it (their §4, around the discussion of surface orientation and heating-element dimension) as supporting evidence for two of the six features in their model: **surface orientation** and the **dimensional feature**. In other words, the advisor's own experiments are already part of the evidence base that the closest prior-art paper builds on.

Why this matters more than the 12 downloads below:

1. **The raw data is one conversation away, not paywalled.** Published papers report reduced results; the advisor has the underlying measurements.
2. **It is unpublished raw data**, which is a genuine novelty asset — re-extracting others' published tables is not.
3. **It carries exactly the features the outline names**: strip orientation → `surface_orientation`, geometry → `dimensional_feature`. Pool boiling, so it merges cleanly with the Serrao-family data.
4. **It probably explains §5 of the outline.** The outline asks for "Experimental facility, test surfaces, their explanation and tabulations" and "Figure 1 — Schematic of the experimental CHF facility." That reads much less like a request to build a new rig than like an expectation of using an existing facility's data — his.

**Suggested ask:** the raw per-test CHF measurements behind the 2021 strip-orientation study — CHF, strip geometry/dimensions, orientation angle, surface material and finish, fluid, pressure — plus whether any later or unpublished runs from the same rig exist.

## Also worth doing in parallel

Email the Serrao authors for the compiled 101-point database (their Data Availability statement invites requests). Corresponding group: Dept. of Nuclear Engineering & Engineering Physics / Dept. of Industrial & Systems Engineering, University of Wisconsin–Madison. If they send it, it validates the entire re-extraction at once — and costs one email.
