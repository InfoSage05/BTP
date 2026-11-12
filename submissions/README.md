# BTP submissions (Prof. Chirodeep Bakli)

Team: Mauryavardhan Singh (23ME10048), Mayank Nagar (23ME10049), Ayushman Paul (23MF3IM02),
Hardik Mahawar (23ME31037). Professor: Prof. Chirodeep Bakli. Supervisor: Prof. Bikash Pattanayak.

## What the professor asked for (emails, Nov 2026)

| # | Deliverable | Rules |
|---|---|---|
| 1 | Video / graphical abstract | **GIFs only, not a moving video.** One slide with minimal text. The template slide was asked for, but he later said a different layout is allowed. Examples: fyfluiddynamics.com/2015/11/ |
| 2 | Podcast | **Under 5 min**, recorded in **our own voices** (AI may clean audio, no AI voice). Must **start on the template slide with names and photos**. Goes on YouTube and may go on the institute site. |
| 3 | Report | DOCX in `templates/Report Template.docx`, due **at least 5 days before the presentation**. |
| 4 | Presentation | PPTX in `templates/Master Template.pptx`. |
| 5 | How to submit | Through the given forms, **not** email attachments. |

## Status

- `01_video_abstract/` — **done**:
  - `CHF_graphical_abstract.gif`: one-page animated poster, 1920x1080, 8 s loop.
  - `Video_Abstract_CHF.pptx`: template slide with GIFs 1–3.
  - `gif1`–`gif4`: the separate GIFs.
- Still missing: member photos (template and podcast need them).
- Not started: podcast script, report, presentation.

## Where every number comes from (scripts assert these on build)

| Claim | Source |
|---|---|
| Split C: tree R² 0.43, ridge-log R² 0.75, 2,310 test rows | `data/chf_long_clean.csv` (refit), `README.md` table |
| Within ±10%: 68 / 39 / 32 % (random / unseen group / unseen dataset, plain mean over 8 datasets) | `results/prof_tasks/FINAL_results.csv` |
| Median R²: ML alone −2.061, physics alone 0.336, physics+ML 0.708; beats physics on 7/8 (loses on Zhao) | `results/prof_tasks/audit_v2_features.csv`, set `F0_base5` |
| 28,582 records, 8 datasets, 53.2 % interval coverage | `docs/manuscript_revision_2026-11-05/_work/paper_text.md` |
| Lab identity explains 82 % of CHF variance (ATF surface set) | `docs/CONFOUNDING_FINDINGS.md` (ICC 0.815) |

Caveats to keep:
- The ±10% figures are a plain mean over datasets. Weighted by records, the unseen-dataset share is 44% (±10%) and 73% (±20%).
- Do not quote "all eight", −1276 or 0.099 (see `docs/INDEPENDENT_AUDIT_2026.md`).

## Rebuild (run from the repo root)

```bash
S=submissions/01_video_abstract/scripts
python $S/gif1_what_is_chf.py submissions/01_video_abstract/gif1_what_is_chf.gif
python $S/gif2_extrapolation.py data/chf_long_clean.csv submissions/01_video_abstract/gif2_extrapolation.gif
python $S/gif3_physics_anchor.py results/prof_tasks/audit_v2_features.csv submissions/01_video_abstract/gif3_physics_anchor.gif
python $S/gif4_unseen_dataset.py results/prof_tasks/FINAL_results.csv submissions/01_video_abstract/gif4_unseen_dataset.gif
python $S/dashboard_abstract.py . submissions/01_video_abstract/CHF_graphical_abstract.gif
python $S/build_slide.py "submissions/templates/Master Template.pptx" submissions/01_video_abstract submissions/01_video_abstract/Video_Abstract_CHF.pptx
```

Needs: numpy, pandas, scikit-learn, matplotlib, pillow, python-pptx.

The GIFs were rendered on macOS in Avenir Next. On Linux, matplotlib falls back to DejaVu Sans, which is wider, so check text fit before replacing the committed GIFs.

The template's background and IIT logo were patched (pre-faded background, PNG logo) so the slide looks the same in Keynote and PowerPoint; see `build_slide.py`.
