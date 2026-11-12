CHF MANUSCRIPT REVISION PACKAGE
Prepared 5 November 2026

Start with CHF_revised_manuscript.docx and AUTHOR_ACTIONS_BEFORE_SUBMISSION.txt.
The manuscript is a revised research draft. Pending author declarations and unresolved scientific validation questions prevent certifying it as submission-ready.

CONTENTS
- CHF_revised_manuscript.docx: editable manuscript, six tables, six figures and nine editable equations.
- Supplementary_material.docx: supporting benchmarks, provenance and exclusions.
- Highlights.docx: five journal-length highlights.
- figures/: six vector PDF figures, six 1000 dpi PNG versions, plot data and rendering script.
- supporting_data/: seven unchanged archived summary CSVs used for numerical tables and figures.
- experiment_manifest.csv: source paths, experiment families and checksums.
- figure_captions.txt: separate captions.
- AUTHOR_ACTIONS_BEFORE_SUBMISSION.txt: missing declarations and scientific issues requiring resolution.

FORMATTING
Body text is 12 pt Times New Roman. Figure lettering is 10 pt Times New Roman at 165.1 mm (6.5 inches) width. Scaling figures will change the lettering size. The 10 pt choice is for readability and is not a verified recollection of the professor's requested size. Use vector PDFs where accepted by the journal. The PDFs embed the Times New Roman fonts.

FIGURE REPRODUCTION
The plotting script reads figures/evidence.json and does not access or modify the research codebase. It requires Python with NumPy, ReportLab and pypdfium2, and installed Times New Roman fonts at the macOS locations specified in the script. Font files are not redistributed. Running the script regenerates the six figures and creates previews in a local _work folder. Figure 2 is a method schematic; the other figures are drawn from recorded numerical results. No measurements were invented.

REVIEW STATUS
All manuscript, supplementary and highlights pages were rendered and visually reviewed. Numbers were checked against the archived summary files; this is not a rerun of model training. The original manuscript and research codebase were not changed. The underlying scientific limitations are retained in the manuscript.
