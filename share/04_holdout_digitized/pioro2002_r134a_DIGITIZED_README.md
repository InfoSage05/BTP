# pioro2002_r134a_horizontal_vertical_chf_DIGITIZED.csv

**Digitized by visual reading of scatter-plot figures — not machine-precise.**
Source: Pioro, Groeneveld, Leung et al. (2002), "Comparison of CHF measurements
in horizontal and vertical tubes cooled with R-134a," Int. J. Heat Mass
Transfer 45, 4435-4450 (`docs/references/1-s2.0-S0017931002001497-main.pdf`).

## What this is
268 points read off the marker clusters in Figs. 2a, 3a, 4a, 5a, 5c, 6a, 6c,
7a, 7c of the paper — CHF (q_cr, kW/m^2) vs. critical quality (x_cr), for a
fixed D=6.92 mm R-134a tube, at 9 combinations of pressure (1.31/1.67/2.03
MPa) and mass flux (500-3000 kg/m^2 s), each split into vertical vs.
horizontal tube orientation. A few rows are tagged
`note=dryout_on_bottom` where the paper distinguishes top-of-tube vs.
bottom-of-tube dryout in a horizontal tube.

## Why it's here
This dataset adds two things nothing else in `data/raw/external/` has:
**flow orientation** (horizontal vs. vertical) as a variable, and **R-134a**
as a fluid (only the pin-fin dataset has a non-water fluid otherwise).

## How it was made (read before using for anything beyond pipeline testing)
There is no data table in the paper — the CHF values only exist as points on
printed scatter plots. Each point above was read by eye off a rendered image
of the figure, using the printed axis gridlines for calibration. This is
standard manual plot-digitization, but with real error: expect roughly
+/-5-10% noise on both axes, and the point density/spacing here reflects my
sampling of the visible marker clusters, not the paper's actual sample
spacing (the true dataset has more points per curve than are listed here).

**Do not treat this as precise experimental data.** It's suitable for what
was asked: testing/exercising a training pipeline on an additional
orientation/fluid axis, sanity-checking a model's behavior on out-of-domain
conditions, or as a rough external validation set. For any conclusion that
matters, re-digitize with a proper tool (e.g. WebPlotDigitizer) against the
original PDF, or go back to the paper's authors/underlying report for the
real numbers.

## Skipped: "An overview of measurements, data compilations and prediction
## methods..." (Groeneveld et al. 2018)
The other PDF you supplied has no extractable data. Its figures (1-4) are
2D density/coverage heatmaps -- axes are condition pairs (e.g. tube diameter
vs. L/D, pressure vs. mass flux) with color representing point *density*,
not CHF. There is no CHF axis anywhere in these plots, so there is nothing
to digitize into a (features -> CHF) row. The paper is a literature review
of 160+ CHF datasets, not a source of new data points.
