# Parametric CAD models

These scripts use [CadQuery](https://github.com/CadQuery/cadquery) and model dimensions in **millimetres**. They generate STEP files for SolidWorks/AutoCAD and STL files for mesh/3D-printing workflows.

## Install CadQuery

Use Python 3.11 or 3.12 on the laptop where you will run the model. CadQuery's
current binary dependencies do not reliably support bleeding-edge Python
versions; the official project currently lists wheels for Python 3.9 through
3.12.

Then create an isolated environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install cadquery
```

If PowerShell blocks activation, run the scripts with the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe cad_models\01_hollow_box.py
.\.venv\Scripts\python.exe cad_models\02_platform.py
```

## Build sequence

```powershell
python cad_models\01_hollow_box.py
python cad_models\02_platform.py
```

The generated files are placed in `cad_models\output`:

- `01_hollow_box.step` and `01_hollow_box.stl`
- `02_platform.step` and `02_platform.stl`
- `02_insert_box.step` and `02_platform_assembly.step`

Reference/render images for each model (moved here from the repo root during the 2026-08-31 cleanup) also live in `cad_models\output`: `01_hollow_box_render.jpg`, `01_hollow_box_render_2.png`, `02_platform_render.jpg`.

## Assumptions

### Hollow box

- Outer width: 200 mm (20 cm)
- Outer breadth/depth: 200 mm (20 cm)
- Outer height: 280 mm (28 cm)
- Difference between outer and inner width/depth: 40 mm (4 cm total)
- Therefore the wall thickness is 20 mm on each side
- Floor thickness: 20 mm
- The top is open

The model intentionally uses a clean, sharp-edged solid. The pictured decorative/inset panel lines are not dimensioned, so they are not added as geometry that could be mistaken for measured features.

### Platform

The platform dimensions are assumed because the reference does not provide numerical dimensions:

- Main body: 400 x 300 x 60 mm
- Lower offset: 240 x 160 x 30 mm
- Lower offset location: centered in X, shifted toward the front by 80 mm
- Carved socket: 140 x 100 x 20 mm
- Socket clearance for insert: 2 mm per side
- Insert box: 136 x 96 x 18 mm

Change the constants at the top of either script and rerun it to regenerate the models.
