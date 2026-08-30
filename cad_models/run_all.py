"""Run both model builders in the intended sequence."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_script(filename: str) -> None:
    path = ROOT / filename
    spec = spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    run_script("01_hollow_box.py")
    run_script("02_platform.py")
    print("All CAD models generated in cad_models/output")

