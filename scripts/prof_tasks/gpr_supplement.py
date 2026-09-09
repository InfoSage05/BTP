"""Add the GPR rows the main grid skipped on the three largest datasets.

The main run was launched while build_models() still excluded GPR above 8000
rows, so D9/D1/M have no GPR row. GPR keeps winning on the small datasets, so
leaving it out of the biggest three would bias the Q2 "consistent everywhere"
ranking against it.

Its training set is subsampled to 2000 rows (fit_predict does this), which is
how a Matern GPR has to be run at this scale anyway. The output is tagged
`GPR_Matern_sub2000` rather than `GPR_Matern`, because a GPR that saw 2000 of
24443 rows is not the same model as one that saw all 44 rows of a small set,
and the two should not be silently pooled under one name.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

import datasets as D
import run_grid as R

TARGETS = ["D9_LUT2006", "D1_NRC", "M_merged_flow"]


def only_gpr():
    return {"GPR_Matern_sub2000": GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-3),
        normalize_y=True, random_state=0)}


def main():
    R.build_models = only_gpr
    R.SCALE_SENSITIVE = R.SCALE_SENSITIVE | {"GPR_Matern_sub2000"}
    # route the subsampling rule at the new name
    orig = R.fit_predict

    def patched(name, model, Xtr, ytr, Xte, num, cat, seed):
        return orig("GPR_Matern" if name.startswith("GPR") else name,
                    model, Xtr, ytr, Xte, num, cat, seed)
    R.fit_predict = patched

    rows = []
    for name in TARGETS:
        ds = D.LOADERS[name]()
        print(f"=== {ds.name} ({ds.n} rows) ===", flush=True)
        rows += R.run_dataset(ds, lambda m: print(m, flush=True))

    new = pd.DataFrame(rows)
    path = os.path.join(R.OUT, "grid_results.csv")
    old = pd.read_csv(path)
    combined = pd.concat([old, new], ignore_index=True)
    combined.to_csv(path, index=False)
    print(f"\nadded {len(new)} GPR rows -> grid_results.csv now {len(combined)} rows")


if __name__ == "__main__":
    main()
