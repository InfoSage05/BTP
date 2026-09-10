"""Test candidate improvements. Judged ONLY on unseen datasets, never on random
splits, so nothing here can be tuned into looking good."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import pipeline as PL, run_pipeline as RP, physics_arms as P

df = RP.build_corpus()

def variant(tr, te, weighted=False, loss="squared_error", stack=False, trees=150):
    y = tr.CHF_kW_m2.to_numpy(float)
    bb = "phys_katto"
    base_tr = np.clip(tr[bb].to_numpy(float), 1e-3, None)
    base_te = np.clip(te[bb].to_numpy(float), 1e-3, None)
    feats = list(PL.DIM)
    tr2, te2 = tr, te
    if stack:
        # every correlation's opinion, expressed relative to the backbone
        for arm in ["phys_biasi", "phys_lut"]:
            tr2 = tr2.assign(**{f"r_{arm}": np.log(
                np.clip(tr2[arm].to_numpy(float), 1e-3, None) / base_tr)})
            te2 = te2.assign(**{f"r_{arm}": np.log(
                np.clip(te2[arm].to_numpy(float), 1e-3, None) / base_te)})
            feats.append(f"r_{arm}")
    X = np.nan_to_num(tr2[feats].to_numpy(float), nan=0.0)
    Xe = np.nan_to_num(te2[feats].to_numpy(float), nan=0.0)
    t = np.log(np.clip(y / base_tr, 1e-6, None))
    w = None
    if weighted:                      # inverse dataset size: NRC is 85% of rows
        cnt = tr.dataset.map(tr.dataset.value_counts()).to_numpy(float)
        w = 1.0 / cnt
        w = w / w.mean()
    m = ExtraTreesRegressor(n_estimators=trees, n_jobs=-1, random_state=0,
                            criterion=loss)
    m.fit(X, t, sample_weight=w)
    return base_te * np.exp(np.clip(m.predict(Xe), -np.log(10), np.log(10)))

VARIANTS = [("baseline", {}),
            ("weighted", dict(weighted=True)),
            ("abs_loss", dict(loss="absolute_error")),
            ("stacked", dict(stack=True)),
            ("weighted+stacked", dict(weighted=True, stack=True)),
            ("weighted+abs", dict(weighted=True, loss="absolute_error")),
            ("all_three", dict(weighted=True, stack=True, loss="absolute_error"))]

rows = []
for held in RP.FLOW:
    tr, te = df[df.dataset != held], df[df.dataset == held]
    y = te.CHF_kW_m2.to_numpy(float)
    rows.append(dict(held=held, variant="pure_physics",
                     R2=P.metrics(y, te.phys_katto.to_numpy(float), 0)["R2"],
                     w10=P.metrics(y, te.phys_katto.to_numpy(float), 0)["within_10pct"]*100))
    for tag, kw in VARIANTS:
        p = variant(tr, te, **kw)
        m = P.metrics(y, p, len(tr))
        rows.append(dict(held=held, variant=tag, R2=m["R2"], w10=m["within_10pct"]*100))
        print(f"  {held:<26}{tag:<18}R2={m['R2']:+7.3f}  w10={m['within_10pct']*100:5.1f}%",
              flush=True)

r = pd.DataFrame(rows)
r.to_csv("../../results/prof_tasks/improve_ablation.csv", index=False)
print("\n" + "=" * 78)
print("MEDIAN ACROSS THE 8 UNSEEN DATASETS")
print("=" * 78)
s = (r.groupby("variant").agg(median_R2=("R2", "median"), median_w10=("w10", "median"),
                              worst_R2=("R2", "min"), wins_w10=("w10", "count"))
     .drop(columns="wins_w10").round(3).sort_values("median_R2", ascending=False))
print(s.to_string())
base = r[r.variant == "baseline"].set_index("held")
for v in [x[0] for x in VARIANTS] + ["pure_physics"]:
    d = r[r.variant == v].set_index("held")
    print(f"  {v:<18} beats baseline on R2 {int((d.R2>base.R2).sum())}/8   "
          f"on +/-10% {int((d.w10>base.w10).sum())}/8")
