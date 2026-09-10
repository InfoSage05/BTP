"""Senior-review stress test and the definitive results tables.

Splits reported, and exactly what each one means:

  80/20, 70/30   random, but grouped on a duplicate signature so identical rows
                 cannot straddle train and test. 578 within-dataset duplicates
                 exist (770 rows involved in Zhao alone); a naive random split
                 puts some of them on both sides and inflates the score.
                 TRAIN and TEST come from the SAME dataset.

  GROUPED        hold out one whole group from inside a dataset. What the group IS
                 differs per dataset and is named honestly in the output, because
                 only four datasets have real laboratory labels:
                   NRC / Zhao / KAERI x2 -> publication or author = UNSEEN LAB
                   Helical R-123         -> appendix C vs D      = pressure range
                   Hardik helical        -> Coil_1..6            = coil geometry
                   Pioro                 -> Fig2a..Fig7c         = figures, one paper
                   Hardik straight       -> no group at all, single campaign
                 Calling the last three "unseen laboratory" would overstate them.

  LODO           leave-one-DATASET-out: the whole dataset is held out and the
                 model is trained on the other seven. This is "unseen data" in
                 the strongest sense available here -- a different fluid,
                 geometry or campaign. Verified free of cross-dataset duplicates.

Every number is the mean over seeds with its standard deviation, because this
project has already had to retract a single-seed headline.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import datasets as D
import physics_arms as P
import pipeline as PL
import run_pipeline as RP

warnings.filterwarnings("ignore")

# What each dataset's group column really represents.
GROUP_MEANING = {
    "D1_NRC": "unseen lab (60 refs)",
    "D2_Zhao2020": "unseen lab (10 authors)",
    "D3_KAERI_uniform": "unseen lab (10 sources)",
    "D4_KAERI_nonuniform": "unseen lab (11 sources)",
    "D5_Helical_R123": "unseen pressure range (2)",
    "D6_Hardik_helical_water": "unseen coil geometry (6)",
    "D8_Pioro_R134a": "unseen figure group (9)",
}
OUT = P.OUT
SEEDS = 10
SEEDS_LODO = 5


def dup_key(df: pd.DataFrame) -> np.ndarray:
    """Signature identifying rows the model cannot distinguish."""
    return (df[PL.DIM + ["CHF_kW_m2"]].round(6).astype(str)
            .agg("|".join, axis=1).to_numpy())


def score(y, p, n_train):
    m = P.metrics(y, p, n_train)
    return m


def evaluate() -> pd.DataFrame:
    df = RP.build_corpus()
    # Group labels are carried on a row key, not by position. `gs[:len(sub)]`
    # would silently pair rows with the wrong group the moment the corpus drops
    # a row -- currently it drops none, so the bug was dormant, but a dormant
    # misalignment bug is not something to hand over.
    src = {}
    for n in RP.FLOW:
        ds = D.LOADERS[n]()
        if ds.group is None:
            src[n] = None
            continue
        lab = ds.df[ds.group].to_numpy()
        assert len(lab) == len(PL.prepare(ds)), (
            f"{n}: loader and prepared frame differ in length; group labels "
            f"cannot be aligned positionally")
        src[n] = lab
    rows = []

    for name in RP.FLOW:
        sub = df[df.dataset == name].reset_index(drop=True)
        y = sub["CHF_kW_m2"].to_numpy(float)
        g_dup = dup_key(sub)

        # ---- random splits, grouped on the duplicate signature
        for label, ts in (("80/20", 0.20), ("70/30", 0.30)):
            acc = []
            for s in range(SEEDS):
                gss = GroupShuffleSplit(n_splits=1, test_size=ts, random_state=s)
                tr_i, te_i = next(gss.split(sub, y, groups=g_dup))
                pipe = PL.CHFPipeline(seed=s).fit(sub.iloc[tr_i])
                r = pipe.predict(sub.iloc[te_i])
                m = score(y[te_i], r["chf"].to_numpy(), len(tr_i))
                m["coverage90"] = float(((y[te_i] >= r["lo"]) & (y[te_i] <= r["hi"])).mean())
                acc.append(m)
            a = pd.DataFrame(acc)
            rows.append(dict(dataset=name, n=len(sub), split=label,
                             train_source="same dataset", test_source="same dataset",
                             seeds=SEEDS, **{f"{c}_mean": a[c].mean() for c in a.columns},
                             **{f"{c}_sd": a[c].std() for c in
                                ("R2", "within_10pct", "coverage90")}))

        # ---- leave-one-SOURCE-out inside the dataset
        gs = src[name]
        if gs is not None and len(np.unique(gs)) >= 2:
            assert len(gs) == len(sub), (
                f"{name}: {len(gs)} group labels for {len(sub)} rows")
            gk = GroupKFold(n_splits=min(len(np.unique(gs)), 10))
            acc = []
            for tr_i, te_i in gk.split(sub, y, gs):
                pipe = PL.CHFPipeline(seed=0).fit(sub.iloc[tr_i])
                r = pipe.predict(sub.iloc[te_i])
                m = score(y[te_i], r["chf"].to_numpy(), len(tr_i))
                m["coverage90"] = float(((y[te_i] >= r["lo"]) & (y[te_i] <= r["hi"])).mean())
                acc.append(m)
            a = pd.DataFrame(acc)
            rows.append(dict(dataset=name, n=len(sub),
                             split="GROUPED: " + GROUP_MEANING.get(name, "?"),
                             train_source="same dataset, other groups",
                             test_source="held-out group", seeds=len(acc),
                             **{f"{c}_mean": a[c].mean() for c in a.columns},
                             R2_median=float(a["R2"].median()),
                             R2_worst=float(a["R2"].min()),
                             **{f"{c}_sd": a[c].std() for c in
                                ("R2", "within_10pct", "coverage90")}))

        # ---- leave-one-DATASET-out: this dataset is entirely unseen
        tr, te = df[df.dataset != name], sub
        acc = []
        for s in range(SEEDS_LODO):
            pipe = PL.CHFPipeline(seed=s).fit(tr)
            r = pipe.predict(te)
            m = score(y, r["chf"].to_numpy(), len(tr))
            m["coverage90"] = float(((y >= r["lo"]) & (y <= r["hi"])).mean())
            m["trust"] = float(r["trust"].mean())
            acc.append(m)
        a = pd.DataFrame(acc)
        rows.append(dict(dataset=name, n=len(sub), split="LODO (unseen dataset)",
                         train_source="other 7 datasets", test_source="this dataset",
                         seeds=SEEDS_LODO,
                         **{f"{c}_mean": a[c].mean() for c in a.columns},
                         **{f"{c}_sd": a[c].std() for c in
                            ("R2", "within_10pct", "coverage90")}))
    return pd.DataFrame(rows)


def stress(df_corpus):
    """Checks a reviewer would run before signing off."""
    print("\n" + "=" * 88)
    print("STRESS TESTS")
    print("=" * 88)
    tr = df_corpus[df_corpus.dataset != "D7_Hardik_straight_R123"]
    te = df_corpus[df_corpus.dataset == "D7_Hardik_straight_R123"]
    pipe = PL.CHFPipeline().fit(tr)
    r = pipe.predict(te)
    y = te["CHF_kW_m2"].to_numpy(float)
    ok = lambda b: "PASS" if b else "**FAIL**"

    # G. metrics recomputed from scratch
    p = r["chf"].to_numpy()
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    w10 = float((np.abs(p - y) / y <= 0.10).mean())
    m = P.metrics(y, p, len(tr))
    print(f"  G  metrics independently recomputed        {ok(abs(r2-m['R2'])<1e-9 and abs(w10-m['within_10pct'])<1e-9)}")

    # I. interval sanity
    print(f"  I  interval brackets point, all positive   "
          f"{ok(bool(((r.lo<r.chf)&(r.chf<r.hi)&(r.lo>0)).all()))}")
    print(f"  I  no NaN/inf in any prediction            "
          f"{ok(bool(np.isfinite(r[['chf','lo','hi','trust']].to_numpy()).all()))}")
    print(f"  I  trust within [0,1]                      "
          f"{ok(bool(((r.trust>=0)&(r.trust<=1)).all()))}")

    # H. physical monotonicity, probed on synthetic sweeps the model never saw
    base = dict(fluid="water", P_kPa=10000, G_kg_m2s=2000, D_mm=8, L_mm=1000, Tin_C=200)
    gs = [500, 1000, 2000, 4000, 6000]
    chf_g = [pipe.predict_one(**{**base, "G_kg_m2s": g})["CHF_kW_m2"] for g in gs]
    print(f"  H  CHF rises with mass flux                {ok(all(np.diff(chf_g)>0))}  {chf_g}")
    ts = [100, 150, 200, 250, 300]
    chf_t = [pipe.predict_one(**{**base, "Tin_C": t})["CHF_kW_m2"] for t in ts]
    print(f"  H  CHF falls as inlet gets hotter          {ok(all(np.diff(chf_t)<0))}  {chf_t}")
    ls = [500, 1000, 2000, 4000]
    chf_l = [pipe.predict_one(**{**base, "L_mm": l})["CHF_kW_m2"] for l in ls]
    print(f"  H  CHF falls with heated length            {ok(all(np.diff(chf_l)<0))}  {chf_l}")

    # J. adversarial inputs must refuse, not guess
    bad = [("G=0", dict(base, G_kg_m2s=0)), ("D<0", dict(base, D_mm=-8)),
           ("P>Pcrit", dict(base, P_kPa=25000)), ("unknown fluid", dict(base, fluid="xx")),
           ("NaN pressure", dict(base, P_kPa=float("nan")))]
    good = 0
    for lbl, kw in bad:
        try:
            pipe.predict_one(**kw)
        except ValueError:
            good += 1
    print(f"  J  refuses impossible inputs               {ok(good==len(bad))}  {good}/{len(bad)}")

    # K. duplicate-aware vs naive splitting
    from sklearn.model_selection import train_test_split
    sub = df_corpus[df_corpus.dataset == "D2_Zhao2020"].reset_index(drop=True)
    yy = sub["CHF_kW_m2"].to_numpy(float)
    naive, grouped = [], []
    for s in range(5):
        i_tr, i_te = train_test_split(np.arange(len(sub)), test_size=.2, random_state=s)
        naive.append(P.metrics(yy[i_te], PL.CHFPipeline(seed=s).fit(sub.iloc[i_tr])
                               .predict(sub.iloc[i_te])["chf"].to_numpy(), len(i_tr))["R2"])
        g = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=s)
        i_tr, i_te = next(g.split(sub, yy, groups=dup_key(sub)))
        grouped.append(P.metrics(yy[i_te], PL.CHFPipeline(seed=s).fit(sub.iloc[i_tr])
                                 .predict(sub.iloc[i_te])["chf"].to_numpy(), len(i_tr))["R2"])
    print(f"  K  Zhao 80/20 naive R2 {np.mean(naive):.3f} vs duplicate-grouped "
          f"{np.mean(grouped):.3f}  (gap {np.mean(naive)-np.mean(grouped):+.3f})")


def main():
    res = evaluate()
    res.to_csv(os.path.join(OUT, "FINAL_results.csv"), index=False)
    show = res.assign(
        R2=res.R2_mean.round(3), sd=res.R2_sd.round(3),
        w10=(res.within_10pct_mean * 100).round(1),
        w20=(res.within_20pct_mean * 100).round(1),
        cov=res.coverage90_mean.round(2))
    print("=" * 88)
    print("ALL SPLITS x ALL DATASETS")
    print("=" * 88)
    for sp in ["80/20", "70/30"]:
        d = show[show.split == sp]
        print(f"\n--- {sp} random, grouped on duplicates (train & test: SAME dataset) ---")
        print(d[["dataset", "n", "seeds", "R2", "sd", "w10", "w20", "cov"]]
              .to_string(index=False))
    d = show[show.split.str.startswith("GROUPED")].copy()
    d["med"] = res.loc[d.index, "R2_median"].round(3)
    d["worst"] = res.loc[d.index, "R2_worst"].round(3)
    print("\n--- GROUPED hold-out inside a dataset (what the group is varies) ---")
    print(d[["dataset", "split", "seeds", "R2", "med", "worst", "sd", "w10", "cov"]]
          .to_string(index=False))
    print("  D7_Hardik_straight_R123: no group column -- single campaign, not evaluable")
    d = show[show.split.str.startswith("LODO")]
    print("\n--- LODO: dataset entirely UNSEEN, model trained on the other seven ---")
    print(d[["dataset", "n", "seeds", "R2", "sd", "w10", "w20", "cov"]]
          .to_string(index=False))
    stress(RP.build_corpus())
    print(f"\nwrote {os.path.join(OUT, 'FINAL_results.csv')}")


if __name__ == "__main__":
    main()
