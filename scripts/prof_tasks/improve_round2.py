"""Round 2. Judged ONLY on leave-one-dataset-out, both metrics, all 8 datasets.

Round 1 verdict: none of weighting / absolute-error loss / naive stacking improved
the median. Baseline stands. Round 2 tests a fixed stacking scheme plus physics
features that were available but unused.

  stack_ind   Stacking WITH an explicit missing-indicator. Round 1 filled an
              undefined correlation with 0.0, which reads as "this arm agrees
              exactly with the backbone" -- a fabricated signal landing on exactly
              the non-water rows, where stacking lost 0.30-0.35 R2.
  we_d        Weber number on DIAMETER (Hall & Mudawar's group). Katto's is on
              heated length. Different groups; we only ever had one.
  hall        Hall & Mudawar as a fourth correlation -- verbatim-sourced in the
              repo and never used.
  bb_ens      Average Katto-anchored and LUT-anchored predictions in log space
              rather than betting on one backbone selection.
  bo          Predict the boiling number directly: Katto-Ohno and Hall-Mudawar are
              both written in that space, so it is the theory's native target.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import datasets as D, physics_arms as P, pipeline as PL, run_pipeline as RP
sys.path.insert(0, os.path.join(P.ROOT, "physics_pipeline", "scripts"))
from physics.correlations import hall_mudawar_chf

def build():
    fr = []
    for n in RP.FLOW:
        ds = D.LOADERS[n](); f = PL.prepare(ds)
        f["dataset"] = n; f["fluid_name"] = ds.fluid
        f["We_D"] = (f["G_kg_m2s"].clip(lower=1e-6) ** 2 * f["_D_m"]
                     / (f["rho_l"] * f["sigma"]))
        try:
            x = f["X"].to_numpy() if "X" in f else np.zeros(len(f))
            f["phys_hall"] = np.clip(np.asarray(hall_mudawar_chf(
                f["G_kg_m2s"].to_numpy(), f["_D_m"].to_numpy(), x,
                f["rho_l"].to_numpy(), f["rho_g"].to_numpy(),
                f["sigma"].to_numpy(), f["h_fg"].to_numpy()), float) / 1e3, 1e-3, None)
        except Exception:
            f["phys_hall"] = np.nan
        fr.append(f)
    cols = (["dataset","fluid_name","CHF_kW_m2","phys_katto","phys_biasi","phys_lut",
             "phys_hall","corr_spread","G_kg_m2s","h_fg","We_D"] + PL.DIM)
    return (pd.concat([f.reindex(columns=cols) for f in fr], ignore_index=True)
            .dropna(subset=["CHF_kW_m2","phys_katto"]).reset_index(drop=True))

def run(tr, te, feats, arms=(), indicator=False, bb_ens=False, bo=False, trees=150):
    y = tr.CHF_kW_m2.to_numpy(float); preds = []
    for bb in (["phys_katto","phys_lut"] if bb_ens else ["phys_katto"]):
        b_tr = tr[bb].to_numpy(float); b_te = te[bb].to_numpy(float)
        if not np.isfinite(b_tr).all() or not np.isfinite(b_te).all():
            continue
        b_tr = np.clip(b_tr,1e-3,None); b_te = np.clip(b_te,1e-3,None)
        f = list(feats); a, b = tr.copy(), te.copy()
        for arm in arms:
            for fr_, base in ((a,b_tr),(b,b_te)):
                v = fr_[arm].to_numpy(float)
                ok = np.isfinite(v) & (v > 0)
                fr_[f"r_{arm}"] = np.where(ok, np.log(np.clip(v,1e-3,None)/base), 0.0)
                if indicator:
                    fr_[f"has_{arm}"] = ok.astype(float)
            f.append(f"r_{arm}")
            if indicator: f.append(f"has_{arm}")
        X = np.nan_to_num(a[f].to_numpy(float),nan=0.0)
        Xe = np.nan_to_num(b[f].to_numpy(float),nan=0.0)
        if bo:
            G=tr.G_kg_m2s.clip(lower=1e-6).to_numpy(); h=tr.h_fg.to_numpy()
            t=np.log(np.clip(y*1e3/(G*h),1e-12,None))
            m=ExtraTreesRegressor(n_estimators=trees,n_jobs=-1,random_state=0).fit(X,t)
            Ge=te.G_kg_m2s.clip(lower=1e-6).to_numpy(); he=te.h_fg.to_numpy()
            preds.append(np.log(np.clip(np.exp(m.predict(Xe))*Ge*he/1e3,1e-6,None)))
        else:
            t=np.log(np.clip(y/b_tr,1e-6,None))
            m=ExtraTreesRegressor(n_estimators=trees,n_jobs=-1,random_state=0).fit(X,t)
            preds.append(np.log(b_te)+np.clip(m.predict(Xe),-np.log(10),np.log(10)))
    return np.exp(np.mean(preds,axis=0))

ARMS=("phys_biasi","phys_lut","phys_hall")
VAR=[("baseline",       dict(feats=PL.DIM)),
     ("stack_ind",      dict(feats=PL.DIM, arms=ARMS, indicator=True)),
     ("stack_noind",    dict(feats=PL.DIM, arms=ARMS, indicator=False)),
     ("we_d",           dict(feats=PL.DIM+["We_D"])),
     ("we_d+stack_ind", dict(feats=PL.DIM+["We_D"], arms=ARMS, indicator=True)),
     ("bb_ens",         dict(feats=PL.DIM, bb_ens=True)),
     ("bo_target",      dict(feats=PL.DIM, bo=True))]

df=build(); rows=[]
for held in RP.FLOW:
    tr,te=df[df.dataset!=held],df[df.dataset==held]
    y=te.CHF_kW_m2.to_numpy(float)
    for tag,kw in VAR:
        m=P.metrics(y,run(tr,te,**kw),len(tr))
        rows.append(dict(held=held,variant=tag,R2=m["R2"],w10=m["within_10pct"]*100))
        print(f"  {held:<26}{tag:<16}R2={m['R2']:+8.3f} w10={m['within_10pct']*100:5.1f}%",flush=True)
r=pd.DataFrame(rows); r.to_csv(os.path.join(P.OUT,"improve_round2.csv"),index=False)
print("\n"+"="*70)
s=r.groupby("variant").agg(median_R2=("R2","median"),median_w10=("w10","median"),
                           worst_R2=("R2","min")).round(3).sort_values("median_R2",ascending=False)
print(s.to_string())
b=r[r.variant=="baseline"].set_index("held")
for v in s.index:
    d=r[r.variant==v].set_index("held")
    print(f"  {v:<16} beats baseline R2 {int((d.R2>b.R2).sum())}/8  w10 {int((d.w10>b.w10).sum())}/8")
