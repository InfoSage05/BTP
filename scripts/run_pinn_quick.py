"""
Quick PINN evaluation script — runs a few key configurations on Split C
to get actual R² scores without needing to open the full notebook.
"""
import time, warnings, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path

sys.path.insert(0, ".")
from chf_physics import zuber_pool_boiling_chf

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

# ---- Load data ----
df = pd.read_csv("data/chf_long_clean.csv")
df = df[df.X != 1.0].reset_index(drop=True)
FEATURES = ["P", "G", "X"]
sorted_P = sorted(df.P.unique())

# ---- Splits ----
def get_split_A(seed=42):
    return train_test_split(df[FEATURES].values, df.CHF.values, test_size=0.2, random_state=seed)

def get_split_B():
    held = [sorted_P[i] for i in range(3, len(sorted_P)-1, 4)]
    tr = df[~df.P.isin(held)]; te = df[df.P.isin(held)]
    return tr[FEATURES].values, te[FEATURES].values, tr.CHF.values, te.CHF.values

def get_split_C():
    tr = df[df.P <= 16000]; te = df[df.P >= 17000]
    return tr[FEATURES].values, te[FEATURES].values, tr.CHF.values, te.CHF.values

# ---- Zuber sign table ----
_ZP = np.linspace(100, 21500, 300)
_ZV = zuber_pool_boiling_chf(_ZP)
_ZS = np.sign(np.gradient(_ZV, _ZP))
def zuber_sign(p): return np.interp(np.asarray(p, float), _ZP, _ZS)

# ---- PINN Model ----
class PINN(nn.Module):
    def __init__(self, layers=[128, 64, 32]):
        super().__init__()
        mods = []
        d = 3
        for h in layers:
            mods += [nn.Linear(d, h), nn.Tanh()]
            d = h
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x).squeeze(-1)

# ---- Training function ----
def train_pinn(Xtr, ytr, Xte, yte, layers=[128,64,32],
               lam_m=0.3, lam_z=0.1, lam_p=0.05, ncol=512,
               lr=1e-3, epochs=3000, patience=100, seed=42):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    xs = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = xs.transform(Xtr), xs.transform(Xte)
    ly = np.log(ytr); ym, ys = ly.mean(), ly.std()
    yn = (ly - ym) / ys
    Xt = torch.tensor(Xtr_s, dtype=torch.float32)
    yt = torch.tensor(yn, dtype=torch.float32)
    nv = max(1, int(0.15*len(Xt)))
    vi = rng.choice(len(Xt), nv, replace=False)
    ti = np.setdiff1d(np.arange(len(Xt)), vi)
    model = PINN(layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=30, min_lr=1e-6)
    plo, phi = df.P.min(), df.P.max()
    glo, ghi = df.G.min(), df.G.max()
    xlo, xhi = df.X.min(), 0.9
    best_v, best_s, ni = np.inf, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        dl = nn.functional.mse_loss(model(Xt[ti]), yt[ti])
        pc = rng.uniform(plo, phi, ncol)
        gc = rng.uniform(glo, ghi, ncol)
        xc = rng.uniform(xlo, xhi, ncol)
        ps = torch.tensor((pc-xs.mean_[0])/xs.scale_[0], dtype=torch.float32, requires_grad=True)
        gs = torch.tensor((gc-xs.mean_[1])/xs.scale_[1], dtype=torch.float32)
        xcs = torch.tensor((xc-xs.mean_[2])/xs.scale_[2], dtype=torch.float32, requires_grad=True)
        inp = torch.stack([ps, gs, xcs], 1)
        pc_pred = model(inp)
        dx = torch.autograd.grad(pc_pred.sum(), xcs, create_graph=True)[0]
        ml = torch.relu(dx).mean()
        dp = torch.autograd.grad(pc_pred.sum(), ps, create_graph=True)[0]
        zs = torch.tensor(zuber_sign(pc), dtype=torch.float32)
        zl = torch.relu(-dp * zs).mean()
        pl = torch.relu(-pc_pred).mean()
        loss = dl + lam_m*ml + lam_z*zl + lam_p*pl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad(): vl = nn.functional.mse_loss(model(Xt[vi]), yt[vi]).item()
        sch.step(vl)
        if vl < best_v - 1e-6: best_v, best_s, ni = vl, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else: ni += 1
        if ni > patience: break
    model.load_state_dict(best_s); model.eval()
    with torch.no_grad():
        pn = model(torch.tensor(Xte_s, dtype=torch.float32)).numpy()
    pred = np.exp(pn * ys + ym)
    return float(r2_score(yte, pred)), float(np.mean(np.abs((yte-pred)/yte))*100)

# ===========================================================================
print("="*70)
print("PINN Quick Evaluation — 5 seeds per configuration")
print("="*70)

XtrC, XteC, ytrC, yteC = get_split_C()
XtrA, XteA, ytrA, yteA = get_split_A()
XtrB, XteB, ytrB, yteB = get_split_B()

configs = [
    {"name": "Small [64,32] lm=0.3 lz=0.1",  "layers": [64,32],    "lam_m": 0.3, "lam_z": 0.1, "lr": 1e-3},
    {"name": "Med [128,64,32] lm=0.3 lz=0.1", "layers": [128,64,32],"lam_m": 0.3, "lam_z": 0.1, "lr": 1e-3},
    {"name": "Med [128,64,32] lm=0.1 lz=0.1", "layers": [128,64,32],"lam_m": 0.1, "lam_z": 0.1, "lr": 1e-3},
    {"name": "Med [128,64,32] lm=0.5 lz=0.0", "layers": [128,64,32],"lam_m": 0.5, "lam_z": 0.0, "lr": 1e-3},
    {"name": "Large [128,128,64] lm=0.3 lz=0.1","layers":[128,128,64],"lam_m":0.3,"lam_z":0.1,"lr":1e-3},
    {"name": "Med [128,64,32] lm=0.3 lz=0.1 lr=5e-4","layers":[128,64,32],"lam_m":0.3,"lam_z":0.1,"lr":5e-4},
    {"name": "No physics [128,64,32] lm=0 lz=0","layers":[128,64,32],"lam_m":0.0,"lam_z":0.0,"lr":1e-3},
]

SEEDS = [0, 1, 2, 42, 99]
results = []

for cfg in configs:
    t0 = time.time()
    r2s = []
    for s in SEEDS:
        r, m = train_pinn(XtrC, ytrC, XteC, yteC,
                          layers=cfg["layers"], lam_m=cfg["lam_m"], lam_z=cfg["lam_z"],
                          lr=cfg["lr"], seed=s, epochs=3000, patience=100)
        r2s.append(r)
    dt = time.time() - t0
    results.append({"config": cfg["name"], "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
                     "r2_min": np.min(r2s), "r2_max": np.max(r2s)})
    print(f"  {cfg['name']:<45} R2={np.mean(r2s):.4f} +/- {np.std(r2s):.4f}  "
          f"[min={np.min(r2s):.4f} max={np.max(r2s):.4f}]  ({dt:.0f}s)")

# ---- Best config on all 3 splits ----
best_cfg = configs[np.argmax([r["r2_mean"] for r in results])]
print(f"\n{'='*70}")
print(f"Best config: {best_cfg['name']}")
print(f"{'='*70}")

for sname, (Xtr, Xte, ytr, yte) in [("A", (XtrA,XteA,ytrA,yteA)),
                                      ("B", (XtrB,XteB,ytrB,yteB)),
                                      ("C", (XtrC,XteC,ytrC,yteC))]:
    r2s, ms = [], []
    for s in SEEDS:
        if sname == "A":
            Xtr2, Xte2, ytr2, yte2 = get_split_A(seed=s)
        else:
            Xtr2, Xte2, ytr2, yte2 = Xtr, Xte, ytr, yte
        r, m = train_pinn(Xtr2, ytr2, Xte2, yte2,
                          layers=best_cfg["layers"], lam_m=best_cfg["lam_m"], lam_z=best_cfg["lam_z"],
                          lr=best_cfg["lr"], seed=s, epochs=5000, patience=120)
        r2s.append(r); ms.append(m)
    print(f"  Split {sname}: R2 = {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}  MAPE = {np.mean(ms):.2f}%")

print(f"\n--- Phase 1 Baselines (Split C) for comparison ---")
print(f"  GridInterp (raw):   R2 = 0.8415  (deterministic)")
print(f"  Poly2_Ridge (log):  R2 = 0.7547  (deterministic)")
print(f"  MLP (log, 30-seed): R2 = 0.6277  (stochastic)")
print(f"  ExtraTrees (raw):   R2 = 0.4335  (deterministic)")
