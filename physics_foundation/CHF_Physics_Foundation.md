# The Physics of Critical Heat Flux — Foundation Document

**Project:** Generalized CHF prediction across surfaces, geometries and fluids
**Purpose:** Assemble every physical law, mechanism and closed-form equation that governs
CHF, so that the predictive model can be built *on physics* rather than on curve-fitting.
**Status:** Research compilation. Section 9 (proposed algorithm) is a discussion draft, not
implemented code.

---

## 0. Why this document exists

Everything in this repository so far predicts CHF by *fitting*: look-up-table
interpolation, tree ensembles, ANNs, GPR. The audit in `CHF Extrapolation Audit.html`
and the `unified_chf_pipeline` results both converge on the same verdict — these models
interpolate beautifully (R² ≈ 0.96–0.97 on random splits) and collapse the moment they
are asked for a surface, a fluid or a pressure they have not seen (strategy-3 R² = +0.17
to +0.22; LOSO folds near zero or negative).

That failure is not a hyperparameter problem. It is a **basis problem**. A model whose
features are `(P, G, x, D, L)` in raw engineering units has no way of knowing that R123
and water are the same physics at different scales. A model whose features are
*dimensionless groups derived from the momentum and energy balances* does.

So this document collects the physics: the conservation laws, the instability criteria,
the trigger mechanisms, the closed-form correlations, and — most importantly for us —
**the asymptotic and monotonicity constraints that any admissible CHF model must obey.**
Those constraints (Section 8) are the raw material for a physics-constrained algorithm.

**A note on rigour.** Every equation below is tagged:

| Tag | Meaning |
|---|---|
| **[V]** | Transcribed verbatim from a primary source read during this research; source cited inline. |
| **[S]** | Structure verified from a secondary source; one or more constants could not be read cleanly and must be checked against the original paper before publication. |
| **[X]** | Known to exist and be relevant, but **not** transcribed here because no clean primary source was obtained. Do not cite from memory. |

Nothing in this document was written from recollection. Where a source could not be
obtained, that is stated rather than papered over.

---

## 1. The phenomenon

### 1.1 The boiling curve and the definition of CHF

Plotting wall heat flux `q″` against wall superheat `ΔT_sat = T_w − T_sat` produces the
boiling curve, with four regimes separated by three transition points
(Liang & Mudawar 2018, §1.2) **[V]**:

1. **Single-phase** — low superheat, convection only.
2. **Nucleate boiling** — bubbles nucleate at the surface; highest heat transfer coefficients.
3. **Transition boiling** — parts of the surface nucleate, parts are vapour-blanketed.
4. **Film boiling** — continuous vapour blanket; lowest heat transfer coefficients.

Transitions: *onset of boiling* → *CHF* → *minimum heat flux (Leidenfrost)*.

> **CHF** is the point at which bubble nucleation is replaced by localised vapour blankets
> merging across the surface.

The engineering consequence is asymmetric and is the entire reason CHF matters:

- In a **temperature-controlled** system (fossil boiler tube), crossing CHF moves you
  along the curve into transition boiling — recoverable.
- In a **heat-flux-controlled** system (a nuclear fuel rod, a power electronics die),
  crossing CHF forces a jump from nucleate boiling straight to film boiling, because
  there is no stable intermediate state at fixed `q″`. The wall temperature excursion
  is immediate and can be hundreds of kelvin. This is *burnout* / *boiling crisis*.

NUREG/KM-0011 §2.1 makes the point explicitly: pre-nuclear boiling systems were mostly
temperature-controlled, so CHF was not urgent; water-cooled reactors are heat-flux
controlled and are *power-limited by CHF*. **[V]**

### 1.2 Two physically distinct crises share one name

This distinction matters enormously for a generalised model, and most ML papers ignore it
(Zhao et al. 2019 note that none of the prior ML publications even attempted to
distinguish them) **[V]**:

| | **DNB** (departure from nucleate boiling) | **Dryout** |
|---|---|---|
| Regime | Subcooled / low quality | High quality, annular flow |
| Mechanism | Vapour blanket forms *at the wall* while bulk is still liquid | The annular liquid *film* is fully evaporated/entrained |
| Quality at CHF | `x < 0` to small positive | `x` large positive |
| Heat flux level | High | Much lower |
| Excursion | "Fast dryout" — drastic, PWR-like | "Slow dryout" — gradual, BWR-like |
| Reactor | PWR | BWR |

Groeneveld (1986), via NUREG/KM-0011 §2.2, uses exactly this fast/slow dryout
language. **[V]**

**Implication for us:** a single smooth function `CHF = f(P, G, x, …)` is being asked to
represent two different physical mechanisms with different governing balances. Any
generalised model should either (a) carry a regime indicator, or (b) be built as a
mechanism-wise mixture. This is a stronger version of the `geometry_family` one-hot fix
already made in `unified_chf_pipeline/scripts/features.py`, which alone moved
`helical_coil_r123` from R² = −10.3 to R² = 0.93–0.96.

### 1.3 CHF detection criteria — the target variable is not uniquely defined

The outline's §"CHF detection criterion" needs this. NUREG/KM-0011 §2.2 documents four
historical methods **[V]**:

1. **Visual** — the test section "started to redden visually" (e.g. Hood & Isakoff 1962).
   Works for fast DNB; fails for slow BWR-type dryout, which produces only modest
   excursions and no discolouration.
2. **Physical burnout** — the section actually fails before power can be cut. Common at
   high flow / high subcooling where CHF is very large.
3. **Change in test-section resistance** — stainless steel (high temperature coefficient
   of resistance) used as one leg of a Wheatstone bridge; the excursion unbalances the
   bridge and trips the supply. Best for very fast excursions. (Dell et al. 1969;
   Matzner et al. 1965; Hewitt et al. 1965.)
4. **Test-section thermocouples** — most common. Attached at the downstream end of the
   heated length. Unreliable for very slow dryouts, where the better criterion is a
   change in the slope `ΔT_w/Δq` (Groeneveld 1986).

Plus a fifth, "byproduct" case: in film-boiling experiments, CHF quality was either taken
as the quality of first surface-temperature rise, or as the average of the last pre-CHF
and first post-CHF quality (Era et al. 1966; Bennett et al. 1967; Herkenrath et al. 1967). **[V]**

> **This is a label-noise source, not a footnote.** Rows in our merged dataset carry CHF
> values defined by different criteria. Merging them without a `detection_criterion`
> column means the model is fitting a target whose definition varies by source — and our
> hardest splits (surface-wise, leave-one-source-out) are *exactly* the splits where the
> criterion changes between train and test. Our `master_chf_dataset.csv` has no such
> column today.

### 1.4 Primary vs secondary parameters

NUREG/KM-0011 §2.3 **[V]**. CHF is a function of five primary parameters: pressure,
inlet temperature, mass flow, diameter, heated length.

Critically: Lee & Obertelli (1963) and Lee (1965) showed that **heated length and inlet
temperature can be replaced by the thermodynamic quality at the CHF location**, provided
`L/D > 50` so upstream history effects are washed out. This is the origin of the
*local conditions hypothesis* (§1.6) and the reason the LUT is indexed on `(P, G, x)`.

Secondary parameters, with NUREG's assessment:

| Parameter | Effect on CHF |
|---|---|
| Orientation | Not significant at high flow where stratification is suppressed (Wong et al. 1990); use a flow-regime map (Taitel & Dukler 1975) to bound it. |
| Test-section material | Generally little effect in flow boiling. But at low flow / DNB conditions, **high-conductivity materials can suppress hot spots under bubbles and raise CHF**. |
| Type of heating (Joule vs nuclear) | No significant effect (Leung et al. 1982). |
| Wall thickness | No discernible effect (Bergles 1963; Bennett et al. 1965), except possibly for very thin walls. Contrast with pool boiling, §4.5 — there it matters a lot. |
| **Surface roughness** | Generally small in flow boiling, *"because the vapor generation rate at the surface usually determines the CHF occurrence."* **But: when roughness exceeds the liquid film thickness in annular-film dryout, premature film breakdown reduces CHF.** |
| Inlet/outlet throttling | Outlet restriction → flow/pressure instability → **significant CHF reduction** (Lowdermilk 1958; Mayinger et al. 1966). A "soft" (unthrottled) inlet permits oscillations; a "hard" inlet suppresses them (Kirillov 1997). |
| Dissolved gas | Up to **30% CHF reduction** at 4,000 Ncm³/kg (Kirillov 1997). |

> **Note the tension** with the outline's premise. NUREG says roughness effects in *flow*
> boiling are usually small; the pool-boiling literature (§4.2) says roughness and
> wettability matter enormously. Both are right — they are different regimes. The
> defensible framing for the paper is: *surface characteristics dominate in pool
> boiling / low-flow DNB and in annular dryout when roughness is film-scale, and are
> second-order in high-flow subcooled DNB.* A generalised model must reproduce that
> regime-dependence, not assert a single global surface sensitivity.

### 1.5 Energy balance and thermodynamic quality

The one exact relation in the whole subject. For a uniformly heated tube, an energy
balance between inlet and the CHF location (Hall & Mudawar 2000b, Eq. 8) **[V]**:

```
h_o = h_i + 4 · (L/D) · (q″/G)
```

In non-dimensional form (their Eq. 9–10) **[V]**:

```
x_o = x_i* + 4 · Bo · (L/D)
```

where the **boiling number** is

```
Bo = q″ / (G · h_fg)
```

and `x_i*` is the *pseudo-inlet quality* `(h_i − h_f,o)/h_fg,o`, i.e. inlet enthalpy
referenced to *outlet*-pressure saturation properties — which is what lets the
correlation avoid needing the inlet pressure at all.

**Thermodynamic equilibrium quality:**

```
x = (h − h_f) / h_fg
```

with saturated properties evaluated at the pressure of the CHF point (usually outlet).
`x < 0` denotes subcooled conditions and is a *pseudo*-quality — there is no vapour, it is
an enthalpy deficit expressed in quality units.

**Why this matters to us:** this equation is an exact constraint linking
`x_o, x_i, L/D, Bo`. It means an "inlet conditions" correlation and an "outlet (local)
conditions" correlation are transformations of one another, *not* independent modelling
choices. Any model that takes `x`, `L/D` and `G` as free independent features is
implicitly allowed to violate energy conservation. **This is a hard constraint we can
impose for free.**

### 1.6 The local conditions hypothesis

> For a water-cooled tube of fixed diameter, CHF is a unique function of the *local*
> pressure, mass flux and thermodynamic quality at the CHF location.

(NUREG/KM-0011 §3.3 **[V]**.) This is the assumption underpinning the entire look-up
table approach. It is an *assumption*, and it is what fails when `L/D < 50`, under
non-uniform axial heating, and under flow instability. Hall & Mudawar note the
consequence precisely: the outlet-conditions correlation *cannot* show an `L/D` effect,
because it hypothesises CHF depends only on local conditions — whereas the inlet-conditions
form does exhibit `L/D`. **[V]**

---

## 2. Dimensional analysis — the backbone of generalisation

This section is the most important one for our modelling goal.

### 2.1 Kutateladze's derivation (1948) — worked through

The archetype for how *all* of this works. From Liang & Mudawar 2018 §1.4 **[V]**:

**Step 1.** Vigorous boiling releases vapour perpendicular to the surface at velocity

```
u_g = q″ / (ρ_g · h_fg)                                              (1)
```

**Step 2.** At CHF, the vapour's kinetic energy just balances the gravitational force on
the suspended liquid:

```
ρ_g · u_g²  ~  g · (ρ_f − ρ_g) · δ*                                  (2)
```

**Step 3.** `δ*` is the linear scale of capillary disturbances — the **capillary length**:

```
δ* = [ σ / (g(ρ_f − ρ_g)) ]^(1/2)                                    (3)
```

**Step 4.** Substituting (1) and (3) into (2) yields the Kutateladze–Zuber form:

```
        q″_CHF
──────────────────────────────────  =  K                             (4)
ρ_g h_fg [ σ g (ρ_f − ρ_g) / ρ_g² ]^(1/4)
```

Kutateladze recommended **K = 0.16** for a large horizontal flat surface.

**The key structural insight**, stated by Liang & Mudawar themselves:

> "because this equation was derived from dimensional analysis premises, it is expected
> that different theoretical CHF models for saturated pool boiling may be arranged into
> the form of Eq. (4), **independent of the CHF mechanism proposed**." **[V]**

So the entire pool-boiling literature reduces to *one functional form and a dimensionless
number K*, where K carries the physics of the mechanism, the surface, the orientation and
the fluid. **That is the template for our model: predict a dimensionless CHF, not a
dimensional one.**

### 2.2 The dimensionless group inventory

| Group | Definition | Physical meaning | Source |
|---|---|---|---|
| **Dimensionless CHF** `K` | `q″/(ρ_g h_fg [σg(ρ_f−ρ_g)/ρ_g²]^{1/4})` | CHF vs. hydrodynamic limit | Kutateladze **[V]** |
| **Boiling number** `Bo` | `q″/(G h_fg)` | Wall evaporation vs. total flow enthalpy capacity | Hall & Mudawar **[V]** |
| **Weber number** `We_D` | `G²D/(ρ_f σ)` | Flow inertia vs. surface tension | Hall & Mudawar **[V]** |
| **Katto number** | `σρ_f/(G²L)` | Inverse Weber on heated length | Katto & Ohno **[V]** |
| **Density ratio** | `ρ_f/ρ_g` or `ρ_g/ρ_f` | Proxy for reduced pressure; sets regime boundaries | multiple **[V]** |
| **Reduced pressure** | `P/P_crit` | Property-collapse coordinate | multiple **[V]** |
| **Capillary length** | `[σ/(g(ρ_f−ρ_g))]^{1/2}` | Bubble/instability length scale | Kutateladze **[V]** |
| **Bond number** | `Bo_d = (ρ_l−ρ_g) g d²/σ` | Buoyancy vs. surface tension | Merilo, via Hardik & Prabhu **[V]** |
| **Jakob number** | `Ja = ρ_f c_p ΔT_sub/(ρ_g h_fg)` | Sensible vs. latent capacity (subcooling) | present in `master_chf_dataset.csv` col `ja` |
| **Kandlikar** `K₁` | `(q″/(G h_fg))² · (ρ_f/ρ_g)` | Evaporation momentum vs. inertia | Kandlikar **[S]** |
| **Kandlikar** `K₂` | `(q″/h_fg)² · D/(ρ_g σ)` | Evaporation momentum vs. surface tension | Kandlikar **[S]** |
| **Thermal activity** `S` | `H·√((ρ c_p k)_w)` | Wall thermal inertia per unit area | Bar-Cohen & McNeil **[V]** |
| `L/D` | — | Heated-length / history effect | multiple **[V]** |
| `D/D_ref` | `D/8mm` | Diameter normalisation | Groeneveld LUT **[V]** |

Kandlikar's interrelations **[S]**: `K₂/K₁ = We`, `K₃/K₁ = Re`, `K₂/K₃ = Ca`, where `K₃`
is evaporation momentum over viscous force. The general dimensionless statement of a tube
CHF correlation, from NUREG/KM-0011 §3.1 **[V]**:

```
  CHF                ⎛ ρ_f        D^0.5              D    ⎞
─────────  =  f      ⎜ ───  ,  ───────────  ,  x  ,  ──── ⎟
 h_fg G              ⎝ ρ_g     (σρ_f)^0.5           D_ref ⎠
```

i.e. **Bo = f(density ratio, Weber-like group, quality, geometry ratio)**.

> This is the single most actionable equation in this document. It says the *correct*
> feature space for a generalised CHF model is four dimensionless numbers, not five
> dimensional ones. It is fluid-agnostic by construction: water and R123 map to the same
> point if their dimensionless groups match. Our current pipeline already takes a
> half-step toward this by training on `log(CHF / physics_baseline)` — which is
> `log(Bo/Bo_ref)` in disguise, and which is exactly what moved strategy-3 from R² = −16.4
> to +0.22. Section 9 argues for taking the full step.

---

## 3. Pool boiling CHF — the five trigger mechanisms

Liang & Mudawar 2018 §1.3 identify five competing mechanisms **[V]**. There is *still* no
consensus on which is correct — that lack of consensus is itself a research finding worth
stating in the paper.

### 3.1 Bubble interference (Rohsenow & Griffith 1955)

CHF when neighbouring isolated bubbles coalesce radially **[V]**:

```
q″_CHF = 0.012 · ρ_g · h_fg · [ (ρ_f − ρ_g)/ρ_g ]^0.6              (5)
```

Chang & Snyder's variant, which introduces **contact angle α** — historically the first
appearance of a surface property in a CHF model **[V]**:

```
q″_CHF = ½ (π/6)^(5/6) (0.0119 α)^(1/2) · ρ_g h_fg [2σg(ρ_f−ρ_g)/ρ_g²]^(1/4)   (6)
```

Both are undermined by later high-speed photography showing coalescence occurs *well
before* CHF.

### 3.2 Zuber hydrodynamic instability (1959) — full derivation

The most influential CHF model ever published. Full chain from Liang & Mudawar §3.1 **[V]**:

**Geometry.** Vapour jets leave the surface at `u_g`; liquid returns between them at `u_f`.
The surface is modelled as repeating square cells, one jet each. Jet diameter is set by
**Rayleigh–Taylor instability**: `D_j = λ_T/2`, where `λ_T` lies between:

```
critical wavelength:        λ_c = 2π √( σ / (g(ρ_f − ρ_g)) )              (7a)
most dangerous wavelength:  λ_d = √3 · λ_c = 2π√3 √( σ/(g(ρ_f−ρ_g)) )     (7b)
```

**Area ratio.** For a square cell of side `λ_T` containing one jet of diameter `λ_T/2`:

```
A_g/A_w = π(λ_T/2)²/4 / λ_T² = π/16                                        (8)
```

**Continuity.**

```
u_f = (ρ_g/ρ_f) · [ (A_g/A_w)/(1 − A_g/A_w) ] · u_g
    = (ρ_g/ρ_f) · [ π/(16 − π) ] · u_g                                     (9)
```

**Energy.** All wall heat goes to vaporisation:

```
q″_CHF = (A_g/A_w) ρ_g h_fg u_g       ⟹      u_g = q″_CHF / ((π/16) ρ_g h_fg)   (10, 11)
```

**Trigger — Kelvin–Helmholtz instability.** The velocity difference across the jet
interface destabilises it; growth merges adjacent jets into a vapour mushroom that blocks
liquid resupply:

```
u_g − (−u_f) = [ (ρ_f + ρ_g)/(ρ_f ρ_g) ]^(1/2) · ( 2πσ/λ_H )^(1/2)         (12)
```

with Zuber's closure `λ_H = π D_j` (13).

**Result.** Combining (7a)–(13) **[V]**:

```
        q″_CHF                       ⎡ π    3    1   ⎤   ⎡      (1 + ρ_g/ρ_f)^(1/2)  ⎤
──────────────────────────────  =    ⎢ ── ───── ──── ⎥ · ⎢ ───────────────────────── ⎥
ρ_g h_fg [σg(ρ_f−ρ_g)/ρ_g²]^(1/4)    ⎣ 24 √(2π)  3^¼ ⎦   ⎣    1 + ρ_g/ρ_f · π/(16−π) ⎦
```

The bracket range (from `λ_c < λ_T < λ_d`) gives **K between 0.119 and 0.157**; Zuber
recommended the intermediate **K = 0.131** **[V]**:

```
q″_CHF = 0.131 · ρ_g h_fg [ σ g (ρ_f − ρ_g)/ρ_g² ]^(1/4)                  (15)
```

This is the `zuber_pool_boiling_chf()` already implemented in `scripts/chf_physics.py`
and used as the pool-boiling half of `compute_physics_baseline_kw_m2` in the unified
pipeline.

**Note what Zuber's model does *not* contain: any surface property.** No contact angle,
no roughness, no material, no wettability. That absence is the launching point for the
entire engineered-surface literature (§4) and for the outline's Gap 1.

### 3.3 Lienhard & Dhir finite-size correction

Setting `λ_H = λ_d` instead of Zuber's `λ_H = πD_j` **[V]**:

```
q″_CHF / q″_CHF,Zuber = 1.14        ⟹     K = 0.149                       (16)
```

Validity: the infinite-flat-plate assumption holds for surface widths **> 3λ_d**
(Lienhard et al.). Zhang et al. (2D simulation): the surface can be treated as infinite
when heater width / capillary length **> 12**. Gogonin: 2 capillary lengths suffice. **[V]**

> **Directly relevant to our pin-fin and flat-heater pool-boiling sources.** A finite
> heater is not a small perturbation on Zuber — it is a different asymptote. If
> `heater_width_mm / capillary_length < 3λ_d`, the Zuber baseline is the wrong
> normaliser and our `log(CHF/baseline)` target inherits that error.

### 3.4 Haramura & Katto macrolayer dryout (1983)

A large hovering bubble sits above a thin liquid **macrolayer** pierced by vapour stems.
CHF is triggered when the macrolayer dries out just before bubble departure **[V]**:

```
q″_CHF = ρ_f h_fg δ (1 − A_g/A_w) f                                       (25)
```

`δ` = macrolayer thickness, `f` = bubble departure frequency. With `δ = λ_H/4` **[V]**:

```
q″_CHF = 0.721 (A_g/A_w)^(5/8) [ (1 − A_g/A_w)^(5/16) ((ρ_f/ρ_g)+1) / ((11/16)(ρ_f/ρ_g)+1)^(3/5) ]^(5/16)
         × ρ_g h_fg [ σ(ρ_f−ρ_g)g/ρ_g² ]^(1/4)                            (26a)
```

with

```
A_g/A_w = 0.0584 (ρ_g/ρ_f)^(1/5)                                          (27)
```

**Macrolayer thickness — four competing closures [V]:**

| Author | `δ` |
|---|---|
| Haramura & Katto | `λ_H/4 = 0.00536 σ ρ_g (1+ρ_g/ρ_f)(ρ_g/ρ_f)^{2/5} (h_fg/q″)²` |
| Rajvanshi et al. | `λ_H/2 = 0.0107 σ ρ_g (1+ρ_g/ρ_f)(ρ_g/ρ_f)^{2/5} (h_fg/q″)²` |
| Kumada & Sakashita | `0.786 [ν_g⁸σ¹¹/(ρ_f⁶g⁵(ρ_f−ρ_g)⁵)]^{1/24} (ρ_g h_fg/q″)^{5/6}` |
| Chappidi et al. | `[ (ρ_g/ρ_f)(A_g/A_w)(1−A_g/A_w)^{−1} √((ρ_f+ρ_g)/(ρ_fρ_g)) √σ τ ]^{2/3}`, τ = bubble hovering time |

Also `δ = 1.1 R` (R = mean vapour stem radius), Chappidi et al. **[V]**

**Status honestly reported:** Xiao & Yu report the macrolayer never dries out; Jung et al.
found *no* evidence of regularly spaced vapour jets or a trapped liquid layer using
high-speed IR. Those observations undermine both the macrolayer model *and* Zuber. **[V]**

### 3.5 Hot / dry spot models

**Yagov (1988, 2014)** — CHF as irreversible growth of dry spots. Uniquely among these
models, it carries **liquid viscosity** dependence.

Low reduced pressure, `P/P_c < 0.001` **[V]**:

```
              0.5 · h_fg^(8/55) σ^(9/11) ρ_g^(13/110) k_f^(7/110) g^(21/55) f(Pr_f)
q″_CHF,l  =  ─────────────────────────────────────────────────────────────────────   (33)
                        ν_f^(1/2) c_p,f^(3/10) R_i^(79/110) T_sat^(21/22)
```

```
                        Pr_f^(9/8)
f(Pr_f) = [ ──────────────────────────────── ]^(4/11)                     (34)
            1 + 2Pr_f^(1/4) + 0.6 Pr_f^(19/24)
```

High reduced pressure, `P/P_c > 0.03` **[V]**:

```
q″_CHF,h = 0.06 h_fg ρ_g^(3/5) σ^(2/5) [ g(ρ_f − ρ_g)/μ_f ]^(1/5)         (35)
```

Blend for `0.001 ≤ P/P_c ≤ 0.03` **[V]**:

```
q″_CHF = ( q″_CHF,h³ + q″_CHF,l³ )^(1/3)                                  (36)
```

> **This blending function is a template worth stealing.** It is a smooth,
> differentiable, physically-motivated way to interpolate between two asymptotic regimes
> — far better behaved than the hard `if` switches in Biasi and Katto–Ohno, and directly
> applicable to our mechanism-mixture idea in §9.

**Theofanous & Dinh** — CHF governed by microlayer instability **[V]**:

```
q″_CHF = λ^(−1/2) · ρ_g h_fg [ σ g (ρ_f − ρ_g)/ρ_g² ]^(1/4)               (37)
```

`λ` decreases with increasing wettability. Kim et al. derived it from Rayleigh's static
meniscus volume formula, for hydrophilic surfaces (α < 90°) **[V]**:

```
λ = [ (1 − sin α)/2 − (π/2 − α)/(2 cos α) ]^(−1/2)                        (38)
```

### 3.6 Galloway & Mudawar interfacial lift-off

CHF occurs when vapour momentum normal to the wall exceeds the restraining pressure force
from interfacial curvature; wetting fronts in the wave troughs are the last cooling
mechanism **[V]**:

```
ρ_g [ q″_l / (ρ_g h_fg (1 + c_p,f ΔT_sub/h_fg)) ]²  =  P_f − P_g          (39)

P_f − P_g = 2√2 π σ δ / λ_c²                                              (40)
```

With wetting-front span = `λ_c/4` (so `q″_CHF = q″_l/4`) **[V]**:

```
q″_CHF = ¼ ρ_g h_fg (1 + c_p,f ΔT_sub/h_fg) · [ 2√2 π σ δ / (ρ_g λ_c²) ]^(1/2)   (41)
```

This model spans pool *and* flow boiling — Mudawar's group found near-vertical pool
boiling interfacial behaviour closely resembles flow boiling CHF. Note the explicit
**subcooling factor `(1 + c_p,f ΔT_sub/h_fg)`** — this is the canonical way subcooling
enters CHF models, and it is a Jakob-number term.

---

## 4. Surface characteristics — the physics behind the outline's core claim

This is the section the paper outline is built on. The honest summary: **the literature
is genuinely contested, and that contest is itself the research opportunity.**

### 4.1 Contact angle and wettability

**The disagreement, stated plainly** (Liang & Mudawar §3.2.2) **[V]**:

- *Against:* Kutateladze–Zuber is independent of surface conditions and Bewilogua et al.
  validated that independence. Stock: contact angle has only weak influence. O'Hanley et
  al.: wettability alone has little influence **on smooth surfaces**.
- *For:* Gaertner; Costello & Frea; Maracy & Winterton; Hahne & Diesselhorst all found CHF
  decreases appreciably with decreasing wettability. Liaw & Dhir: at a contact angle of
  **107°, CHF is only half** the hydrodynamic-theory prediction.
- *Caution:* Liao et al. warn a robust CHF–contact-angle relation is hard to establish
  because fluid and surface thermal properties vary synchronously with temperature, which
  itself moves the contact angle.

Liang & Mudawar's conclusion **[V]**:

> "contact angle is important to modeling pool boiling CHF even for smooth surfaces, let
> alone the complicated contact angle variations resulting from external influences. This
> fact points to a need for more systematic studies of pool boiling CHF from smooth surfaces."

**Models incorporating contact angle α:**

| Author | `K` | Notes |
|---|---|---|
| Kirichenko & Chernyakov | `K = 0.171(1 + 0.324×10⁻³α²)^{1/4} / (0.018α)^{1/2}` | **[V]** |
| Ramilison et al. | `K = 0.0044 (π − α)³ R_a^{0.125}` | **[V]** contact angle *and* roughness |
| Kim et al. | see §4.2 Eq. (21) | **[V]** angle + roughness + peak spacing |
| Kandlikar | see §4.3 | **[V]** angle + orientation |
| Theofanous & Dinh / Kim | Eq. (38) above | **[V]** |

Note α in these expressions is in **degrees** for Kirichenko & Chernyakov and Ramilison
(the `(π − α)³` form implies radians for Ramilison — **this unit ambiguity must be
resolved against the primary sources before use**).

### 4.2 Surface roughness

Same pattern of disagreement **[V]**:

- *No effect:* O'Hanley et al., Berenson, Lyon, Ramilison & Lienhard, Nishio & Chandratilleke.
- *Effect:* Bailey et al. (CHF above Eq. 15 prediction, attributed to roughness);
  Guan et al. (up to **15% enhancement** at `R_a` = 5 µm vs. smooth, for pentane/hexane/FC-72).
- *Strong effect:* Kim et al., copper, water, moderate wettability α = 60–70°: CHF rose
  from **77.5 W/cm² at `R_a` = 0.041 µm to 162.5 W/cm² at `R_a` = 2.36 µm** — a 2.1×
  increase, ascribed to **capillary wicking** on the rougher surface.
- *Saturating effect:* Kim et al. on superhydrophilic aluminium (α = 7–16.3°): CHF rose
  from 165 to 215 W/cm² between `R_a` = 0.11 and 0.35 µm, then became **negligibly
  dependent** on roughness over 0.35 < `R_a` ≤ 2.93 µm.

**Kim et al. combined roughness + wettability correlation** — the single most useful
closed form for the outline's purposes **[V]**:

```
        ⎧ (1 + cos α)  ⎡ 2     π                       cos α    R_a ⎤ ⎫^(1/2)
K = 0.811⎨ ─────────── ⎢ ─  +  ─ (1 + cos α) + 351.2 ─────────  ─── ⎥ ⎬
        ⎩      16      ⎣ π     4                     1 + cos α   S_m ⎦ ⎭     (21)
```

where `S_m` is the **mean spacing between roughness peaks**. Note this needs *two*
roughness descriptors (`R_a` and `S_m`), not one — amplitude alone is insufficient.

> **Data gap in our repository.** `master_chf_dataset.csv` has `roughness_factor` at
> **0.6% coverage** and no `S_m`, no contact angle column at all. The outline's Gap 1 and
> the SHAP ranking it proposes (contact angle, roughness among the top features) are not
> currently supportable from our merged data. This is the single largest gap between the
> outline's ambition and the dataset's reality, and it should be resolved before the
> results section is written.

### 4.3 Orientation

Consistent finding across Lyon, Priarone, El-Genk & Bostanci, Beduz et al., Brusstar et
al. **[V]**: CHF decreases *slightly* from θ = 0° to 90°, then **rapidly** toward 180° as
buoyancy traps vapour against the surface.

Howard & Mudawar's photographic study identifies three mechanism regions **[V]**:

- **θ = 0–60°** (upward-facing): buoyancy removes vapour vertically.
- **θ = 60–165°** (near-vertical): a wavy liquid–vapour interface propagates along the surface.
- **θ > 165°** (downward-facing): vapour stratification, greatly reduced CHF.

Their conclusion is important for us: *"it is impossible to account for orientation
effects using a single CHF model,"* and three separate models should be developed. **[V]**
Yang et al. also found a **transition angle** beyond which CHF drops sharply, which
*increases* with heater size while CHF *decreases*.

**Orientation correlations for `K`** (Liang & Mudawar Table 2, θ in degrees) **[V]**:

| Author | `K` | Range |
|---|---|---|
| **Kandlikar** | `K = [(1+cos α)/16] · [ 2/π + (π/4)(1 + cos α) cos θ ]^{1/2}` | θ = 0–90° |
| Liao et al. | `K = 0.131[−0.73 + 1.73/(1+10^{−0.021(185.4−θ)})]·[1 + ((55−α)/100)(0.56 − 0.0013θ)]` | θ = 0–180° |
| Priarone | `K_FC-72 = 0.165 f(θ)`, `K_HFE-7100 = 0.21 f(θ)`, `f(θ) = 1 − 0.001117θ + 7.79401×10⁻⁶θ² − 1.37678×10⁻⁷θ³` | θ = 0–175° |
| El-Genk & Bostanci | `K = [(0.229 − 4.27×10⁻⁴θ)^{−6} + (0.577 − 2.98×10⁻³θ)^{−6}]^{−1/6}` | θ = 0–180° |
| Vishnev | `K = 0.0125(190 − θ)^{1/2}` | θ = 0–180° |
| Arik & Bar-Cohen | `K = 0.131(1 − 0.001117θ + 7.79401×10⁻⁶θ² − 1.37678×10⁻⁷θ³)` | θ = 0–180° |
| Chang & You | `q″/q″_max = 1 − 0.0012 θ tan(0.414θ) − 0.122 sin(0.318θ)` | θ = 0–180° |
| El-Genk & Guo | `K_water = 0.034 + 0.0037(180−θ)^{0.656}`; `K_N2 = 0.033 + 0.0096(180−θ)^{0.479}`; `K_He = 0.002+0.0051(180−θ)^{0.633}` | θ = 90–180° |
| Brusstar & Merte | `K = (π/24)\|sin θ\|^{1/2}[1 + 0.102(ρ_g/ρ_f)^{1/4}(ρ_f c_p,f ΔT_sub/(ρ_g h_fg))]` | θ = 90–180° |

**The Kandlikar model is the one to build on** — it is the only entry that is
simultaneously theoretical (a force balance on a bubble: evaporation momentum vs. surface
tension vs. hydrostatic) and carries *both* the receding contact angle α and the
orientation θ. Our dataset has `orientation` and `angle_deg` columns (0.2% coverage).

### 4.4 Wickability — the modern paradigm

Rahman, Ölçeroğlu & McCarthy engineered ~40 surfaces of varying wickability and showed
**CHF increases linearly with surface wickability** (wicked volume flux), reaching
**260 W/cm²** on the highest-wickability hierarchical structure. The controlling
mechanism is capillary-driven liquid supply to the surface, not contact angle per se.
Surfaces studied included biotemplated nanostructures, square micropillar arrays of
varying diameter/height/pitch, and hierarchical structures. **[S]** — obtained via search
summary; the primary papers (Rahman et al., *Langmuir* 2014; *Sci. Rep.* 2017) were
paywalled/CAPTCHA-blocked and **the linear coefficient was not obtained. Must be read
before citing a number.**

This matters because it *reconciles* §4.1 and §4.2: contact angle and roughness are both
imperfect proxies for the thing that actually matters — the surface's ability to pull
liquid back to the dry spot. Kim et al.'s roughness result was already ascribed to
wicking. Our dataset's `porosity`, `mbl_lateral_um`, `mbl_total_um`, `fin_spacing_um`
columns are wickability-relevant geometry, at 0.6% coverage.

### 4.5 Wall thermal properties — the "thermal activity" parameter

Unlike flow boiling (§1.4), in pool boiling the wall itself matters. Bar-Cohen & McNeil
define **[V]**:

```
S = H · √( (ρ c_p k)_w )
```

`H` = wall thickness. Then:

```
q″_CHF/q″_CHF,asy = S/(S + 0.8)     (Bar-Cohen & McNeil)      (17)
q″_CHF/q″_CHF,asy = S/(S + 0.1)     (Watwe & Bar-Cohen)       (18)
q″_CHF/q″_CHF,asy = 1 − exp[ −(S/2.44)^{0.8498} − (S/2.44)^{0.0581} ]   (Golobič & Bergles, sat. FC-72)  (20)
```

Eq. (17) gives 90% of asymptotic CHF at S = 8, 99% at S = 85; Eq. (18) at S = 1 and 10.
Thresholds: stainless steel needs ≥ 0.8 mm to be free of heat-capacity effects
(Tachibana et al.); copper in liquid helium saturates at 0.35 mm (Grigoriev et al.). **[V]**

**Watwe & Bar-Cohen composite**, combining hydrodynamics, conduction, heater size and
subcooling — a good example of a fully-assembled `K` **[V]**:

```
      π   ⎛   S    ⎞ ⎧          ⎡        ⎛ g(ρ_f − ρ_g) ⎞^(1/2) ⎤ ⎫ ⎧      ⎛ρ_f⎞^(3/4) c_p,f      ⎫
K =  ── · ⎜ ────── ⎟ ⎨ 1 + ⎢0.3014 − 0.01507 L ⎜ ──────────── ⎟     ⎥ ⎬ ⎨1 + 0.03⎜───⎟      ──── ΔT_sub ⎬
     24   ⎝ S+0.1  ⎠ ⎩          ⎣        ⎝      σ       ⎠      ⎦ ⎭ ⎩      ⎝ρ_g⎠      h_fg      ⎭   (19)
```

Our dataset has `surface_material` at 0.6% coverage — enough to *identify* material but
not to compute `S` (no wall thickness column).

### 4.6 Pressure and viscosity effects on `K`

**[V]** Deev et al.: helium data fit Eq. (4) with **K = 0.2** for `P/P_c ≤ 0.75`; above
0.75 the Kutateladze–Zuber form breaks down entirely. Labuntsov et al.: water and ethanol
show `(P/P_c)^{0.15}` dependence. Dhir and Wang et al. independently place the CHF
**maximum at `P/P_c ≈ 0.35`**.

```
Bewilogua (cryogens):  q″/q″_max = 0.421 + 3.58(P/P_c) − 6.19(P/P_c)² + 2.21(P/P_c)³   (22)
Wang et al.:           K = 0.18 − 0.14 (P/P_c)^{5.68}                                   (23)
Soziev & Khrizolitova (very low P):  K = 0.16{1 + [σg(ρ_f−ρ_g)]^{1/2}/P}^{1/2}         (24)
Borishanskii (viscosity): K = {0.13 + 4[ρ_f σ^{3/2}/(μ_f² g(ρ_f−ρ_g))^{1/2}]}^{−2/5}   (Table 1)
```

> **The `P/P_c ≈ 0.35` maximum is a hard, checkable constraint.** Any model we build
> must reproduce a non-monotonic pressure trend with a peak near `P/P_c = 0.35`
> (≈ 7.7 MPa for water). This is precisely the trend the existing repo notes the
> CoolProp-based Zuber implementation reproduces at 6–7 MPa without tuning. It is a free
> validation test, and a model that gets it wrong is wrong regardless of its R².

Sakashita's observation is subtle and worth noting: Kutateladze–Zuber captures the
pressure trend for ethanol and R-141b by tuning `K`, but **underestimates it for water** —
because water's wettability *improves with increasing pressure* while ethanol's and
R-141b's does not. Pressure and surface effects are coupled, not separable. **[V]**

---

## 5. Flow boiling CHF — correlations

### 5.1 Why there are 500+ of them

NUREG/KM-0011 §3.1 **[V]**: Clerici (1966) listed 50+ CHF correlations for water-cooled
tubes; the count now exceeds **500**. The reason is stated precisely:

> "The choice of correlation parameters (P, G, X_CHF, D) is essentially correct, but the
> **functional relationship between these parameters varies with flow conditions** — hence,
> the large proliferation of correlations."

And bluntly:

> "none of the early flow boiling CHF correlations for tubes carry much credibility today,
> partially because the pool boiling CHF equations had a **physical basis**, whereas the
> tube CHF correlations were **virtually all empirical**."

NUREG singles out **Katto (1992)** and **Lee & Mudawar (1988)** for special mention
because they (1) have a phenomenological basis, (2) rest on a large database, and (3) have
wider validity. That is a direct endorsement of the physics-first strategy.

**This paragraph is the literature justification for the whole project.** It is the
strongest available citation for the outline's "Limitations of existing CHF correlations".

### 5.2 The look-up table method

The generalisation response to the proliferation problem. Built on the local conditions
hypothesis (§1.6). The 2006 LUT covers **P = 0.1–21 MPa, G = 0–8,000 kg m⁻² s⁻¹,
x = −0.50 to 1.00**, normalised to a **vertical 8 mm water-cooled tube**, with linear
interpolation between entries. **[V]**

NUREG's stated advantages **[V]**: greater accuracy, wider applicability, correct
asymptotic trends, less computing time, easily updated.

Lineage: Doroshchuk et al. (1975, 5,000 points) → Groeneveld AECL-UO 1986 (15,442 points)
→ Kirillov et al. → 1995 LUT (combined ~24,000 de-duplicated points, with Huang & Cheng
1994 smoothing to remove sharp variations at data/extrapolation boundaries) → 2006 LUT
(+27 further datasets). **[V]**

> **Important caveat for our project, already correctly identified in the pipeline
> README:** the LUT is *smoothed and partly correlation-filled*, not raw measurement. The
> `unified_chf_pipeline` deliberately excludes it from the merged training data for this
> reason. It remains a legitimate **baseline** and a legitimate **normaliser**, but not a
> source of independent data points.

### 5.3 LUT correction factors K1–K8

The mechanism by which the 8 mm-tube table is extended to real geometries. From the VVER
methodology paper (arXiv:2203.15048), citing IAEA-TECDOC-1203 and Groeneveld **[V]**:

```
CHF_bundle = K1 · K2 · K3 · K4 · K5 · K6 · K7 · K8 · CHF_table                (8)
```

The factors assume **independence**, which the source explicitly calls "a first-order
approximation."

| | Description | Formula |
|---|---|---|
| **K1** | Tube diameter | `K1 = (8/D_h)^n` for 2 ≤ D_h ≤ 25 mm; `= 0.57` for D_h > 25 mm  (9) |
| **K2** | Bundle geometry | `K2 = min[1, (0.5 + 2(s−d)/d)·exp(−0.5\|x\|^{1/3})]`  (11) |
| **K3** | Mid-plane spacer (CANDU) | `K3 = 1 + A exp(−0.1 L_s/D_h)`, `A = 1.5 ξ^{0.5}(0.001G)^{0.2}`  (12,13) |
| **K4** | Heated length | `K4 = 1` for L/D_h < 5; `= exp[e^{2α} L/D_h]` for L/D_h ≥ 5, with `α = xρ_f/(xρ_f + (1−x)ρ_g)`  (14,15) |
| **K5** | Axial flux distribution | `K5 = 1` for x ≤ 0; `= q_local/q_BLA` for x > 0, `q_BLA = (1/L_B)∫q(z)dz` over the boiling length  (16,17) |
| **K6** | Radial flux distribution | `K6 = 1` for x ≤ 0; `= q(z)_max/q(z)_av` for x > 0  (18) |
| **K7** | Flow orientation | `K7 = 1 − exp[−(A/3)^{0.5}]`, `A = ((1−x)/(1−α))² fG²/(g D_h ρ_f(ρ_f−ρ_g)α^{0.5})`  (19,20) |
| **K8** | Vertical low flow | for −400 < G < 0: `K8·CHF_table = 2CHF_p − CHF(\|G\|)`; `CHF_p = B(1−α)CHF(G=0,x=0)`  (21,22) |

Note `K3 = 1` and `K5 = K6 = 1` for subcooled conditions; `K7 = 1` for horizontal flow;
`K8 = 1` for upward flow. IAEA-TECDOC-1203 recommends using **either K3 or K4**, not both.
The source also warns **K2's formula is based on saturated qualities and has not been
tested for negative subcooled qualities.** **[V]**

**The diameter exponent `n` is contested** — the single most-studied correction:

- Groeneveld originally: `n = 0.5` (used to normalise the LUT itself).
- Others: `n = 1/3`.
- Biasi's correlation: `0.4` for D < 10 mm, `0.6` otherwise.
- Wong correlation (lowest RMS per Tanase et al.): `n = 0.58[1 − 0.25exp(−2x)](1 − 15D_h^{−6}G)` **[V]**
- Tanase et al. (2009) tabulate `n` by (P, G, x) region — including **negative values**
  (`n = −0.2` to `−0.3`) at low mass flux, meaning the diameter trend *reverses*. **[V]**

**Tanase et al. Table 3 — `n` by regime [V]:**

| P (kPa) | G (kg m⁻²s⁻¹) | x: −0.5..−0.25 | −0.25..0 | 0..0.5 | 0.5..1 |
|---|---|---|---|---|---|
| 100–14000 | 0–250 | −0.2 | −0.2 | −0.2 | −0.3 |
| | 250–3000 | 0.4 | 0.4 | 0.5 | 0.6 |
| | 3000–8000 | 0.3 | 0.3 | 0.4 | 0.4 |
| 14000–21000 | 0–250 | −0.2 | −0.2 | −0.2 | −0.3 |
| | 250–3000 | 0.4 | 0.2 | 0.4 | 0.4 |
| | 3000–8000 | 0.3 | 0.2 | 0.2 | 0.2 |

Boltenko's physical explanation **[V]**: at low quality (`x < 0.2`) CHF **increases** with
tube ID; at high quality the trend **reverses**. Tanase's sensitivity study found constant
`n = 0.4–0.5` predicts the whole LUT range satisfactorily (RMS 7.1–7.3% for HBM).

> **Directly usable.** This table is a ready-made, physics-derived, regime-dependent
> feature transform. Rather than letting a model learn the diameter effect from scratch
> — which it demonstrably cannot do across held-out geometries — we can normalise every
> row to 8 mm using `K1` with a regime-appropriate `n`, and let the model learn only the
> residual. This is the same "divide by a physics baseline first" trick that already
> rescued strategy 3, applied to geometry instead of fluid.

### 5.4 Biasi et al. (1967)

As implemented in `scripts/chf_physics.py` (citing Todreas & Kazimi, *Nuclear Systems I*).
Units: D in cm, G in g cm⁻² s⁻¹, P in atm, output kW/m². **[V]** (as implemented)

```
q_low  = (1.883e4 / (D^α G^(1/6))) · (A/G^(1/6) − x)
q_high = (3.78e4 · B / (D^α G^0.6)) · (1 − x)
CHF    = max(q_low, q_high)

α = 0.6 for D < 1 cm, else 0.4
A = 0.7249 + 0.099 P exp(−0.032 P)
B = −1.159 + 0.149 P exp(−0.019 P) + 8.99P/(10 + P²)
```

**Structural note for modelling:** the `G^(1/6)` terms in the denominator **diverge as
G → 0**, which is why the repo's `hybrid_reference_chf` dispatches to Zuber at G = 0.
That is a correctness requirement, not a nicety — and it is a concrete example of why a
naive "just use a correlation as a prior" approach breaks at regime boundaries.

### 5.5 Katto & Ohno (1984) generalised correlation

The most physically-grounded widely-used flow-boiling correlation; derived by *vectorial
dimensional analysis* postulating hydrodynamic control, yielding four regimes (L, H, N,
HP). Recommended by the NASA cryogenic study as the best performer for LH₂ flow boiling
(91 points, within a factor of 2 more than 90% of the time). **[V]** (transcribed from
NASA NTRS 20230009827)

Master form, with inlet subcooling:

```
q″_CHF = q″_co [ 1 + K(h_f,sat − h_f)/h_fg ]                               (8)
```

Sub-correlations (`We⁻¹ ≡ σρ_f/(G²L)`):

```
q″_co,2  = C_Kc (σρ_f/(G²L))^{0.043} (D_H/L) G h_fg                        (9)   [S: C_Kc]

q″_co,3  = 0.1 (ρ_g/ρ_f)^{0.133} (σρ_f/(G²L))^{0.333} · G h_fg / (1 + 0.0031 L/D_H)   (10)

q″_co,4  = 0.098 (ρ_g/ρ_f)^{0.133} (σρ_f/(G²L))^{0.433} (L/D_H)^{0.27} · G h_fg / (1 + 0.0031 L/D_H)   (11)

q″_co,5  = 0.0384 (ρ_g/ρ_f)^{0.6} (σρ_f/(G²L))^{0.173} · G h_fg / (1 + 0.28(σρ_f/(G²L))^{0.233} L/D_H)  (12)

q″_co,13 = 0.234 (ρ_g/ρ_f)^{0.513} (σρ_f/(G²L))^{0.433} (L/D_H)^{0.27} · G h_fg / (1 + 0.0031 L/D_H)   (13)
```

Subcooling factors:

```
K6 = 1.043 / (4 C_Kc (σρ_f/(G²L))^{0.043})                                 (14)  [S: C_Kc]
K7 = (5/6) · [ 0.0124 + D_H/L ] / [ (ρ_g/ρ_f)^{0.133} (σρ_f/(G²L))^{0.333} ]   (15)
K9 = 1.12 · [ 1.52(σρ_f/(G²L))^{0.233} + D_H/L ] / [ (ρ_g/ρ_f)^{0.6}(σρ_f/(G²L))^{0.173} ]   (16)
```

Regime selection — **the interesting part** **[V]**:

```
q″_co = min(q″_co,2, q″_co,3, q″_co,4)     if ρ_g/ρ_f < 0.15
      = min(q″_co,2, q″_co,5, q″_co,13)    if ρ_g/ρ_f > 0.15               (17)

K     = max(K6, K7)                        if ρ_g/ρ_f < 0.15
      = max(K6, K7, K9)                    if ρ_g/ρ_f > 0.15               (18)
```

> **`ρ_g/ρ_f = 0.15` is a real, physically-motivated regime boundary** — the density ratio
> at which the controlling CHF mechanism changes. It is a natural gate for a mixture-of-
> experts architecture, and it comes free from the physics rather than from cross-validation.
> Note also that the whole correlation is expressed in `Bo`, `σρ_f/(G²L)`, `ρ_g/ρ_f` and
> `L/D` — exactly the four-group space of §2.2.

**[S] flag:** `C_Kc` is the Katto–Ohno length-dependent constant (a piecewise function of
`L/D`). Its value did not extract cleanly from the OCR and **must be read from Katto &
Ohno (1984), *Int. J. Heat Mass Transfer* 27(9):1641** before implementation.

### 5.6 Shah (1987)

The UCC (upstream conditions correlation) version **[V]**:

```
Bo = q″_CHF/(G h_fg) = 0.124 (D_H/L_E)^{0.89} (10⁴/Y)^n (1 − x_iE)         (19)

L_E  = L                        if x_in ≤ 0
     = L + D_H x_in/(4 Bo)      if x_in > 0                                (20)

x_iE = x_in                     if x_in ≤ 0
     = 0                        if x_in > 0                                (21)

Y = (G D_H c_p,f / k_f) · (ρ_f 2 g D_H / G²)  ...                          (22)  [S: partial OCR]

F3 = (1.25×10⁵/Y)^{0.833 x_eq}                                             (31)
c  = 0 if P/P_cr ≤ 0.6; 1 if P/P_cr > 0.6                                  (32)
```

Shah's own selection rule **[V]**: use the UCC method when `Y ≤ 10⁶` or
`L_E > 160/(P/P_cr)^{1.14}`; otherwise use whichever version yields the **lower** `Bo`
(i.e. the conservative branch). Described in the source as "based on a vast amount of data
and can be applied to various fluids."

**[S] flag:** the full `Y` definition and Eqs. 23–30/33–36 did not extract cleanly.

### 5.7 Hall & Mudawar (2000) — the cleanest dimensionless correlation

**This is the most important correlation in this document for our purposes**, because it
is fully dimensionless, has only five constants, and — uniquely — its functional form was
*derived from observed parametric trends* rather than guessed and regressed. From Hall &
Mudawar, *IJHMT* 43:2605 **[V]**:

**Outlet (local) conditions form:**

```
Bo = C₁ We_D^{C₂} (ρ_f/ρ_g)^{C₃} [ 1 − C₄ (ρ_f/ρ_g)^{C₅} x_o ]            (7)

C₁ = 0.0332,  C₂ = −0.235,  C₃ = −0.681,  C₄ = 0.684,  C₅ = 0.832
```

with `Bo = q″/(G h_fg)`, `We_D = G²D/(ρ_f σ)`.

**Inlet conditions form** (obtained by substituting the energy balance, §1.5) **[V]**:

```
             C₁ We_D^{C₂} (ρ_f/ρ_g)^{C₃} [ 1 − C₄(ρ_f/ρ_g)^{C₅} x_i* ]
Bo  =  ───────────────────────────────────────────────────────────────────    (11)
              1 + 4 C₁ C₄ We_D^{C₂} (ρ_f/ρ_g)^{C₃+C₅} (L/D)
```

All saturated properties at outlet pressure. Reported accuracy: **MAE 10.3%, RMS 14.3%**
for the inlet form — *more accurate than the CHF look-up table* on their database.

Validity **[V]**: `0.25×10⁻³ ≤ D ≤ 15×10⁻³ m`, `2 ≤ L/D ≤ 200`,
`300 ≤ G ≤ 30,000 kg m⁻²s⁻¹`, `1×10⁵ ≤ P ≤ 200×10⁵ N m⁻²`, `−2.0 ≤ x_i ≤ 0.0`,
`−1.0 ≤ x_o ≤ 0.0` (subcooled outlet; ~85% of their subcooled database).

The authors' own claim about *why* it generalises is the thesis of this document **[V]**:

> "Superiority of the correlations was attributed to the **systematic development of the
> functional forms of the correlations from the CHF parametric effects**; thus,
> re-optimization of the constants, when additional subcooled CHF data become available,
> is **not expected to produce an appreciable increase in accuracy**."

> **Read that again.** A five-constant physics-derived form that does not improve when
> refitted on more data — that is the definition of a model that has captured the physics
> rather than the dataset. This is precisely the property our surface-wise and LOSO splits
> are trying to measure, and it is the target our proposed algorithm should aim at.

### 5.8 W-3 (Tong) and Bowring — **[X] NOT TRANSCRIBED**

Both are heavily cited (W-3 is the standard PWR DNB correlation; Bowring 1972 is a
standard tube correlation) and both appear in the ML-CHF literature as baselines. **No
clean primary source was obtained during this research** — the accessible sources
described them without giving full constants.

**Do not write these into the paper from memory.** Obtain:
- Tong, L.S. (1967/1972), W-3 correlation — via the Westinghouse justification document
  `ML17254A842.pdf` on nrc.gov, or Todreas & Kazimi *Nuclear Systems I*.
- Bowring, R.W. (1972), AEEW-R789.
- Tong's **F-factor** for non-uniform axial heat flux — MIT OCW 22.313J has a note on the
  shape factor (`chf_shape.pdf`).

### 5.9 Analytical/mechanistic flow-boiling models

NUREG/KM-0011 §3.2 divides them into two families **[V]**:

**(1) Annular film dryout models** — mass balance on the liquid film; CHF = film
depletion. Originated by Hewitt and co-workers (Hewitt & Hall-Taylor 1970; Bennett et al.
1967). Models differ in their closures for **droplet entrainment, deposition, interfacial
friction and interfacial heat transfer**. Good for annular flow at medium-to-high pressure
and flow, void fraction > 50%.

**(2) Bubbly layer models** — CHF at low quality when the near-wall bubble layer becomes
so thick and vapour-saturated that liquid mixing with the cooler core stops. Tong (1965,
1968), Tong & Hewitt (1972), **Weisman & Pei (1983)**, Ying & Weisman (1986). Good for
high-pressure, high-flow, low-quality/subcooled conditions.

NUREG's honest assessment, which we should quote **[V]**:

> "CHF models tend to be **less accurate than empirical correlations** over the range of
> the correlation's database... the evaluation process is complex and time consuming."

There are 50+ such models; Weisman (1992) reviewed them.

**(3) Liquid sublayer dryout** — the third family, favoured in the modern DNB literature
because of direct experimental evidence in internally heated round tubes. DNB is caused by
complete evaporation of a thin superheated liquid layer beneath a vapour blanket sliding
along the wall. Heat balance (Zhao et al. 2019, Eq. 2) **[V]**:

```
CHF = ρ_f δ h_fg U_B / L_B
```

`δ` = liquid sublayer thickness, `U_B` = vapour blanket velocity, `L_B` = vapour blanket
length. The whole modelling problem reduces to closing these three quantities. The **Liu
model** is the most recent in the series, analysing instabilities at both the
sublayer/blanket and blanket/bulk interfaces. Reported limitations **[V]**: closure
relations exist essentially only for **round tubes**, and the model is **deficient at low
subcooling** — and it is outperformed by the purely data-driven LUT.

> Note the pattern across §5.9: mechanistic models are more *transferable* in principle
> but less *accurate* in practice, because their closures are themselves empirical and
> tube-specific. This is the core tension our project sits in, and the honest framing for
> the paper's contribution.

---

## 6. Geometry-specific physics

### 6.1 Helical coils

Physically distinct from straight tubes because of **centrifugal force and secondary
flow**. Hardik & Prabhu (2017), *Applied Thermal Engineering* 112:1223 **[V]**:

Mechanism: phase separation, with low-density vapour driven to the **inner** side and
high-density liquid to the **outer** side of the curved tube; a thin liquid film forms
circumferentially under combined gravity and centrifugal forces. Result: large
circumferential wall-temperature variation and inner-side film dryout. Their IR
measurements confirm **inner-half wall temperature exceeds outer-half**.

Observed trends **[V]**: CHF increases with mass flux; decreases with quality; for the
same quality, decreases with increasing tube diameter *and* coil diameter. **In the
high-quality region CHF in a helical coil exceeds that in a vertical straight tube** — by
up to ~4× versus a straight *horizontal* tube.

**Merilo-type dimensionless correlation** (their Eq. 7) **[V]**:

```
q″/(G h_fg) = C · Re_l^{−0.34} · Z³ · Bo_d^{0.358} · (μ_l/μ_g)^{−2.18} · (L_h/d)^{−0.511}
              · (ρ_l/ρ_g − 1)^{1.27} · (1 − x_i)^{1.64}                    (7)

Re_l = G d/μ_l ,   Z = μ_l/(σ d ρ_l)^{0.5} ,   Bo_d = (ρ_l − ρ_g) g d²/σ ,
x_i = c_p(T_i − T_sat)/h_fg
```

with the constant `C` carrying **all** the geometry/regime information **[V]**:

```
C = 575    Merilo — high-pressure horizontal tubes                          (8)
C = 25.79  Baburajan et al. — low-pressure horizontal tubes                 (9)
C = 110    Hardik & Prabhu — low-pressure helical coils                     (10)
```

**LUT-normalised form** — the most useful result for us **[V]**:

```
CHF_coil / CHF_LookUpTable = 1.637 x + 0.568                               (11)
```

90% of data within 25%; RMS deviation 17%; average deviation −1.6%.

Physical explanation given **[V]**: at **low quality**, gravity dominates and coil CHF is
*below* a vertical tube; at **high quality**, liquid-film redeposition by secondary flow
dominates and coil CHF *exceeds* the vertical tube — hence the positive slope in `x`.

> **Two lessons.** (a) Eq. (11) is a one-line, physically-explained transfer function from
> straight-tube physics to coil physics — exactly the kind of structure a generalised
> model should contain explicitly rather than relearn. (b) The `C`-only variation across
> Eqs. (8)–(10) shows the *dimensionless shape* transfers across geometry and pressure
> while only the *scale* changes. That is the strongest single piece of evidence in this
> document for the §9 architecture.

Our `helical_coil_r123` source (257 rows) is the LOSO fold that went from R² = −409 to
+0.24 with physics normalisation. Adding curvature ratio `d/D_coil` as a feature and
Eq. (11) as a prior is the obvious next step.

### 6.2 Mini/micro-channels

Kandlikar's framework: surface tension, evaporation momentum, viscous shear and inertia
govern the flow patterns and CHF. Two new groups **[S]** (structure from secondary
sources; **verify against Kandlikar 2001/2004 before use**):

```
K₁ = (q″/(G h_fg))² (ρ_f/ρ_g)      — evaporation momentum / inertia
K₂ = (q″/h_fg)² D/(ρ_g σ)          — evaporation momentum / surface tension
K₃ = ...                            — evaporation momentum / viscous
```

with `K₂/K₁ = We`, `K₃/K₁ = Re`, `K₂/K₃ = Ca`. A high `K₁` means evaporation momentum
dominates and will alter interface motion.

Kandlikar's physical picture for CHF: a large bubble on the surface subject to
**momentum force** (expanding), opposed by **surface tension and hydrostatic forces**;
when the balance fails the bubble advances and covers the surface → CHF. This is the same
force balance as his pool-boiling model (§4.3).

### 6.3 Pin fins / structured pool-boiling surfaces

No closed-form correlation obtained. Our `pinfin_chf_water_fc72` source (175 rows) carries
`fin_shape`, `fin_array`, `fin_width_um`, `fin_height_um`, `fin_spacing_um`, `coverage`,
`porosity`, `mbl_lateral_um`, `mbl_total_um` — these are **wickability geometry** (§4.4)
and should be modelled through a wicking/capillary framework, plus the finite-heater-size
correction of §3.3, rather than as generic tabular features. This is the LOSO fold still
sitting at R² = −0.04.

---

## 7. What ML has and hasn't done — the gap, sourced

For the outline's §"Existing ML-based CHF prediction" and §"Research gap".

**The benchmark context.** The OECD/NEA established a Task Force on AI/ML for Scientific
Computing in Nuclear Engineering (2022), whose **Phase 1 CHF exercise** (NEA/WKP(2023)1,
Le Corre, Delipei, Wu, Zhao) covers regression, classification, VVUQ, dimensionality
reduction and anomaly detection, built on a US NRC-provided CHF database. **[V]** The
24,579-point NRC database at the core of it is the same `nrc_groeneveld_24579pt` source
that is 86% of our merged dataset.

**The gap, in the words of the primary sources:**

Zhao et al. (2019), MIT/ORNL **[V]**:
- Standalone ML tools "can be prone to undesired, **unphysical solutions** due to their
  purely data-driven nature and 'black-box' feature."
- Of prior CHF-ML publications: none distinguished DNB from dryout; **none employed
  cross-validation** to assess architectures; none evaluated hyperparameter sensitivity or
  discussed regularisation.

Their **hybrid ("grey-box") framework**, in two flavours **[V]**:
- **Series:** ML estimates intermediate variables that are closures in a physics model
  (e.g. `δ`, `U_B`, `L_B` in the sublayer-dryout model).
- **Parallel** (their choice, = physics-informed ML-aided): a fixed-structure prior model
  `ŷ_p = f(x)` produces a baseline; ML is trained on the **residual** `ε = y − ŷ_p`;
  prediction is `ŷ_h = ŷ_p + ε̂_m`.

Their feature vector is at least six variables: `P, G, x_e, D_e, D_h, L_h` — note **both**
equivalent and heated diameter. Metric **[V]**:

```
rRMSE = sqrt( (1/n) Σ ( (y⁽ⁱ⁾ − ŷ_h⁽ⁱ⁾)/y⁽ⁱ⁾ )² )                          (1)
```

Prior models used: Groeneveld 2006 LUT, and the mechanistic Liu model. They also propose
**window-type extrapolation mapping** to decide where new (expensive) experiments are
actually needed.

For non-circular channels, they note **heated diameter `D_h` rather than hydraulic
diameter `D_e`** is recommended for the diameter correction, "as the former feature better
describes vapor formation and development in subcooled and low-quality flow." **[V]**

> **Two direct implications for our pipeline.** (a) We use `diameter_mm` with no
> `D_h`/`D_e` distinction — for the annulus (378 rows) and plate (48 rows) families this
> is a known, sourced modelling error. (b) Our repo already implements the *parallel*
> hybrid (residual learning on Biasi/Zuber) and found it fails on Split C because "the
> formula's error changed shape completely" outside the training range. That negative
> result is real and worth publishing — and it argues for *multiplicative* rather than
> *additive* residuals (§9), which is what the unified pipeline's `log(CHF/baseline)`
> target already does successfully.

---

## 8. Physical constraints any admissible CHF model must satisfy

**This is the operational payload of the document.** Each item is a testable assertion
derived from the physics above, usable as (a) a validation check, (b) a training penalty,
or (c) an architectural constraint. Nothing here requires new data.

### 8.1 Hard constraints (violation = the model is wrong)

| # | Constraint | Source |
|---|---|---|
| C1 | `CHF > 0` strictly, everywhere in the domain. | definition |
| C2 | **Energy balance:** `x_o = x_i* + 4·Bo·(L/D)` must hold exactly. `x`, `L/D`, `G`, `q″` are not independent. | §1.5 **[V]** |
| C3 | `∂CHF/∂x ≤ 0` — CHF decreases monotonically with quality, in every regime and geometry examined. | §5, §6.1 **[V]** |
| C4 | `∂CHF/∂G ≥ 0` in flow boiling (all helical-coil and tube sources agree). | §6.1 **[V]** |
| C5 | As `G → 0`, the flow-boiling solution must approach the **pool-boiling** solution (Zuber/Kutateladze), not diverge. Biasi's `G^{1/6}` denominators violate this. | §5.4 **[V]** |
| C6 | As `P → P_crit`: `σ → 0`, `h_fg → 0`, `ρ_f − ρ_g → 0`, therefore `CHF → 0`. No feature may diverge here. | §2.1 |
| C7 | Dimensional homogeneity: predictions must be invariant under a change of unit system. Only satisfiable if the model operates on dimensionless groups. | §2.2 |

Constraint **C6** is exactly the failure mode already documented in
`docs/CHF_Physics_Approaches_Explained.md`: the `ln(1 − P/P_crit)` feature exploded above
16,000 kPa and drove Split-C predictions to negative thousands. The physics said in
advance that this feature was inadmissible.

### 8.2 Soft constraints (strong expectations; deviations need explanation)

| # | Constraint | Source |
|---|---|---|
| S1 | Pool-boiling CHF vs. pressure is **non-monotonic with a maximum near `P/P_c ≈ 0.35`**. | §4.6 **[V]** |
| S2 | Dimensionless `K` for saturated pool boiling on a large horizontal surface lies in **0.119–0.157** (Zuber's bracket), extending to ~0.13–0.2 across the literature. A model predicting `K = 0.5` or `K = 0.02` is reporting a different phenomenon. | §3.2, §4.6 **[V]** |
| S3 | CHF **decreases** with decreasing wettability (increasing contact angle); at α = 107° it can be **half** the hydrodynamic prediction. | §4.1 **[V]** |
| S4 | CHF **increases** with roughness where wicking is active, then **saturates** (no further gain for `R_a` ≳ 0.35 µm on superhydrophilic surfaces). Not monotonic without bound. | §4.2 **[V]** |
| S5 | CHF decreases **slightly** for orientation 0°→90°, then **sharply** 90°→180°. Three distinct mechanism regions. | §4.3 **[V]** |
| S6 | Diameter effect **reverses sign** at low mass flux (`n < 0` for G < 250) and at high quality. A single global `D^{−0.5}` is wrong. | §5.3 **[V]** |
| S7 | Helical-coil CHF is **below** straight-tube at low quality and **above** it at high quality; the crossover is real, not noise. | §6.1 **[V]** |
| S8 | For finite heaters, Zuber applies only for width **> 3λ_d** (or > 12 capillary lengths, or > 2, depending on source). Below that, expect systematic deviation. | §3.3 **[V]** |
| S9 | Regime boundary at `ρ_g/ρ_f = 0.15` (Katto–Ohno) separates two different controlling mechanisms. | §5.5 **[V]** |
| S10 | Wall thermal activity `S` matters in **pool** boiling (asymptote at S ≈ 10–85) and not in **flow** boiling. | §4.5, §1.4 **[V]** |

### 8.3 Data-integrity constraints specific to our dataset

| # | Issue | Evidence |
|---|---|---|
| D1 | CHF label definition varies by source (4+ detection criteria). No `detection_criterion` column exists. | §1.3 |
| D2 | Contact angle: **no column**. Roughness: 0.6% coverage, and no peak-spacing `S_m` — Kim et al. Eq. (21) needs both. | §4.2 |
| D3 | No `D_h` vs `D_e` distinction; known to matter for annulus/plate rows. | §7 |
| D4 | No wall thickness → cannot compute thermal activity `S` for pool-boiling rows. | §4.5 |
| D5 | No flow-instability / throttling flag, though throttling causes "significant" CHF reduction and dissolved gas up to 30%. | §1.4 |
| D6 | Inter-laboratory reproducibility on nominally identical test sections is ~**10%** (European 1970 exercise, 8–10 labs; plus the U.S.S.R. 1984/85 study). **This is the irreducible noise floor.** Any reported MAPE below ~10% on pooled multi-source data is measuring overfitting, not accuracy. | NUREG §2.4 **[V]** |

> **D6 deserves emphasis in the paper.** Our strategy-1 random-split results (R² ≈ 0.96–0.97)
> sit well inside the inter-laboratory reproducibility band. That is not a triumph; it is
> a sign the model is fitting source-specific bias. The surface-wise and LOSO numbers,
> unflattering as they are, are the physically meaningful ones.

---

## 9. Proposed algorithm — a physics-first architecture (discussion draft)

This section is a **proposal for discussion**, not an implemented design. It follows from
§8 and from the one empirical fact our own pipeline has already established: dividing by a
physics baseline before log-space training moved strategy 3 from R² = −16.4 to +0.22 and
LOSO `helical_coil_r123` from −409 to +0.24. **The physics normalisation did more than
every model-selection decision combined.** The proposal is to take that principle
further rather than to add more model capacity.

### 9.1 The core idea

Stop predicting CHF. Predict a **dimensionless correction to a physics scale**:

```
CHF = Φ_physics(x) · exp( g_θ(π(x)) )
```

- `Φ_physics` — a closed-form, regime-dispatched physical scale, computed from CoolProp
  properties with **zero fitted parameters**.
- `π(x)` — the **dimensionless** feature vector (§2.2), never raw engineering units.
- `g_θ` — a learned, *bounded*, smooth correction in log space.

This is multiplicative, not additive — which matters, because the repo's additive
residual experiment (Approach 2) failed on Split C for exactly the reason the physics
predicts: an additive correction learned inside the domain has the wrong *magnitude* when
the baseline itself changes by an order of magnitude outside it. A multiplicative
correction is scale-free, which is why the `log(CHF/baseline)` target already works.

### 9.2 Layer 1 — the physics scale `Φ_physics`

Regime-dispatched, with **smooth blending** (Yagov's cube-root blend, §3.5, rather than
hard switches):

| Regime | Scale | Source |
|---|---|---|
| Pool boiling, large heater | Zuber Eq. (15), `K = 0.131` | §3.2 |
| Pool boiling, finite heater | × Lienhard–Dhir / size correction | §3.3 |
| Pool boiling, structured surface | × Kandlikar `K(α, θ)` and/or Kim `K(α, R_a, S_m)` | §4.3, §4.2 |
| Flow boiling, subcooled | Hall & Mudawar Eq. (7)/(11) | §5.7 |
| Flow boiling, saturated | Katto–Ohno Eq. (17) with the `ρ_g/ρ_f = 0.15` gate | §5.5 |
| Helical coil | straight-tube scale × `(1.637x + 0.568)` | §6.1 |
| Any tube, `D ≠ 8mm` | × `K1 = (8/D)^n`, `n` from the Tanase regime table | §5.3 |

Every one of these is closed-form and needs no training data for the target fluid. That
is the property that makes cross-fluid and cross-surface generalisation possible at all.

### 9.3 Layer 2 — the dimensionless feature map `π(x)`

Replace `(P, G, x, D, L)` with:

```
π = [ Bo_ref,  We_D,  ρ_f/ρ_g,  P/P_c,  x,  L/D,  D/D_ref,
      Ja = ρ_f c_p ΔT_sub/(ρ_g h_fg),
      Bo_d = (ρ_f−ρ_g)g D²/σ,
      d/D_coil            (curvature, coils)
      cos α, θ, R_a/S_m, wicking flux   (surface, where available)
      S = H√((ρc_p k)_w)  (pool boiling, where available) ]
```

Rationale: NUREG's own general form (§2.2) says a tube CHF correlation *is* a function of
`(ρ_f/ρ_g, D^{0.5}/(σρ_f)^{0.5}, x, D/D_ref)`. Katto–Ohno, Hall–Mudawar and Merilo are
all written in exactly this space. A model in this space can map water at 15 MPa and R123
at 1 MPa onto the same point; a model in raw units cannot.

### 9.4 Layer 3 — the constrained learner `g_θ`

- **Bounded output.** `g_θ ∈ [−ln 3, +ln 3]` (a hard tanh), i.e. the correction may not
  move the answer by more than ~3×. This replaces the current post-hoc
  `CHF_CLIP_MAX = 30,000` clip — which is a symptom-level fix for the ANN blow-up that
  turned one fold's R² into −1,600,000 — with a structural guarantee.
- **Monotonicity in `x` (C3)** and in `G` (C4), enforced either by a monotone architecture
  (monotone lattice / constrained GBM) or by the autograd penalty already implemented in
  the repo's collocation network.
- **Energy-balance consistency (C2)** as a training penalty on `x_o − x_i* − 4Bo(L/D)`.
- **Trust decay.** `g_θ → 0` as the input moves away from the training manifold, so
  predictions gracefully fall back to pure physics out-of-domain instead of extrapolating
  a learned correction. This directly addresses the documented Approach-2 failure.

### 9.5 Layer 4 — honest evaluation

Keep the existing four strategies, and add the §8 checks as a **physics-consistency
scorecard** reported alongside R²/RMSE/MAE/MAPE:

- Fraction of predictions violating C1–C7.
- Does the model reproduce the `P/P_c ≈ 0.35` pool-boiling maximum? (S1)
- Is predicted `K` inside 0.119–0.157 for saturated pool boiling on large flat heaters? (S2)
- Does the helical-coil crossover (S7) appear?
- Sign of the diameter effect at low `G` (S6).

> A model that scores R² = 0.95 while violating C3 and missing S1 is worse than a model at
> R² = 0.85 that satisfies both, because only the latter can be trusted at a condition
> nobody has tested. **That argument — not a leaderboard number — is the paper's
> contribution**, and it maps directly onto the outline's Gap 4.

### 9.6 Honest assessment of this proposal

- It will **almost certainly lower** strategy-1 (random split) scores. Constraints cost
  in-distribution accuracy. That trade is the point, but it must be stated, not hidden.
- It cannot manufacture the missing data (D1–D5). The surface-characteristics claims in
  the outline remain unsupportable until contact angle and roughness coverage improve well
  beyond 0.6%.
- Several constants are still **[S]** or **[X]** (Katto–Ohno `C_Kc`, Shah's `Y`, Kandlikar's
  `K₁`/`K₂`, W-3, Bowring, the Rahman wickability coefficient). These must be read from
  primary sources before any of it is implemented or published.
- The biggest risk is that `Φ_physics` is *itself* wrong for the exotic regimes (pin-fin,
  R123 coils) — in which case we have moved the error rather than removed it. The
  physics-consistency scorecard is what will detect that.

---

## 10. Sources

### 10.1 Read in full during this research (primary)

| Source | Location | What it provided |
|---|---|---|
| Liang & Mudawar (2018), *IJHMT* 117:1352, "Pool boiling CHF Part 1: mechanisms, models, correlations" | fetched from Purdue | §2.1, §3 (all five mechanisms), §4 — the single richest source here |
| NUREG/KM-0011 (`ML19029B306.pdf`) | `docs/references/` | §1.1–1.4, §1.6, §2.2, §5.1, §5.2, §5.9, D6 |
| Hall & Mudawar (2000), *IJHMT* 43:2605, Part II subcooled correlations | fetched from Purdue | §1.5, §5.7 |
| Hall & Mudawar (2000), *IJHMT* 43:2573, Part I database (`Critical_heat_Flux.pdf`) | `docs/references/` | database scope, CHF definition |
| Hardik & Prabhu (2017), *Appl. Therm. Eng.* 112:1223 | `docs/references/` | §6.1 |
| Tanase et al. (2009), *NED* 239:289, diameter effect | `docs/references/` | §5.3 diameter exponent table |
| Zhao, Shirvan, Salko & Guo (2019), arXiv:1906.11124 | fetched | §7, §5.9(3) |
| NASA NTRS 20230009827, cryogenic CHF correlations | fetched | §5.5, §5.6 |
| arXiv:2203.15048, VVER rod bundle CHF methodology | fetched | §5.3 K1–K8 |
| NEA/WKP(2023)1 Phase 1 CHF benchmark specs | `docs/references/` | §7 benchmark context |
| Rashidi et al. (2022), `Applications_ML.pdf` | `docs/references/` | ML-boiling review, dimensionless input sets |
| `scripts/chf_physics.py` (this repo) | — | §5.4 Biasi as implemented, Zuber implementation |

### 10.2 Consulted but not fully mined

`CHF_Look_Up_Table.pdf`, `ML22264A009.pdf` (457 pp), `Critical Heat Flux of Flowing Water.pdf`,
`Critical heat flux prediction through machine learning model for narrow.pdf`,
`Prediction of "critical heat flux" for supercritical water and CO2.pdf`,
`A Three-Stage Bayesian Transfer Learning Framework.pdf`,
`Boiling_pressure_drop...straight_tubes.pdf`, `CRITICAL_HEAT_FLUX_CHF_AND_POST-CHF_HEAT.pdf`,
`sample_reference.pdf`, `external_coil_tube_chf_appendix.pdf`,
`helical_coil_chf_research_data.pdf`.

Extracted text for all of the above is in the session scratchpad; re-extract with
`pypdf` if needed.

### 10.3 Must be obtained before publication

1. **Katto & Ohno (1984)**, *IJHMT* 27(9):1641 — the `C_Kc` constant. **[S]**
2. **Shah (1987)** — full `Y` definition and Eqs. 23–36. **[S]**
3. **Kandlikar (2001)**, *J. Heat Transfer* 123(6):1071 — pool-boiling force balance
   (the `K` form in §4.3 is verified; the derivation is not). **[V]**/**[S]**
4. **Kandlikar (2004)**, *J. Heat Transfer* 126(1):8 — `K₁`, `K₂`, `K₃` definitions. **[S]**
5. **Tong W-3** and **Bowring (1972) AEEW-R789** — full forms. **[X]**
6. **Tong F-factor** for non-uniform axial flux (MIT OCW 22.313J `chf_shape.pdf`). **[X]**
7. **Rahman, Ölçeroğlu & McCarthy**, *Langmuir* 2014 / *Sci. Rep.* 2017 — the linear
   CHF–wickability coefficient. **[S]**
8. **Groeneveld et al. (2007)**, *NED* 237:1909 — the K-factor formulas as given by the
   original authors (§5.3 is transcribed from a secondary source that cites IAEA-TECDOC-1203).
9. **Weisman & Pei (1983)**, *IJHMT* 26(10):1463 — bubble-crowding model equations. **[X]**
10. **Liu et al.** liquid sublayer dryout closures for `δ`, `U_B`, `L_B`. **[X]**

### 10.4 Web sources used

- Liang & Mudawar pool boiling CHF review — https://engineering.purdue.edu/mudawar/files/articles-all/2018/2018-01.pdf
- Hall & Mudawar subcooled CHF correlations — https://engineering.purdue.edu/mudawar/files/articles-all/2000/2000_2.pdf
- NASA cryogenic CHF — https://ntrs.nasa.gov/api/citations/20230009827/downloads/CHF%20Paper_submitted.pdf
- VVER rod bundle CHF methodology — https://arxiv.org/pdf/2203.15048
- Zhao et al. physics-informed ML CHF — https://arxiv.org/pdf/1906.11124
- Kandlikar pool boiling CHF model (abstract) — https://asmedigitalcollection.asme.org/heattransfer/article-abstract/123/6/1071/445633/A-Theoretical-Model-to-Predict-Pool-Boiling-CHF
- Rahman et al. wickability (abstract) — https://pubs.acs.org/doi/full/10.1021/la5030923
- CHF wickability paradigm (abstract) — https://www.nature.com/articles/s41598-017-05036-2
- Katto & Ohno improved generalized correlation (abstract) — https://www.sciencedirect.com/science/article/abs/pii/001793108490276X
