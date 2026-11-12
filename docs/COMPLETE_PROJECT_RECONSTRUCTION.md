# Critical Heat Flux Prediction — Complete Project Reconstruction

**A self-contained technical research diary, methodology document, literature review, development history, and final pipeline documentation.**

Prepared for a reader with **zero prior knowledge** of this project.

Date of this reconstruction: 14 September 2026
Project: B.Tech Project (BTP), Critical Heat Flux prediction using physics-anchored machine learning

---

## HOW TO READ THIS DOCUMENT

This document is long on purpose. It is written so that a new researcher joining the project tomorrow can read it from top to bottom and never need to ask "what does that mean?" or "why did they do that?"

### Evidence labels used throughout

Every substantive claim in this document carries one of these labels so you always know how much to trust it:

| Label | Meaning |
|---|---|
| **[LITERATURE]** | Established by previous researchers in published work. |
| **[LITERATURE — UNVERIFIED]** | Appears in our own literature notes but was assembled by topic/title matching and has **not** been confirmed against the full text of the paper. Do not quote these numbers to anyone. |
| **[OUR EXPERIMENT]** | Something we actually ran and measured. A number is attached and the file it came from is named. |
| **[OUR INTERPRETATION]** | Our explanation of an observation. Could be wrong. |
| **[OUR PROPOSED IDEA]** | Something we proposed but have not validated. |
| **[ASSUMPTION]** | Something the pipeline currently assumes without proof. |
| **[OPEN QUESTION]** | Unresolved. |

### Where the numbers come from

Every number labelled **[OUR EXPERIMENT]** in this document can be traced to one of these files in the project repository:

| File | What it contains |
|---|---|
| `results/prof_tasks/FINAL_results.csv` | The definitive per-dataset results across all four split types. Newest and most authoritative. |
| `results/prof_tasks/final_review.log` | Human-readable version of the above, plus the stress-test results. |
| `results/prof_tasks/pipeline_loso.csv` | Pipeline versus pure physics versus pooled machine learning, on held-out datasets. |
| `results/prof_tasks/pipeline_coverage.csv` | How often the prediction intervals actually contained the true answer. |
| `results/prof_tasks/grid_results.csv` | The big model comparison: 804 runs of 13 machine-learning model families. |
| `results/prof_tasks/physics_results.csv` | 157 runs of 9 physics and hybrid approaches. |
| `results/prof_tasks/leakage_evidence.csv` | Proof that certain data columns contain the answer. |
| `results/prof_tasks/exhibitA_energy_balance.csv` | Proof that the energy-balance identity reconstructs the target. |
| `results/prof_tasks/improve_ablation.csv`, `improve_round2.csv` | Eleven attempts to improve the pipeline, and their results. |
| `docs/CONFOUNDING_FINDINGS.md` | The laboratory-identity confounding analysis. |

### A warning about one section

**Section 7 (Literature Review)** contains a table inherited from an earlier stage of the project. Most of its entries are explicitly flagged **[LITERATURE — UNVERIFIED]**. Those rows were assembled by matching paper titles and topics, **not** by reading the full texts. They are included because they shaped our thinking at the time, but the specific numbers in them must be confirmed before they appear in any submitted manuscript. This is stated plainly rather than hidden.

---

# PART I — STAGE 0: THE ORIGINAL RESEARCH PROBLEM

## 1.1 What is boiling, physically?

Start with the simplest possible picture. You have a metal surface. You heat it. You put water on it. The water carries heat away from the metal.

How well the water carries heat away depends on **how** the water is boiling. There are several distinct regimes, and they behave completely differently.

**Regime 1 — Single-phase convection.** The surface is warm but below the boiling point of the liquid. The liquid simply flows past and warms up. Heat transfer is modest and entirely predictable.

**Regime 2 — Nucleate boiling.** The surface is now hotter than the liquid's boiling point. Tiny vapour bubbles form at microscopic pits and scratches on the surface (these are called *nucleation sites*), grow, detach, and rise away. This is extremely efficient at removing heat, for two reasons. First, turning liquid into vapour absorbs a large amount of energy (the *latent heat of vaporisation*). Second, each departing bubble violently stirs the liquid next to the wall, dragging cool liquid in to replace it. Nucleate boiling is the regime every heat-transfer device is designed to operate in.

**Regime 3 — Film boiling.** So many bubbles are now being produced that they merge into a **continuous blanket of vapour** that completely covers the surface. Liquid can no longer touch the metal. Vapour is a terrible conductor of heat compared to liquid. Heat transfer collapses.

**The Critical Heat Flux is the boundary between Regime 2 and Regime 3.**

## 1.2 Why this boundary is dangerous, not merely inefficient

Here is the crucial point, and it is the reason this entire field exists.

The transition from nucleate boiling to film boiling is **not gradual and not gentle**. Once the vapour blanket forms, the metal surface loses its cooling almost instantly. But the heat source — a nuclear fuel rod, an electric heater — is still producing heat at the same rate. The heat has nowhere to go.

The result is that the metal temperature **jumps by several hundred degrees Celsius in a fraction of a second**. In the heat-transfer literature this event has names that convey how serious it is:

- **Burnout** — because the surface can literally melt or rupture.
- **Boiling crisis** — the general term.
- **Departure from Nucleate Boiling (DNB)** — the name used when it happens in subcooled or low-quality flow, typical of pressurised water reactors.
- **Dryout** — the name used when it happens in high-quality flow, typical of boiling water reactors.

In a nuclear reactor, the fuel is uranium dioxide pellets sealed inside metal tubes called **cladding**. If the cladding overheats it can balloon, rupture, and release radioactive material into the coolant. Preventing this is a primary safety requirement.

Therefore reactors are not operated at the Critical Heat Flux. They are operated at a **fraction** of it, with a safety factor. That fraction is quantified by the **Critical Heat Flux Ratio** or **Departure from Nucleate Boiling Ratio** — the predicted Critical Heat Flux divided by the actual operating heat flux. Regulators require this ratio to stay above a specified minimum at all times, under all anticipated transients.

**The consequence of this regulatory structure is the entire motivation for our project:**

- If you **under-predict** the Critical Heat Flux, you are forced to run the reactor at lower power than it could safely sustain. That is a permanent, expensive loss of electricity generation over the plant's 60-year lifetime.
- If you **over-predict** it, you believe you have a safety margin that does not exist. That is an accident.

So the accuracy of Critical Heat Flux prediction has direct consequences in both money and safety. A prediction method that is 20% conservative costs a utility hundreds of millions of dollars. A prediction method that is 20% optimistic can destroy a core.

## 1.3 Why this is scientifically hard

If the problem were easy it would have been solved in 1960. It was not. Here is why.

**Reason 1 — There is no closed-form theory.** Critical Heat Flux is not like the ideal gas law, where you can derive the answer from first principles. It is the outcome of a chaotic, multi-scale, multi-phase process involving bubble nucleation at micrometre scales, bubble coalescence, turbulent liquid transport, droplet entrainment, and liquid-film dynamics — all interacting simultaneously. Nobody can write down an exact equation for it.

**Reason 2 — The mechanism itself changes.** At low vapour content near a subcooled wall, the failure mechanism is a thin liquid layer under a vapour blanket drying out (**Departure from Nucleate Boiling**). At high vapour content in annular flow, the failure mechanism is a thin liquid film on the tube wall being depleted by evaporation and droplet entrainment faster than droplets deposit back onto it (**dryout**). These are *physically different events*. A single equation is being asked to describe two different phenomena.

**Reason 3 — Many variables matter, and they interact.** Pressure, mass flow rate, inlet temperature, tube diameter, heated length, orientation, surface roughness, surface material, wall thickness, whether heating is uniform along the tube or not, whether flow is upward or downward, and the working fluid itself. These do not act independently.

**Reason 4 — Experiments are expensive and destructive.** Measuring Critical Heat Flux means deliberately driving a test section to burnout. Each measurement can destroy the test section. Data accumulates slowly, in small campaigns, at different laboratories, over decades. *(This fact becomes extremely important later — see Part VI.)*

## 1.4 The classical engineering solution, and what was wrong with it

Because no theory exists, the field built two practical tools.

**Tool 1 — Empirical correlations.** An engineer runs experiments, plots the results, and fits an algebraic formula. Examples include the Biasi correlation (1967), the Bowring correlation (1972), and the Katto–Ohno correlation (1984). These are explained in full detail in Part VIII. They are compact, fast, and physically interpretable, but each is valid only inside the range of conditions it was fitted to.

**Tool 2 — Look-up tables.** Instead of fitting a formula, tabulate the answer. The **Groeneveld 2006 Look-Up Table** is the international standard: a large numerical table listing Critical Heat Flux at every combination of pressure, mass flux, and vapour quality, for a reference 8-millimetre vertical tube carrying water. It is explained fully in Section 10.9.

**The gap that motivated us.** Both tools are accurate inside their fitted range and untrustworthy outside it. Over the last decade researchers began applying machine learning to the same databases, reporting excellent accuracy. The natural question for a final-year project was:

> **Can modern machine learning predict Critical Heat Flux better than sixty years of empirical correlations, and can it do so in a way that is trustworthy outside the conditions it was trained on?**

That was our Stage 0 question.

---

# PART II — STAGE 1: OUR INITIAL HYPOTHESIS

## 2.1 What we believed at the start

This section deliberately records what we thought **before** we knew better. It does not apply hindsight.

Our initial reasoning was:

1. Large, curated experimental databases of Critical Heat Flux exist publicly.
2. Critical Heat Flux depends on a manageable number of measured variables.
3. Modern machine-learning models — particularly tree ensembles and neural networks — are excellent at learning complicated non-linear relationships from tabular data.
4. Therefore: assemble the databases, feed the standard variables in, train strong models, and we should beat the classical correlations.

We expected to produce a paper structured as: *"we collected N thousand data points, trained several models, and achieved R² = 0.99, which exceeds the Groeneveld look-up table's accuracy."*

## 2.2 What we planned concretely

- **Input variables** (the standard set used throughout the published literature): pressure, mass flux, vapour quality at the burnout point, tube inner diameter, heated length, and inlet subcooling.
- **Output variable**: Critical Heat Flux, in kilowatts per square metre.
- **Models**: tree ensembles first (Random Forest, Extreme Gradient Boosting), then Gaussian Process Regression and neural networks.
- **Validation**: the standard approach — split the data randomly, train on 80%, test on 20%.
- **Metrics**: coefficient of determination, mean absolute percentage error, root-mean-square error.

## 2.3 What we did not yet understand

Recorded honestly, these were our blind spots at Stage 1:

1. We did not know that one of our "input" variables is **algebraically derivable from the output**. [See Part IV.]
2. We did not know that some datasets ship **the raw measurement from which the answer was computed** as an ordinary column. [See Part V.]
3. We did not know that these databases are **compilations of many small laboratory campaigns**, which makes a random split meaningless. [See Part VI.]
4. We did not appreciate that a high coefficient of determination on a random split is not evidence of physical understanding.

Every one of these four blind spots was discovered by measurement during the project, and each one changed the design. That chain of discoveries *is* the project.

---

# PART III — STAGE 2: THE FIRST IMPLEMENTATION, AND THE 0.99

## 3.1 What we built first

**Objective.** Establish a baseline: how well do standard machine-learning models predict Critical Heat Flux from the standard published feature set?

**Dataset.** We began with the largest available: the **Nuclear Regulatory Commission / Groeneveld flow-boiling database**, 24,443 experimental water flow-boiling points in vertical tubes. (Full description in Section 10.1.)

**Input features (the "local" formulation, so called because vapour quality is the *local* value at the burnout location):**

| Symbol | Full name | Unit | Meaning in plain words |
|---|---|---|---|
| P | Pressure | kPa | How hard the fluid is being squeezed. Sets the boiling temperature and all fluid properties. |
| G | Mass flux | kg/(m²·s) | How much fluid mass passes through one square metre of pipe cross-section every second. "How fast the coolant is moving," in mass terms. |
| x | Vapour quality at burnout | dimensionless | The fraction of the flowing mass that is vapour at the point where burnout occurred. x = 0 means all liquid, x = 1 means all vapour. Can be negative, which is a bookkeeping convention meaning the liquid is still below its boiling point (subcooled). |
| D | Tube inner diameter | mm | Width of the pipe. |
| L | Heated length | mm | How long the heated section of pipe is. |
| ΔH_in,sub | Inlet subcooling enthalpy | kJ/kg | How much energy per kilogram the incoming liquid still needs to absorb before it starts boiling. "How cold the water is when it enters, expressed as energy." |

**Output:** q″_CHF, the Critical Heat Flux, in kW/m².

**Preprocessing.** Remove rows with non-positive or non-finite Critical Heat Flux. Train on the logarithm of Critical Heat Flux rather than the raw value, because Critical Heat Flux spans roughly three orders of magnitude across the database (from about 50 to about 16,339 kW/m² in this dataset); training on the logarithm makes the model care about *relative* error rather than absolute error, which is the engineering-relevant quantity.

**Splitting.** Random 80% train / 20% test.

**Models.** Thirteen families, described fully in Part XI.

**Metrics.** Defined fully in the Glossary (Part XXVI); briefly:
- **Coefficient of determination (R²)** — 1.0 is perfect; 0.0 means "no better than always guessing the average"; negative means "worse than always guessing the average."
- **Mean absolute percentage error (MAPE)** — the average size of the error as a percentage of the true value.
- **Root-mean-square error (RMSE)** — a typical error size in the original units, kW/m².
- **Fraction within ±10%** — the fraction of predictions within ten percent of the measured value. This is the metric a thermal-hydraulics engineer actually cares about.

## 3.2 The result

**[OUR EXPERIMENT]** Source: `results/prof_tasks/grid_results.csv`.

Using the local formulation with a random 80/20 split, the best model on the Nuclear Regulatory Commission dataset was **Extra Trees, coefficient of determination = 0.987**, with 78.1% of predictions within ±10%.

Across all ten datasets in the local formulation, the mean of the best coefficient of determination per dataset was **0.973**, with 86.3% within ±10%.

These are, by the standards of the published literature, excellent numbers.

## 3.3 The moment of doubt

Before writing this up, we asked a question that turned out to define the project:

> **What is the model actually using to make these predictions?**

We examined feature importance. Vapour quality dominated. That is not surprising in itself — quality is physically central to dryout.

But then we noticed something in the structure of the variables themselves. And that observation is the subject of the next part.

---

# PART IV — THE FIRST MAJOR DISCOVERY: THE CIRCULARITY OF VAPOUR QUALITY

This is the single most important finding in the project. It is explained here in complete detail because it is what everything else was rebuilt around.

## 4.1 The physical set-up

Picture a straight, uniformly heated tube. Liquid enters at the bottom, colder than its boiling point. Heat is applied evenly along the whole heated length. As the liquid travels up, it absorbs heat. First it warms to its boiling point. Then it starts boiling, and the fraction of vapour steadily increases. At some point, burnout occurs.

## 4.2 The energy balance

Now apply **conservation of energy**, one of the most basic laws in physics: energy cannot be created or destroyed. In steady state, *all the heat put into the tube wall must appear as energy gained by the fluid.*

### Step 1 — How much heat goes in?

Heat flux is heat per unit area per unit time. The heated area of a circular tube is its inner circumference multiplied by its heated length:

```
Heated area  =  pi * D * L
```

So the total thermal power going into the fluid is:

```
Power in  =  q_CHF * pi * D * L
```

- `q_CHF` = heat flux, in watts per square metre (W/m²)
- `D` = tube inner diameter, in metres (m)
- `L` = heated length, in metres (m)
- `pi` = 3.14159...

### Step 2 — How much energy does the fluid carry away?

Mass flow rate is mass flux multiplied by the cross-sectional area of the tube. The cross-sectional area of a circle is:

```
Cross-sectional area  =  pi * D^2 / 4
```

So:

```
Mass flow rate  =  G * pi * D^2 / 4
```

- `G` = mass flux, in kilograms per square metre per second (kg/(m²·s))

Now, how much energy has each kilogram of fluid gained between the inlet and the burnout point? Two contributions:

**Contribution A — warming the liquid up to its boiling point.** The incoming liquid was colder than saturation. Bringing it up to saturation temperature costs `ΔH_in,sub` joules per kilogram. This quantity is called the **inlet subcooling enthalpy**.

**Contribution B — boiling a fraction of it.** Converting one kilogram of saturated liquid entirely into vapour costs `h_fg` joules per kilogram, the **latent heat of vaporisation**. If only a fraction `x` of the mass has been vaporised, the cost is `x * h_fg`.

Total energy gained per kilogram:

```
Energy gain per kg  =  x * h_fg  +  ΔH_in,sub
```

So the total rate of energy absorption is:

```
Power absorbed  =  (G * pi * D^2 / 4) * (x * h_fg + ΔH_in,sub)
```

### Step 3 — Set them equal

```
q_CHF * pi * D * L  =  (G * pi * D^2 / 4) * (x * h_fg + ΔH_in,sub)
```

Cancel `pi` from both sides. Cancel one factor of `D`. Divide both sides by `L`:

```
q_CHF  =  G * D * (x * h_fg + ΔH_in,sub) / (4 * L)
```

**This is the equation. It is exact.**

### Symbol-by-symbol explanation

| Symbol | Name | Unit | Plain meaning |
|---|---|---|---|
| q_CHF | Critical Heat Flux | W/m² | What we are trying to predict. |
| G | Mass flux | kg/(m²·s) | How much coolant mass flows through a square metre of pipe cross-section per second. |
| D | Tube inner diameter | m | Pipe width. |
| L | Heated length | m | Length of the heated section. |
| x | Vapour quality at burnout | — | Fraction of the mass that is vapour at the burnout point. |
| h_fg | Latent heat of vaporisation | J/kg | Energy needed to boil one kilogram of saturated liquid. Determined by pressure. |
| ΔH_in,sub | Inlet subcooling enthalpy | J/kg | Energy needed to bring the entering liquid up to its boiling point. |
| 4 | — | — | Arises purely from the geometry: the ratio of (cross-sectional area) to (perimeter) for a circle is D/4. |

### What the equation says in words

> *The heat you can push through the wall equals the energy the fluid is carrying away, and the energy it is carrying away is fully determined by how much fluid is flowing and how much of it has boiled.*

### What happens if a term changes

- **Increase G (faster flow):** more mass passes per second, so more energy is carried away, so Critical Heat Flux rises. Physically correct — faster coolant flow permits more heating.
- **Increase L (longer heated tube):** the same power is spread over more area, so the flux at burnout is lower. Physically correct — longer tubes burn out at lower flux.
- **Increase ΔH_in,sub (colder inlet):** the fluid has more capacity to absorb heat before boiling, so Critical Heat Flux rises. Physically correct.
- **Increase x (more vapour at burnout):** more energy has been absorbed, so more heat was applied. Physically correct, but see the warning below.

### Assumptions behind the equation

1. Steady state — nothing changing in time.
2. Uniform heating along the length. *(For non-uniform heating the derivation must be done as an integral along the tube; this is exactly why one of our datasets behaves differently — see Section 10.4.)*
3. No heat loss to the surroundings; all wall power enters the fluid.
4. Circular cross-section (the factor of 4).
5. Thermodynamic equilibrium — the vapour and liquid are at the same temperature.

## 4.3 Where this equation comes from

**[LITERATURE]** This is **not our invention**. It is a standard steady-state energy balance that appears in essentially every thermal-hydraulics textbook. In the specific form above it is **Equation 8 of Hall & Mudawar (2000b)**, *"Critical heat flux (CHF) for water flow in tubes — II. Subcooled CHF correlations,"* International Journal of Heat and Mass Transfer, Volume 43, pages 2605–2640.

It is also already written in this project's own physics foundation document, at Section 1.5, where it is tagged **[V]** meaning "verified against source."

> **This must be stated clearly to any reviewer.** Presenting this relation as a discovery would be a serious error. A reviewer for *Nuclear Engineering and Design* would recognise it immediately. Our contribution is **not** the equation. Our contribution is the demonstration of what it implies about how the field measures accuracy.

## 4.4 The experiment that proved the problem

### Experiment 1 — Energy-balance reconstruction

**Objective.** Determine how much of the reported machine-learning accuracy can be reproduced by this algebraic identity alone, with no fitting whatsoever.

**Hypothesis.** If the relation is exact, then applying it directly should reproduce the measured Critical Heat Flux very closely — and if it does, then a machine-learning model given the same inputs is mostly re-deriving arithmetic, not learning physics.

**Setup.** Take all 24,443 rows of the Nuclear Regulatory Commission database. For each row, take the reported G, D, L, x, and inlet subcooling. Compute h_fg from the reported pressure using CoolProp (a standard open-source thermophysical property library implementing the internationally accepted equations of state). Apply the formula. Compare to the reported Critical Heat Flux.

**Number of fitted parameters: zero.** Nothing is trained. Nothing is tuned.

**Results.** **[OUR EXPERIMENT]** Source: `results/prof_tasks/exhibitA_energy_balance.csv`

| Metric | Value |
|---|---|
| Rows | 24,443 |
| Coefficient of determination (R²) | **0.99977** |
| Median absolute relative error | **0.372%** |
| Fraction within ±10% | **100.0%** |
| Fraction within ±1% | **77.3%** |
| Median ratio (predicted ÷ actual) | 1.00043 |

**Interpretation.** A formula with **zero fitted parameters** reproduces the "difficult" target with a coefficient of determination of 0.99977 and gets **every single one of 24,443 predictions within ten percent**.

Compare this to the best machine-learning result on the same data with the same inputs: Extra Trees, coefficient of determination 0.987, with 78.1% within ±10%.

**The zero-parameter formula beats the trained machine-learning model.**

**What this means.** The reported 0.987 was not evidence that the model had learned boiling physics. The model was, imperfectly, re-discovering an algebraic identity that was already sitting inside its input variables.

## 4.5 Why this is "circular" and not "cheating"

This distinction matters and is worth getting right, because the two words mean different things and only one of them is accurate.

**It is not cheating.** The relation is correct physics. It is published. Using it is legitimate. If you know the vapour quality at burnout, you genuinely can compute the Critical Heat Flux, and you should.

**It is circular.** Here is the problem, and it is a problem of *timing*, not of correctness:

> **The vapour quality at the burnout point does not exist until burnout has already happened.**

Think about what an engineer actually wants. They are designing a reactor channel. They know the pressure, the coolant flow rate, the tube dimensions, and the inlet temperature. They want to know: *at what heat flux will this channel burn out?*

They do **not** know the vapour quality at the burnout point. That is a property of the burnout event. You only learn it after you have already run the experiment and destroyed the test section.

So a model that requires vapour quality as an input can explain the past perfectly and **predict nothing**. It answers "given that burnout happened here, what was the heat flux?" — a question nobody needs answered — rather than "will burnout happen, and where?"

**An analogy for explaining this verbally:**

> Imagine predicting the final score of a football match. You build a model and report 99.98% accuracy. Then someone asks what your inputs were, and one of them is "the number of goals each team scored." Your model is not wrong — the final score genuinely *is* determined by the goals scored. But it is useless, because you only know the goals after the match is over. The thing you were supposed to predict was smuggled into the inputs.

That is the exact structure of the problem here.

## 4.6 Does the circularity apply to every dataset? We checked.

**[OUR EXPERIMENT]** We did not assume this generalises. We tested it per dataset, and the answer is genuinely mixed:

| Dataset | Does the identity hold? | Why |
|---|---|---|
| Nuclear Regulatory Commission | **Yes**, exactly (R² = 0.99977) | Uniformly heated tubes, all required variables reported. |
| KAERI uniform heating | **Yes** | Uniformly heated tubes. |
| KAERI **non-uniform** heating | **No** | Heating is *not* uniform along the tube, so assumption 2 of the derivation is violated. The simple algebraic form does not apply; an integral along the axial power profile would be required. |
| Groeneveld 2006 Look-Up Table | **Not applicable** | The table contains no diameter, no heated length, and no inlet subcooling. Three of the equation's terms do not exist. See Section 4.7. |

This is an important methodological point in its own right: **the circularity is real but not universal**, and claiming otherwise would be over-generalising.

## 4.7 Why the identity does not apply to the 2006 Look-Up Table — and why quality *must* be an input there

This question was raised directly and deserves a complete answer, because on the surface it looks like a contradiction: *if outlet quality should not be used as an independent input, why does the international standard table use it as one of only three inputs?*

### What the Look-Up Table actually contains

**[OUR EXPERIMENT]** We inspected the file directly (`share/05_lookup_table_reference/groeneveld_2006_lut_long_format.csv`). It has exactly **four columns**:

```
P, G, X, CHF
```

- 24 distinct pressure values, from 100 to 21,000 kPa
- 21 distinct mass flux values, from 0 to 8,000 kg/(m²·s)
- 23 distinct quality values, from −0.5 to 1.0
- 24 × 21 × 23 = **11,592 rows**

**There is no diameter column. There is no heated length column. There is no inlet subcooling column.**

### Answer to part one: why the energy balance cannot be applied

Look again at the equation:

```
q_CHF  =  G * D * (x * h_fg + ΔH_in,sub) / (4 * L)
```

To evaluate it you need `D`, `L`, and `ΔH_in,sub`. The table contains **none of the three**.

- `D` is fixed by convention at 8 mm for the whole table (it is a *normalised* table, and other diameters are handled by a separate correction factor), so it carries no information and cannot vary.
- `L` simply does not exist as a concept in the table. A table entry is not an experiment on a particular tube; it is a smoothed value representing many experiments.
- `ΔH_in,sub` does not exist either. The table is indexed by *local* conditions at the burnout plane, deliberately abstracting away what happened upstream.

So the identity is not "violated" by the table — it is **inapplicable**. Two of its inputs are absent, and the third is a constant.

This also corrects an earlier incorrect explanation given during the project. At one point the poor performance on the table was attributed to Critical Heat Flux "rising versus falling with quality." **That explanation was wrong.** The identity is an exact algebraic constraint, not a trend, and trends have nothing to do with it. The real reason is simply that the table has no diameter and no length.

### Answer to part two: why quality *must* be an input to the table

This one has a clean, decisive numerical answer.

The Groeneveld table is, **by definition**, the function:

```
CHF  =  f(P, G, x)
```

That is what the table *is*. Its entire structure is a three-dimensional grid indexed on those three variables. Remove quality and you are left with pressure and mass flux — 24 × 21 = **504 distinct cells**, and each of those cells contains **23 different Critical Heat Flux values**, one for each quality.

That is a **one-to-many mapping**. For a single (pressure, mass flux) pair there is not one right answer; there are twenty-three, and they differ enormously.

**[OUR EXPERIMENT]** We quantified exactly how badly this bounds any possible model. Consider a hypothetical *perfect oracle* that has memorised the table and, given only pressure and mass flux, returns the exact average of all Critical Heat Flux values in that cell. That is the best any function of (P, G) alone can possibly do.

| Predictor | Theoretical best possible R² |
|---|---|
| Perfect oracle using **(P, G) only** | **0.1538** |
| Perfect oracle using **(P, G, x)** | **1.0000** |

Additional measurement: within a single (pressure, mass flux) cell, Critical Heat Flux spans a **median factor of about 200×** between its largest and smallest value.

**Interpretation.** Our measured result of R² = −0.003 for a model on the table without quality is **not a model failure**. The ceiling is 0.15. Any model that cannot beat the cell mean will land near zero, and a model with no usable signal lands slightly below it. There is no information in (P, G) alone from which Critical Heat Flux can be recovered.

**Therefore the two statements are not in conflict:**

1. *For predicting a future experiment*, quality is unusable, because you do not know it in advance.
2. *For the Look-Up Table*, quality is mandatory, because the table is defined as a function of it and removing it destroys the mapping entirely.

The table is not a predictive tool in the sense we need. It is a **local closure relation** — it answers "given the local conditions at this plane, what is the critical flux here?" A reactor thermal-hydraulics code uses it *inside an iterative loop*: the code marches up the channel computing quality from the energy balance, queries the table at each step, and finds where the critical condition is first met. The quality is supplied by the code's own energy balance, not by the user.

**[OUR EXPERIMENT]** We implemented precisely this scheme as one of our approaches, named `ITER_closure_GB` — a local-form model solved jointly with the energy balance from inlet conditions only, which is the legitimate, non-circular way to use a local-form model.

## 4.8 The cost of removing quality

Once we removed vapour quality, results fell exactly as they should.

**[OUR EXPERIMENT]** Source: `results/prof_tasks/ANSWERS_Q1_Q2.txt`

We defined three **formulations** — three different sets of input variables:

| Formulation | What it contains | Honest? |
|---|---|---|
| **local** | Everything including vapour quality at burnout | No — circular |
| **inlet** | Inlet conditions only, no local quality | Yes |
| **reduced** | Local features with quality removed | Yes |

Mean of the best coefficient of determination per dataset:

| Split | local | inlet | reduced |
|---|---|---|---|
| 80/20 random | **0.973** | 0.961 | **0.743** |
| 70/30 random | 0.973 | 0.961 | 0.741 |
| Leave-one-source-out | −1.144 | −2.732 | −3.115 |

Fraction within ±10%:

| Split | local | inlet | reduced |
|---|---|---|---|
| 80/20 random | 86.3% | 86.5% | **50.8%** |
| 70/30 random | 85.3% | 83.1% | 48.9% |

**The drop from 0.973 to 0.743 is the size of the illusion.** Nearly a quarter of the apparent explanatory power of the standard feature set comes from a variable that cannot be known in advance.

**Decision taken.** All headline results from this point forward use honest features only. The local formulation is retained in the benchmark **solely as a diagnostic**, clearly labelled, never as a headline.

---

# PART V — THE SECOND DISCOVERY: COLUMNS THAT CONTAIN THE ANSWER

## 5.1 Why we went looking

The circularity discovery taught us a general lesson: **do not trust a column just because it has a physical-sounding name.** We therefore audited every column of every dataset and asked, for each one, a blunt question:

> *Can this column, alone or combined with one or two others, be used to reconstruct the target?*

This is a different problem from circularity. Circularity is about a physically meaningful variable that is unknowable in advance. This is about **data leakage** — a column that is, in disguise, the answer itself.

## 5.2 How Critical Heat Flux is actually measured

To understand the leakage you need to understand how the number in the "CHF" column got there. It was not measured directly. There is no instrument that reads out heat flux.

What is actually measured in a burnout experiment is **electrical power**. The test section is a metal tube used as an electrical resistance heater. The experimenter increases the current until the wall temperature runs away, and records the electrical power at that instant. Then the heat flux is *computed*:

```
q_CHF  =  Power / Heated area  =  Power / (pi * D * L)
```

So `Power`, `D`, and `L` are the *raw measurements*; Critical Heat Flux is a *derived quantity*.

**Therefore: if a dataset reports both the power and the geometry, it has reported the answer twice.** A machine-learning model given those columns is not predicting anything. It is performing a division.

## 5.3 The audit results

**[OUR EXPERIMENT]** Source: `results/prof_tasks/leakage_evidence.csv`

| Dataset | Leaking columns | Reconstruction formula | R² | Within ±10% |
|---|---|---|---|---|
| KAERI uniform | Power, Perimeter, Length | Power / (Perimeter × Length) | **0.9999934** | **100%** |
| KAERI non-uniform | Power, Perimeter, Length | Power / (Perimeter × Length) | **0.9999928** | **100%** |
| Hardik helical (water) | Q_W | Q_W / (π × d × L) | **0.9793** | **98.1%** |
| Hardik helical (water) | CHF_ratio | CHF ÷ look-up-table CHF | — | it *is* the target, merely rescaled |

**Interpretation.** On the two KAERI datasets, three ordinary-looking columns reconstruct the target essentially exactly — a coefficient of determination of 0.9999934 and every single prediction within ten percent. This is not a model result. It is arithmetic.

The `CHF_ratio` column in the Hardik helical dataset is a subtler case and worth explaining: it is the measured Critical Heat Flux **divided by** the value the Groeneveld look-up table predicts for the same conditions. Researchers publish this because it is a useful diagnostic. But it is a direct function of the target. Feeding it to a model as an input is equivalent to handing the model the answer multiplied by a known number.

## 5.4 What we did about it

**Decision.** We implemented a **hard leakage whitelist** in the data-loading layer (`scripts/prof_tasks/datasets.py`). Every dataset object carries an explicit `banned` field naming columns that may never be used as model inputs, regardless of what any downstream code requests.

**[OUR EXPERIMENT]** The banned lists, as implemented:

| Dataset | Banned columns |
|---|---|
| KAERI uniform | Power, Perimeter, Area, MassFlow, InletEnthalpy |
| KAERI non-uniform | Power, Perimeter, Area, MassFlow, InletEnthalpy, CHFLocation, QualityPosition |
| Helical R-123 | Q_watt |
| Hardik helical water | Q_W, CHF_ratio |

**Why a whitelist rather than a manual exclusion.** Because we had already been burned. A manual exclusion is a thing you remember to do; a whitelist enforced at load time is a thing the code cannot forget. Every later experiment inherits the protection automatically.

**Why `MassFlow` and `InletEnthalpy` are banned even though they sound legitimate.** They are dimensional restatements of mass flux and inlet subcooling that, combined with the geometry columns present in those files, permit the energy balance to be closed. Banning them removes a second route to the same leak.

## 5.5 The general lesson

**[OUR INTERPRETATION]** These two discoveries — circularity and leakage — are different symptoms of one underlying issue:

> In a physical science, the variables in a dataset are related to each other by **conservation laws**. A machine-learning practitioner accustomed to, say, predicting customer churn does not expect the input columns to be algebraically linked to the target. In thermal-hydraulics they routinely are.

This means **a machine-learning audit of a physics dataset must include a conservation-law audit.** That is not standard practice, and we believe it should be.

---

# PART VI — THE THIRD DISCOVERY: THE SPLIT ITSELF WAS WRONG

## 6.1 The observation that started it

Even after removing quality and banning the leaking columns, something still looked too good. We asked where the remaining data came from.

**The Nuclear Regulatory Commission database is not one experiment.** It is a *compilation*. Its 24,443 rows were assembled from approximately **60 different source publications**, produced by different laboratories, in different countries, over several decades, using different test rigs, different instrumentation, different tube materials, and different experimental conventions.

## 6.2 Why that breaks a random split

A random 80/20 split assumes that every row is an **independent sample from one population**. That assumption is false here.

Rows from the same laboratory campaign share:
- the same test rig and its systematic calibration biases,
- the same batch of tubing, with its particular surface finish and wall thickness,
- the same instrumentation and its particular offsets,
- the same experimenter's conventions for defining the burnout point,
- and often a densely sampled grid of very similar conditions.

So when you split randomly, you place **siblings on both sides of the wall**. The test set contains rows that are near-duplicates of training rows, from the same rig, on the same day.

**[OUR INTERPRETATION]** What the model learns in that situation is partly *"which laboratory is this?"* rather than *"what is the boiling physics?"* And laboratory identity is a perfectly good predictor — right up until you show the model data from a laboratory it has never seen, which is exactly what happens when the model is deployed.

## 6.3 Quantifying it: the confounding analysis

**[OUR EXPERIMENT]** Source: `docs/CONFOUNDING_FINDINGS.md`, generated by `scripts/run_confounding_analysis.py`.

We measured the **Intraclass Correlation Coefficient (ICC)**. In plain words: *what fraction of the total variation in the target is explained by study identity alone — that is, by nothing more than knowing which laboratory produced the row?*

| Data family | Rows | Studies | Rows per study | ICC (study identity alone) | Extra explained by adding real physical features |
|---|---|---|---|---|---|
| Advanced-fuel-cladding surface set | 55 | 7 | 7.9 | **0.815** | **+0.042** |
| Pin-fin set | 175 | 16 | 10.9 | 0.659 | +0.303 |
| Nuclear Regulatory Commission flow boiling | 24,443 | 60 | 407 | 0.345 | +0.556 |

**How to read this.** On the surface dataset, simply knowing *which laboratory ran the test* explains **81.5%** of the variation in Critical Heat Flux. Adding the actual physical measurements — surface roughness and contact angle, the very quantities the source papers were written about — buys only **4.2 percentage points** more.

**The pattern is monotonic in rows-per-study.** The fewer rows each laboratory contributes, the more of the signal is laboratory identity. This is a dose-response relationship, which is the strongest form of evidence available in an observational study.

## 6.4 Simpson reversals: pooling inverts the physics

**[OUR EXPERIMENT]** Same source. This finding is striking enough to state separately.

A **Simpson reversal** occurs when a relationship visible in pooled data points in the *opposite direction* to the relationship inside every individual subgroup.

| Family | Feature | Correlation in pooled data | Correlation within each study | Reversed? | Studies agreeing with the within-study sign |
|---|---|---|---|---|---|
| Surface set | **Surface roughness (Ra)** | −0.293 | **+0.439** | **YES** | **5 of 5** |
| Surface set | Contact angle | −0.774 | −0.332 | No, but 2.3× overstated | 5 of 6 |
| Pin-fin | **Subcooling** | −0.092 | **+0.537** | **YES** | **6 of 6** |
| Pin-fin | Roughness factor | +0.156 | +0.466 | No, but 3× understated | 13 of 13 |
| Pin-fin | Porosity | +0.456 | +0.671 | No | 7 of 7 |
| Nuclear Regulatory Commission | all 7 features | — | — | **none** | 13–56 of 19–58 |

**Why the roughness reversal matters physically.** The pooled data says *rougher surfaces have lower Critical Heat Flux*. That contradicts the capillary-wicking mechanism which every one of those source papers invokes as its explanation. Within each study individually, **all five testable studies show the physically expected positive effect**.

So a researcher who pools these datasets and reports "roughness reduces Critical Heat Flux" would be publishing a conclusion that is the exact opposite of what their own constituent experiments show.

**Crucially: no reversals occur in the large Nuclear Regulatory Commission set.** The pathology is specific to sparse compilations with few rows per study. This is a bounded, defensible claim rather than a sweeping one.

## 6.5 The four splits we adopted

**Decision.** From this point on, every result is reported under four different splitting protocols, not one. Each answers a different question.

| # | Split | What it does | The question it answers |
|---|---|---|---|
| 1 | **80/20 random** | 80% train, 20% test, chosen at random | "How well does it interpolate among conditions like the ones it has seen?" — this is what the literature reports |
| 2 | **70/30 random** | Same, with a larger test set | "Is result 1 an artefact of the particular split ratio?" |
| 3 | **Grouped hold-out** | Hold out an entire laboratory / author / coil / pressure range | "What happens with a source it has never seen?" |
| 4 | **Leave-One-Dataset-Out (LODO)** | Train on seven datasets, test on the entire eighth | "What happens with a completely new experimental campaign?" |

Splits 1 and 2 were explicitly requested by the project supervisor. Splits 3 and 4 are the honest ones.

## 6.6 Duplicate-aware splitting

**[OUR EXPERIMENT]** A further refinement. We found near-duplicate rows within datasets — the same physical condition recorded more than once. If a duplicate pair is split across train and test, the test score is inflated because the model has literally memorised that exact row.

We implemented a **duplicate key**: a signature computed from the five dimensionless features plus the target value. Rows sharing a signature are forced into the same side of the split, using scikit-learn's `GroupShuffleSplit`.

**Measured effect.** On the Zhao dataset, the naive 80/20 coefficient of determination was 0.847; with duplicate-aware grouping it was 0.864 — a gap of −0.017.

**[OUR INTERPRETATION]** Note that the gap here is *negative*, meaning duplicate-aware grouping scored slightly **higher**, not lower. This was an honest surprise and worth recording: an earlier hypothesis had been that a fabricated-diameter bug (Section 17.4) was generating the duplicates and that fixing the split would lower the score. Only 48 of 770 duplicates turned out to be artefacts of that bug. **The hypothesis was wrong.** The duplicates are mostly genuine repeated measurements.

## 6.7 What the honest splits revealed

**[OUR EXPERIMENT]** Source: `results/prof_tasks/ANSWERS_Q1_Q2.txt`

Mean of the best coefficient of determination per dataset, honest (reduced) features:

| Split | Mean best R² |
|---|---|
| 80/20 random | 0.743 |
| 70/30 random | 0.741 |
| **Leave-one-source-out** | **−3.115** |

**A model that scores 0.74 on a random split scores −3.1 on a laboratory it has not seen.** Negative means worse than ignoring all the inputs and predicting the average of the training set every time.

**This is the result that forced the entire architecture.** No amount of model tuning fixes a −3.1. The problem is structural: an unconstrained model, asked about conditions outside its training envelope, has nothing to fall back on. It extrapolates, and tree ensembles extrapolate as flat step functions while neural networks extrapolate as arbitrary polynomials. Neither has any notion of what is physically possible.

**The conclusion we drew, and it is the thesis of the project:**

> The model needs something underneath it that cannot be wrong by a factor of a thousand. It needs **physics as a floor**.

---

# PART VII — LITERATURE INVESTIGATION: HOW READING CHANGED THE PROJECT

## 7.1 Why we read what we read

The literature search was not a bibliography exercise. Each stage of reading was triggered by a specific question raised by a specific failure. This section preserves that chain.

```
We needed a baseline that could not be wrong by 1000x
        |
        v
  Read classical correlations  (Katto-Ohno, Biasi, Bowring)
        |
   Question: which is best, and over what range?
        |
        v
  Read the Groeneveld 2006 Look-Up Table documentation
        |
   Question: the table needs local quality -- how do reactor codes use it?
        |
        v
  Read Hall & Mudawar (2000a, 2000b)
        |
   Discovery: the energy-balance identity -- the circularity problem
        |
        v
  Question: does the ML literature use local quality as an input?
        |
        v
  Read recent ML-for-CHF papers
        |
   Observation: most report random-split R^2 ~ 0.98 with local quality
        |
        v
  Question: has anyone anchored ML to physics instead of replacing it?
        |
        v
  Read physics-informed ML and grey-box literature
        |
   Found: residual learning on a correlation prior; transfer learning on the LUT
        |
        v
  Question: does residual learning survive out-of-domain?
        |
        v
  Read Wu, Gui & Wu -- reports residual models diverge out-of-domain
        |
        v
  DESIGN DECISION: residual learning, but BOUNDED. Part XIV.
```

## 7.2 Papers we physically hold and have read

**[LITERATURE]** These are in `docs/references/` in the repository. These are the ones we can defend citing.

| Reference | Relevance to us |
|---|---|
| `CHF_Look_Up_Table.pdf` — Groeneveld 2006 Look-Up Table documentation | Defines our reference standard and one of our three physics backbones. |
| `An overview of measurements, data compilations and prediction methods.pdf` | Survey of how the field compiles data — directly informed our confounding analysis. |
| `Hardik_Kumar_Prabhu_2017_CHF_straight_tubes_R123_IJHMT.pdf` | Source paper for dataset D7. |
| `Hardik_Prabhu_2017_CHF_helical_coils_water_lowP_ATE.pdf` | Source paper for dataset D6, including the Table 4 coil geometry we needed. |
| `Hardik_Prabhu_2018_CHF_helical_coils_R123_IJTS.pdf` | Source paper for dataset D5. |
| `Diameter effect on CHF.pdf` | Background for the 8 mm diameter normalisation in the look-up table. |
| `Critical Heat Flux of Flowing Water.pdf` | Classical flow-boiling background. |
| `Benchmark on AI and ML for Scientific Computing in Nuclear Engineering — Phase 1 CHF Exercise Specifications.pdf` | An OECD/NEA benchmark specification; establishes that the community itself considers this an open problem. |
| `Yang_etal_2026_transfer_learning_CHF_2006LUT_ATE.pdf` | The transfer-learning-on-the-look-up-table approach; the closest published relative of our design. |
| `A Three-Stage Bayesian Transfer Learning Framework to Improve.pdf` | Transfer learning for related thermal-hydraulic problems. |
| `Critical heat flux prediction through machine learning model for narrow.pdf` | Machine learning applied to narrow channels. |
| `Prediction of "critical heat flux" for supercritical water and CO2.pdf` | Supercritical regime — outside our scope but clarified the validity boundary. |
| `Kefer_Kohler_Kastner_1989_inclined_evaporator_tubes_CHF_IJMF.pdf` | Inclined-tube effects. |
| `ML19029B306.pdf`, `ML22264A009.pdf` | United States Nuclear Regulatory Commission documents relating to the database. |
| Plus approximately 20 further journal papers in `docs/references/` | General background on machine learning for boiling heat transfer. |

## 7.3 The inherited literature matrix — and an important warning

**[LITERATURE — UNVERIFIED]** The project contains an earlier literature synthesis at `docs/manuscript/literature_survey_reviewed.md`, which includes a sixteen-paper comparison table. That table carries its own disclaimer, which we reproduce here verbatim because it is essential:

> *"Dataset sizes, architectures and reported metrics for refs. [58], [61]–[75] are drawn from title/topic matching pending full-text confirmation and should be treated as provisional."*

**In plain words: most of the numbers in that table were inferred from paper titles and topics, not read from the papers.** They are not verified. They must not be quoted to a supervisor, put in a manuscript, or used as evidence of anything, until someone opens each paper and confirms them.

**This is flagged prominently because it directly affects a claim discussed in Section 18.4 — whether the published literature systematically over-reports accuracy. Our own literature table is not adequate evidence for that claim.**

The directions those papers represent, which *did* legitimately shape our thinking, are summarised below **without their unverified numbers**:

| Research direction | What it proposes | How it influenced us |
|---|---|---|
| Deep autoencoder feature extraction | Learn a compressed latent representation before regression | We considered and did not adopt: our feature count is small (5), so compression offers nothing. |
| Bayesian-optimised heterogeneous ensembles | Combine different architectures, weight by variance | Influenced our stacking experiments (Part XVI) — which failed. |
| Recurrent networks over axial position | Treat the tube's axial coordinate as a sequence to capture upstream boiling history | Conceptually relevant to the non-uniform-heating dataset. **Not implemented.** [OUR PROPOSED IDEA] |
| Look-up-table-guided grey-box networks | Use the look-up table as a deterministic backbone, learn a correction | **Closest published relative to our design.** Confirmed the backbone-plus-correction idea was sound. |
| Transfer learning pretrained on the look-up table | Pretrain on the table, fine-tune on experiments | An alternative way to inject physics — into the weights rather than into the architecture. **Not implemented.** [OUR PROPOSED IDEA] |
| Physics-informed loss penalties | Penalise violation of conservation residuals during training | We use a structural constraint (a bound) instead of a soft loss penalty. See Section 15.6 for why. |
| Comparative study of physics-informed paradigms | Compares feature engineering vs residual learning vs penalty methods | **The most directly useful finding we took from the literature:** residual learning reduces the burden on the network inside the training domain but *can worsen errors outside it* if the baseline correlation misbehaves asymptotically. This is precisely the failure mode our bound is designed to prevent. |

## 7.4 What we learned from the literature, stated plainly

1. **[LITERATURE]** Mechanistic first-principles models exist but depend on empirical closure constants that must be recalibrated per regime. They are not deployable as a universal predictor.
2. **[LITERATURE]** Computational fluid dynamics faces three obstacles for this problem: no universally accepted local criterion for the critical condition, high sensitivity to bubble-dynamics tuning parameters, and prohibitive cost. Not viable for our purpose.
3. **[LITERATURE]** Empirical correlations and look-up tables remain the operational workhorses. This validated our decision to use them as the backbone rather than discard them.
4. **[LITERATURE]** Residual learning on a physics prior is an established idea — we are **not** claiming it as novel.
5. **[LITERATURE]** The known weakness of residual learning is out-of-domain divergence. **Our bounded correction is a direct response to this documented weakness.**
6. **[OPEN QUESTION]** Whether the specific accuracy figures reported across this literature are inflated by circularity and random splitting. We have strong evidence **on our own corpus**. We have **not** audited the published papers. See Section 18.4.

---

# PART VIII — THE PHYSICS, FROM FIRST PRINCIPLES

This part explains every piece of physics the pipeline uses, assuming no prior knowledge. It is written so it can be reproduced on paper.

## 8.1 The physical system

**What we are studying.** A tube of circular cross-section, internal diameter `D`, with a heated section of length `L`. A fluid enters at one end at a known pressure and a known temperature. Heat is applied to the wall. We want to know the maximum wall heat flux before the boiling crisis occurs.

**What acts on the system.**
- A pressure that sets every thermophysical property of the fluid.
- A mass flow, which supplies fresh liquid and provides turbulent mixing.
- A wall heat flux, which is the driving load.
- An inlet enthalpy deficit (subcooling), which is stored cooling capacity.
- Surface tension, which resists the coalescence of bubbles into a continuous film.
- Gravity and buoyancy, which matter at low flow rates.

**Which quantities are known before the test.** Pressure, mass flux, diameter, heated length, inlet temperature, fluid identity.

**Which quantities are unknown before the test.** Critical Heat Flux itself; the vapour quality at burnout; the location of burnout along the tube.

**This known/unknown distinction is the entire basis of our honest feature set.**

## 8.2 The fluid properties, and where they come from

Every correlation needs the properties of the fluid *at saturation, at the operating pressure*. These are:

| Symbol | Name | Unit | Plain meaning |
|---|---|---|---|
| ρ_l (`rho_l`) | Saturated liquid density | kg/m³ | Mass of one cubic metre of the boiling liquid. |
| ρ_g (`rho_g`) | Saturated vapour density | kg/m³ | Mass of one cubic metre of the vapour above it. |
| σ (`sigma`) | Surface tension | N/m | How strongly the liquid surface resists being broken up. Falls to zero at the critical point. |
| h_fg | Latent heat of vaporisation | J/kg | Energy to boil one kilogram of saturated liquid. |
| h_f | Saturated liquid enthalpy | J/kg | Energy content of one kilogram of liquid at its boiling point. |
| T_sat | Saturation temperature | K | The boiling point at this pressure. |
| P_crit | Critical pressure | Pa | Above this pressure, liquid and vapour become indistinguishable and boiling does not exist. |

**How we obtain them.** Via **CoolProp**, an open-source thermophysical property library. For water it implements the **IAPWS** formulation — the International Association for the Properties of Water and Steam, the internationally agreed reference equation of state. These are genuine steam-table values, not hand-fitted approximations.

**Implementation note.** Property lookups are cached per (fluid, pressure) pair, because the same pressures repeat thousands of times across a dataset and each CoolProp call is comparatively slow.

**A bug worth recording here.** [OUR EXPERIMENT] For a period, an unrecognised fluid name silently fell back to **water's** properties. This meant the model would return a confident, completely wrong number computed from the wrong substance. It was caught because ammonia returned the exact value water returns (1960.0). The fix, at source in `physics_arms.cp_name`, is to **raise an error** rather than substitute. Failing loudly is always preferable to being confidently wrong.

## 8.3 The five dimensionless groups — each one explained fully

This was raised as a specific complaint: the short names were used without explanation. Here is the complete treatment.

### Why use dimensionless numbers at all?

This is the central idea that makes the whole pipeline possible, so it is worth getting right.

Suppose you run an experiment with water at 100 bar in a 10 mm tube, and another with refrigerant R-134a at 10 bar in a 5 mm tube. In raw units these have almost nothing in common — different densities, different surface tensions, different pressures, different sizes.

But if you form the right **ratios**, quantities whose units cancel out, the two experiments can land at *the same point*. A dimensionless number expresses a *balance between two competing physical effects*, and that balance can be identical in two physically different systems.

**The consequence for us is decisive:**

> The model never sees the text string "R-134a". It sees a set of ratios describing the balance of forces in that fluid. This is the **only** reason a fluid that appears in zero training rows is predictable at all.

This is the same principle that lets aircraft designers test a small model in a wind tunnel and apply the result to a full-size aircraft: match the Reynolds number, and the flow behaves the same way.

---

### Group 1 — `Katto_We_inv` : the inverse Weber number

**Formula as implemented:**

```
Katto_We_inv  =  sigma * rho_l / (G^2 * L)
```

**Symbol by symbol:**
- `sigma` = surface tension, N/m
- `rho_l` = saturated liquid density, kg/m³
- `G` = mass flux, kg/(m²·s)
- `L` = heated length, m

**Check that it is dimensionless:** numerator is (N/m)·(kg/m³) = (kg/s²)·(kg/m³) = kg²/(m³·s²). Denominator is (kg/(m²·s))²·m = (kg²/(m⁴·s²))·m = kg²/(m³·s²). They match — the units cancel. Good.

**What it means physically.** The ordinary **Weber number** is the ratio of *inertial force* (the momentum of the moving fluid, which tears liquid structures apart) to *surface-tension force* (which holds them together). This is its **inverse**, so:

> **`Katto_We_inv` is large when surface tension dominates and the liquid film is cohesive and hard to break up. It is small when inertia dominates and the liquid is being torn into droplets.**

**Why it matters for the boiling crisis.** In the high-quality dryout regime, the failure event is the tearing and depletion of a thin liquid film on the wall. Whether the film survives is precisely a contest between surface tension holding it together and vapour shear ripping it off. This group *is* that contest, expressed as a number.

**Why heated length `L` and not diameter `D`?** Because this is the form used in the Katto–Ohno correlation, where the relevant length scale for film development is the distance over which the film has been evaporating. *(A real bug was once found here: the Weber number had been computed on the wrong length scale. It is now on heated length, matching the source.)*

---

### Group 2 — `rho_ratio_gl` : the vapour-to-liquid density ratio

**Formula:**

```
rho_ratio_gl  =  rho_g / rho_l
```

**Symbol by symbol:**
- `rho_g` = saturated vapour density, kg/m³
- `rho_l` = saturated liquid density, kg/m³

**Dimensionless because** both are densities; the units cancel exactly.

**Typical values.** For water at atmospheric pressure, roughly 0.0006 — vapour is about 1,600 times less dense than liquid. As pressure rises toward the critical pressure, the two densities converge, and the ratio approaches **1.0**.

**What it means physically.** It is a direct measure of **how far the fluid is from its critical point** and of **how much the vapour expands** when liquid boils.

- Ratio very small (low pressure): boiling one kilogram of liquid produces an enormous volume of vapour, which violently displaces liquid from the wall. Vapour blanketing is easy.
- Ratio near 1 (near-critical pressure): liquid and vapour are almost the same substance. There is little expansion, phase separation becomes weak, and surface tension approaches zero.

**Why it matters.** This group is what allows a single model to span water at 200 bar and a refrigerant at 10 bar. Both may sit at a density ratio of, say, 0.1, and at that point their boiling behaviour is genuinely analogous. **This group is the main carrier of fluid-to-fluid transferability.**

---

### Group 3 — `LtoD_dim` : the length-to-diameter ratio

**Formula:**

```
LtoD_dim  =  L / D
```

**Symbol by symbol:**
- `L` = heated length, m
- `D` = tube inner diameter, m

**Dimensionless because** both are lengths.

**Typical values in our corpus:** roughly 25 to over 500.

**What it means physically.** It describes the **shape** of the channel — is it a short stub or a long thin pipe? Equivalently, it measures how much opportunity the fluid has had to accumulate vapour before it exits.

**Why it matters.** A long tube at fixed heat flux will have boiled far more of its fluid by the exit than a short one. Long tubes therefore burn out at *lower* heat flux. The length-to-diameter ratio captures this directly, and every classical correlation includes it in some form.

**An honest caveat.** [ASSUMPTION] Because the energy balance also contains `D` and `L`, this group carries a *weak* trace of the same information the energy balance uses. It is not circular — it uses no outlet quality — but it is not perfectly independent either. A planned "clean feature re-run" (Section 21.5) removes it to check sensitivity. **That re-run has not been performed.**

---

### Group 4 — `dh_sub_star` : the dimensionless inlet subcooling

**Formula:**

```
dh_sub_star  =  dh_sub / h_fg
```

**Symbol by symbol:**
- `dh_sub` = inlet subcooling enthalpy, J/kg — the energy per kilogram the entering liquid must absorb before it begins to boil
- `h_fg` = latent heat of vaporisation, J/kg — the energy per kilogram to boil it once it gets there

**Dimensionless because** both are energies per unit mass.

**What it means physically.** The ratio of **"how much cooling capacity was stored in the cold inlet liquid"** to **"how much energy boiling actually costs."**

- Value 0: the liquid enters exactly at its boiling point. No stored capacity.
- Value 0.5: the liquid must absorb half as much energy to reach boiling as it would take to boil it completely. Substantial stored capacity.

**Why it matters.** Colder inlet means more energy can be absorbed before significant vapour appears, which pushes the burnout point further along the tube and raises the Critical Heat Flux. This effect is large and every correlation includes it.

**How `dh_sub` is obtained.** Three cases, handled explicitly in the code:
1. If the dataset reports inlet subcooling enthalpy directly, use it.
2. If the dataset reports inlet **temperature**, compute the enthalpy at that temperature and pressure with CoolProp and subtract from the saturated liquid enthalpy: `dh_sub = h_f − h_in`.
3. If neither is reported, set `dh_sub = 0` — the **saturated-inlet limit** — and record a flag `dh_sub_known = False` so downstream code knows this is an assumption rather than a measurement.

The value is clipped at zero, because a negative subcooling would mean the fluid entered already boiling, which the correlations do not handle.

---

### Group 5 — `P_red` : the reduced pressure

**Formula:**

```
P_red  =  P / P_crit
```

**Symbol by symbol:**
- `P` = operating pressure, Pa
- `P_crit` = the fluid's critical pressure, Pa — 22.064 MPa for water; approximately 3.66 MPa for R-123; approximately 4.06 MPa for R-134a

**Dimensionless because** both are pressures.

**What it means physically.** **How close is this fluid to its critical point, as a fraction?**

This is the single most important idea for comparing different fluids. Water at 11 MPa has a reduced pressure of 0.5. R-134a at 2.03 MPa also has a reduced pressure of 0.5. **At that point the two fluids are in thermodynamically corresponding states** — their density ratios, surface tensions, and latent heats are all in similar relative positions relative to their own critical points.

**Why it matters.** This group, together with the density ratio, is what makes a model trained mostly on water applicable to a refrigerant at all. It is the numerical expression of the **principle of corresponding states**.

**Why reduced pressure rather than raw pressure?** Because raw pressure is meaningless across fluids. 4 MPa is modest for water (reduced pressure 0.18) but supercritical for R-134a (above its critical pressure entirely, where Critical Heat Flux does not exist). A model given raw pressure would learn a water-specific rule and apply it catastrophically to a refrigerant.

---

### The five together

```
Row of raw data            Dimensionless representation
-----------------          ---------------------------
P       kPa       ---\
G       kg/m2s    ----\    Katto_We_inv   (inertia vs surface tension)
D       mm        -----+-> rho_ratio_gl   (distance from critical point)
L       mm        ----/    LtoD_dim       (channel shape)
T_in    C         ---/     dh_sub_star    (stored cooling capacity)
fluid   name      --/      P_red          (corresponding state)
```

**Note carefully: the fluid's *name* goes in, but only its *properties* come out.** That is deliberate, and it is what makes transfer to an unseen fluid possible.

**A sixth quantity used only as a diagnostic target:**

```
Bo_actual  =  CHF * 1000 / (G * h_fg)
```

This is the **boiling number** — the ratio of the applied heat flux to the maximum flux that the flowing mass could absorb if every kilogram were fully vaporised. It is the natural dimensionless form of the target itself, and one of our physics-informed approaches predicts this rather than raw Critical Heat Flux.

---

# PART IX — THE CLASSICAL CORRELATIONS AND THE SCIENTISTS BEHIND THEM

This part explains the three physics correlations the pipeline can use as its backbone. For each one: who developed it, what problem they were solving, the equation, every symbol, the assumptions, and exactly how we used it.

## 9.1 Yoshiro Katto and Hiroshi Ohno — the generalised correlation (1984)

### Who they were

**Yoshiro Katto** was a Japanese mechanical engineer at the University of Tokyo, working in heat transfer through the 1960s to 1980s. His field was two-phase flow and boiling heat transfer. He is one of the most-cited figures in the Critical Heat Flux literature.

**The specific problem he studied.** Most correlations of his era were fitted to water, at a particular range of pressures, in a particular range of tube sizes — and failed outside that range. Katto wanted something different: a correlation that works for **any fluid**, derived from dimensionless groups so that the principle of corresponding states does the generalising.

**What he developed.** Katto's approach, refined with **Hiroshi Ohno** in 1984, was to recognise that flow-boiling Critical Heat Flux does not follow one law but **several distinct regimes**, each governed by a different physical mechanism. He derived a separate expression for each regime from dimensional analysis, and then specified rules for selecting which one applies.

**What their contribution means in plain language.**

> *Rather than fitting one curve to everything, identify the competing physical mechanisms, write an expression for each, and at any given condition let the mechanism that fails first decide the answer.*

That last idea — **"the answer is the minimum of the candidate mechanisms"** — is physically deep. A system fails by whichever route is weakest, exactly as a chain breaks at its weakest link.

### The equation as we implement it

**Step 1 — The saturated-inlet Critical Heat Flux.**

Five candidate expressions are evaluated. Using `We_inv` for the inverse Weber number, `dr` for the density ratio, `LD` for the length-to-diameter ratio, and `Ghfg` for `G * h_fg`:

```
q2  = C * We_inv^0.043 * (D/L) * Ghfg

q3  = 0.1 * dr^0.133 * We_inv^0.333 * Ghfg / (1 + 0.0031*LD)

q4  = 0.098 * dr^0.133 * We_inv^0.433 * LD^0.27 * Ghfg / (1 + 0.0031*LD)

q5  = 0.0384 * dr^0.6 * We_inv^0.173 * Ghfg / (1 + 0.28 * We_inv^0.233 * LD)

q13 = 0.234 * dr^0.513 * We_inv^0.433 * LD^0.27 * Ghfg / (1 + 0.0031*LD)
```

**Step 2 — The regime gate.** The selection depends on whether the density ratio is below or above **0.15**:

```
if dr < 0.15 :  q_co = min(q2, q3, q4)
else         :  q_co = min(q2, q5, q13)
```

**This threshold of 0.15 is a physically motivated boundary where the controlling mechanism changes, not a tuned parameter.** Below it, the vapour is very light relative to the liquid and the dynamics are dominated by rapid vapour expansion; above it, the two phases are comparable and different terms dominate.

**Step 3 — The subcooling correction.** Three candidate coefficients:

```
K6 = 1.043 / (4 * C * We_inv^0.043)

K7 = (5/6) * (0.0124 + D/L) / (dr^0.133 * We_inv^0.333)

K9 = 1.12 * (1.52 * We_inv^0.233 + D/L) / (dr^0.6 * We_inv^0.173)
```

```
if dr < 0.15 :  K = max(K6, K7)
else         :  K = max(K6, K7, K9)
```

**Step 4 — The final answer.**

```
q_CHF  =  q_co * (1 + K * dh_sub / h_fg)
```

### Plain-English interpretation of the final equation

> *First compute what the Critical Heat Flux would be if the liquid entered exactly at its boiling point (`q_co`). Then multiply by a factor greater than one to account for the fact that colder inlet liquid can absorb extra heat before boiling. The size of that boost is `K`, and `K` depends on the channel shape and the fluid state.*

Note the final term is `dh_sub / h_fg`, which is exactly our dimensionless group `dh_sub_star`. The correlation is built from the same groups we feed the machine-learning model. That is not a coincidence — it is why those groups were chosen.

### Symbols

| Symbol | Meaning | Unit |
|---|---|---|
| q_CHF | Critical Heat Flux | W/m² |
| q_co | Critical Heat Flux at saturated inlet | W/m² |
| G | Mass flux | kg/(m²·s) |
| D | Hydraulic diameter | m |
| L | Heated length | m |
| rho_l, rho_g | Saturated liquid and vapour density | kg/m³ |
| sigma | Surface tension | N/m |
| h_fg | Latent heat of vaporisation | J/kg |
| dh_sub | Inlet subcooling enthalpy | J/kg |
| We_inv | `sigma * rho_l / (G² * L)` | — |
| dr | `rho_g / rho_l` | — |
| LD | `L / D` | — |
| C | The Katto constant, a function of L/D | — |
| K | Subcooling coefficient | — |

### Assumptions

1. Uniform heating along the heated length.
2. A circular channel, or one whose behaviour is captured by its hydraulic diameter.
3. Steady state.
4. The fluid is below its critical pressure.
5. The regime boundary at a density ratio of 0.15 is universal across fluids.

### How we used it

**Directly implemented**, as `phys_katto` in `scripts/prof_tasks/physics_arms.py`, calling `katto_ohno_chf` in `physics_pipeline/scripts/physics/correlations.py`.

**It is the default backbone of the pipeline.** [OUR EXPERIMENT] When the pipeline is allowed to choose its backbone freely on training rows, it selects Katto–Ohno on **seven of the eight** datasets.

**The property that makes it our default:** its inputs are `G, D, L, rho_l, rho_g, sigma, h_fg, dh_sub`. **It does not use vapour quality.** It is therefore an *honest* correlation in our sense — everything it needs is known before the experiment.

### Known limitation in our implementation

**[ASSUMPTION — UNVERIFIED]** The constant `C` is implemented as a piecewise function of L/D:

```
if L/D < 50   :  C = KATTO_C_SMALL
if L/D > 150  :  C = KATTO_C_LARGE
otherwise     :  linear interpolation between the two
```

This is tagged **`[S] UNVERIFIED`** in the source code. It means: **the specific values of `C` have not been checked against Katto & Ohno's original 1984 paper.** This uncertainty propagates into every result that uses the Katto backbone, which is most of them.

**This is a genuine open item and must be verified before publication.** The code includes a `return_branches` option specifically so that the sensitivity of the result to `C` can be measured — it reports which of the candidate expressions supplied the minimum, and `C` only matters when `q2` was selected.

---

## 9.2 Biasi and co-workers — the water-specific correlation (1967)

### Who they were

**L. Biasi** and colleagues worked at CISE, an Italian nuclear research organisation, in the 1960s. Their field was reactor thermal-hydraulics during the first major expansion of nuclear power.

**The specific problem.** Provide reactor designers with a simple, fast, reliable formula for Critical Heat Flux in water-cooled vertical tubes across the pressure and flow range of interest to power reactors.

**What they developed.** A two-branch empirical correlation fitted to a large body of water data. Like Katto, they recognised that one expression is not enough — but their two branches correspond to the low-quality and high-quality regimes rather than to density-ratio regimes.

**What their contribution means in plain language.**

> *For water in tubes, two simple algebraic expressions — one that dominates at low vapour content and one at high vapour content — cover the practical operating range, and you take whichever gives the larger answer.*

### The equation as we implement it

```
q_low   =  (1.883e4 / (D_cm^alpha * G_g^(1/6))) * (A / G_g^(1/6)  -  X)

q_high  =  (3.78e4 * B / (D_cm^alpha * G_g^0.6)) * (1 - X)

q_CHF   =  max(q_low, q_high)          [kW/m^2]
```

with the pressure-dependent coefficients:

```
A  =  0.7249 + 0.099 * P_atm * exp(-0.032 * P_atm)

B  = -1.159 + 0.149 * P_atm * exp(-0.019 * P_atm) + 8.99 * P_atm / (10 + P_atm^2)
```

and the diameter exponent:

```
alpha = 0.6  if D < 1 cm
alpha = 0.4  if D >= 1 cm
```

### Symbols and — critically — units

**This correlation is *dimensional*, meaning it only produces correct answers if the inputs are in the exact units it was fitted in.** This is a common source of error.

| Symbol | Meaning | Required unit |
|---|---|---|
| P_atm | Pressure | **atmospheres** (kPa ÷ 101.325) |
| G_g | Mass flux | **grams per cm² per second** (kg/(m²·s) ÷ 10) |
| D_cm | Diameter | **centimetres** |
| X | Vapour quality | dimensionless |
| q_CHF | Critical Heat Flux | kW/m² |

### Plain-English interpretation

> *Both branches say the same basic thing: Critical Heat Flux falls as vapour quality rises, because more vapour means less liquid available to cool the wall. The first branch falls off with quality directly; the second is proportional to `(1 − X)`, the liquid fraction. Smaller tubes give higher Critical Heat Flux, which is why the diameter appears in the denominator. The pressure coefficients `A` and `B` capture how water's properties change with pressure.*

### Assumptions and limitations

1. **Water only.** The coefficients are fitted to water; they are meaningless for another fluid.
2. **Vertical tubes, uniform heating.**
3. **Requires vapour quality `X` as an input** — so by our own standard it is a *local* correlation, subject to the circularity concern when used for forward prediction.
4. Valid only within the pressure and flow range of the original fit.

### How we used it

**Directly implemented**, as `phys_biasi`. It is available as one of three backbone candidates, **but only for water and only when a quality column exists**; otherwise it returns "not a number" and is excluded.

**An implementation detail that pinned a real bug.** The function selects `alpha` with a **scalar** branch (`if D < 1 cm`). If you pass an array of diameters, the comparison collapses and one exponent is applied to all rows. Our wrapper therefore loops over each *distinct* diameter and calls the correlation separately for each. Before this fix, an 8 mm default was being applied to every dataset, which penalised the correlation for a mistake in the *caller*, not in the correlation itself.

---

## 9.3 Derek Groeneveld — the 2006 Look-Up Table

### Who he is

**Derek C. Groeneveld** is a Canadian nuclear engineer, long associated with Atomic Energy of Canada Limited and the University of Ottawa. His field is reactor thermal-hydraulics, specifically Critical Heat Flux.

**The specific problem he addressed.** Every empirical correlation is accurate only in the range it was fitted to. Given that sixty years of experiments existed worldwide, Groeneveld asked a different question: instead of fitting yet another formula, why not **consolidate all the data into a single standardised table** that anyone can interpolate?

**What he developed, with international collaborators.** The **Critical Heat Flux Look-Up Table**, revised across several editions, most notably in 1995 and 2006. The 2006 edition consolidates over **30,000 experimental tube data points** from laboratories worldwide into a smoothed, thermodynamically consistent table.

**What this contribution means in plain language.**

> *Stop arguing about which formula is right. Pool every experiment humanity has performed, smooth the result into a table indexed by the three variables that matter locally, publish it as an international standard, and let everyone interpolate the same numbers.*

**Why it was a major contribution.** It gave the field a single reference standard, removed correlation-choice as a source of disagreement between organisations, and — because it is tabulated rather than fitted — it can represent non-monotonic behaviour that no simple algebraic form captures.

### The structure of the table

**[OUR EXPERIMENT]** Verified by direct inspection of the file:

| Property | Value |
|---|---|
| Indexed on | Pressure, mass flux, vapour quality |
| Pressure values | 24, from 100 to 21,000 kPa |
| Mass flux values | 21, from 0 to 8,000 kg/(m²·s) |
| Quality values | 23, from −0.5 to 1.0 |
| Total entries | 24 × 21 × 23 = **11,592** |
| Reference geometry | **Vertical tube, 8 mm inner diameter, water** |
| Entries with Critical Heat Flux = 0 | 504 (all at quality = 1.0, i.e. pure vapour — no liquid left to boil) |
| Usable non-zero entries | **11,088** |

**Diameter correction.** For a tube of a different diameter, the table value is scaled by an empirical factor of the form `(8/D)^n` with `n` approximately 0.33 to 0.50. **[LITERATURE]**

### How we used it

**Directly implemented**, as `phys_lut`, using trilinear interpolation on the regular grid via SciPy's `RegularGridInterpolator`, with extrapolation permitted outside the grid bounds.

**When it is available.** Only for **water**, and only when a vapour quality column exists — because the table is indexed on quality. On four of our eight datasets it returns "not a number" and is unavailable.

**A finding this caused.** [OUR EXPERIMENT] One of our eleven improvement attempts, `bb_ens`, was supposed to ensemble the three backbones. It returned **byte-identical** results to the baseline. Investigation showed the look-up-table arm is unavailable on half the datasets, so the ensemble was always silently dropping it and reducing to the baseline. **It was a no-op.** This was only caught because the numbers were suspiciously identical rather than merely similar.

### Limitations

1. **[LITERATURE]** It cannot extrapolate above 21 MPa.
2. It contains no surface-condition information at all.
3. It is for tubes only; other geometries need separate treatment.
4. It is an empirical consolidation, not a fundamental law.
5. **It requires local vapour quality**, so using it for forward prediction requires the iterative closure described in Section 4.7.

---

## 9.4 Warren Rohsenow, Novak Zuber, and pool boiling — why they are out of scope

**Novak Zuber** was a thermal-hydraulics researcher who in **1959** derived a Critical Heat Flux expression for **pool boiling** — boiling in a stationary pool of liquid with no forced flow — from a hydrodynamic instability argument. He proposed that the boiling crisis occurs when the counterflow of rising vapour and falling liquid becomes unstable (a Helmholtz instability), and derived a formula depending only on fluid properties and gravity.

**Why it matters to us indirectly.** **[LITERATURE]** The authors of the Groeneveld look-up table used Zuber's correlation to construct the table's zero-mass-flux skeleton and to extrapolate to the 21 MPa pressure level. So Zuber's work is embedded in a tool we use.

**Why it is out of scope for our pipeline.** Zuber's correlation has **no mass flux and no quality dependence** — it is pressure-only. Our problem is **flow boiling**, where mass flux is the dominant variable.

**Explicit scope decision.** The project supervisor directed that pool boiling be excluded entirely. Accordingly:
- All pool-boiling datasets are excluded from every result in this document.
- The pipeline **actively refuses** a request with mass flux equal to zero, with the error message: *"This pipeline covers FLOW boiling only; G = 0 is pool boiling and needs a pool-boiling backbone (Zuber), which is not fitted here."*
- **[OUR EXPERIMENT]** One row with zero mass flux was found in the Zhao dataset and is dropped at load time with an assertion.

---

## 9.5 John Hall and Issam Mudawar — the energy balance and the database

**Issam Mudawar** is a professor of mechanical engineering at Purdue University in the United States, working on boiling and two-phase cooling. **John Hall** was his collaborator.

**The specific problem they addressed.** In two companion papers published in 2000 in the *International Journal of Heat and Mass Transfer*, Hall and Mudawar assembled a large consolidated database of water flow-boiling Critical Heat Flux in tubes and used it to systematically assess existing correlations.

**Paper I** (pages 2573–2604) compiles and statistically characterises the database. **Paper II** (pages 2605–2640) develops subcooled Critical Heat Flux correlations from it.

**Their contribution in plain language.**

> *Before you propose a new correlation, assemble all the data honestly, state its ranges, and evaluate every existing correlation against it on equal terms.*

**The specific equation we take from them.** **Equation 8 of Paper II** is the steady-state energy balance derived in full in Part IV:

```
q_CHF  =  G * D * (x * h_fg + ΔH_in,sub) / (4 * L)
```

**How we used it.** **Not as a predictor.** We used it as a **diagnostic instrument** — to prove that the standard feature set is algebraically circular. This is, as far as we are aware, an unusual use of it, and it is the foundation of our methodological argument.

**[IMPORTANT]** This relation is **theirs, not ours.** It is in the textbooks. It is already in this project's own foundation document at Section 1.5. Any suggestion that we discovered it would be a serious misrepresentation and would be recognised instantly by a reviewer.

---

# PART X — THE DATASETS IN FULL DETAIL

## 10.1 Overview

**[OUR EXPERIMENT]** Eight flow-boiling datasets, **28,582 experimental points** in total, plus the look-up table as a separate reference object.

| ID | Dataset | Rows | Fluid | Geometry | Groups | CHF range (kW/m²) |
|---|---|---|---|---|---|---|
| D1 | Nuclear Regulatory Commission / Groeneveld | 24,443 | water | vertical tube | 60 references | 50 – 16,339 |
| D2 | Zhao 2020 | 1,864 | water | tube, annulus, plate | 10 authors | 800 – 19,300 |
| D3 | KAERI uniform heating | 651 | water | tube | 10 sources | 82 – 17,200 |
| D4 | KAERI non-uniform heating | 888 | water | tube | 11 sources | 143 – 28,800 |
| D5 | Helical coils, R-123 | 257 | R-123 | helical coil | 2 pressure ranges | 38 – 420 |
| D6 | Hardik helical, water | 156 | water | helical coil | 6 coils | 286 – 2,984 |
| D7 | Hardik straight, R-123 | 55 | R-123 | horizontal tube | none | 75 – 337 |
| D8 | Pioro, digitised | 268 | R-134a | horizontal and vertical tube | 9 figure groups | 8 – 430 |
| D9 | Groeneveld 2006 Look-Up Table | 11,088 | water | normalised 8 mm tube | none | 15 – 44,338 |

**Note on totals:** 24,443 + 1,864 + 651 + 888 + 257 + 156 + 55 + 268 = **28,582**. The look-up table (D9) is a smoothed reference table, **not raw measurement**, and is deliberately excluded from the merged experimental corpus.

## 10.2 D1 — The Nuclear Regulatory Commission / Groeneveld database

- **Rows:** 24,443. **Fluid:** water. **Geometry:** vertical tubes.
- **Origin:** A consolidated compilation from approximately **60 source publications**. Distributed via United States Nuclear Regulatory Commission documentation.
- **Why we needed it:** It is by far the largest available; it is the reference dataset the field uses; and its size makes statistical statements possible.
- **Grouping variable:** the source reference identifier — this is what makes the "unseen laboratory" split possible.
- **Critical note:** **The vapour quality column is back-computed from Critical Heat Flux via the energy balance.** This is not a defect of the database — it is how the compilers standardised heterogeneous sources — but it is precisely why using that column as a model input is circular. See Part IV.
- **Both inlet and local formulations are constructible** from this dataset, which makes it the only one where the formulations can be compared cleanly.

## 10.3 D3 — KAERI, uniform heating

- **Rows:** 651. **Fluid:** water. **Geometry:** tubes, uniformly heated.
- **Origin:** Korea Atomic Energy Research Institute, technical report TR-1665.
- **Groups:** 10 distinct sources.
- **Leakage:** severe — `Power`, `Perimeter`, `Area`, `MassFlow`, `InletEnthalpy` are all banned.

## 10.4 D4 — KAERI, non-uniform heating

- **Rows:** 888. **Fluid:** water. **Geometry:** tubes, **non-uniformly** heated.
- **Why it is separate and important:** the heat flux varies along the tube length. This breaks assumption 2 of the energy-balance derivation, which is why the circularity identity does **not** hold here. It is our only dataset with axially varying power.
- **Groups:** 11 sources.
- **Banned columns:** `Power`, `Perimeter`, `Area`, `MassFlow`, `InletEnthalpy`, `CHFLocation`, `QualityPosition`.

**A correction recorded honestly.** [OUR EXPERIMENT] At one stage it was asserted that no dataset in the corpus contains *inlet* quality. **That was wrong.** This dataset's `Quality` column is **inlet** quality, not outlet quality. We verified this by reconstructing inlet quality independently and correlating: **r = 0.999999 across all 888 rows.** The column was renamed `X_inlet` and the dataset's local formulation was removed, since there is no circular local form here.

## 10.5 D2 — Zhao 2020

- **Rows:** 1,864 (after dropping one row with zero mass flux). **Fluid:** water. **Geometry:** tubes, annuli, and flat plates.
- **Groups:** 10 authors.
- **Limitation:** **no inlet enthalpy is reported**, so the inlet formulation cannot be constructed at all.

**The most consequential bug in the project.** [OUR EXPERIMENT] This dataset names its diameter column `D_e [mm]` — "equivalent diameter." Our property-enrichment code looked for a column named `D_mm`, did not find it, and **silently substituted a default of 8 mm for all 1,864 rows.**

The consequence: the physics backbone was computed with a fabricated geometry for this entire dataset, and **nothing reported an error.** The correlation was being blamed for a mistake in the data loader.

**The fix, in two parts:**
1. Explicitly map `D_mm` to the dataset's `D_e [mm]` column.
2. More importantly, change the enrichment function so that a missing geometry column sets a flag `geom_known = False` and **issues a warning**, rather than substituting silently. The principle: *a default that nothing announces is indistinguishable from a measurement.*

## 10.6 D5 — Helical coils, R-123

- **Rows:** 257. **Fluid:** R-123 (a refrigerant). **Geometry:** helical coils.
- **Groups:** 2 pressure ranges — a weak grouping, which is why its grouped-split result is based on only 2 folds and is correspondingly unreliable.
- **Banned:** `Q_watt`.

**A data-recovery exercise.** [OUR EXPERIMENT] The source paper does not publish the tube diameter for each coil. However, it publishes both the electrical power and the Critical Heat Flux. Since `CHF = Q / (π · d · L)`, the diameter can be **recovered by inversion**. Doing so yielded a consistent set of values:

```
Coil_1: 9.5 mm   Coil_2: 7.5 mm   Coil_3: 5.5 mm
Coil_4: 9.5 mm   Coil_5: 7.5 mm   Coil_6: 5.4 mm
```

**[OUR INTERPRETATION]** The fact that these recovered values cluster into three clean groups (approximately 9.5, 7.5, and 5.5 mm) with one slight outlier is itself evidence that the inversion is correct — random noise would not produce that structure.

**[ASSUMPTION]** These diameters are *inferred*, not published. This is documented in the dataset schema note.

## 10.7 D6 — Hardik helical coils, water

- **Rows:** 156. **Fluid:** water at low pressure. **Geometry:** helical coils.
- **Groups:** 6 distinct coils — a genuinely useful grouping for the "unseen geometry" split.
- **Source paper held:** yes, `Hardik_Prabhu_2017_CHF_helical_coils_water_lowP_ATE.pdf`.
- **Coil diameters** taken from **Table 4 of the source paper**: Coil_1 6.0 mm, Coil_2 6.0 mm, Coil_3 8.0 mm, Coil_4 8.0 mm, Coil_5 9.7 mm, Coil_6 10.0 mm.
- **Banned:** `Q_W` and `CHF_ratio`, both demonstrated leaks (Section 5.3).

## 10.8 D7 — Hardik straight tubes, R-123

- **Rows:** 55. **Fluid:** R-123. **Geometry:** horizontal straight tubes.
- **No grouping column** — it is a single experimental campaign, so the grouped split cannot be evaluated.
- **Explicit caveat recorded in the schema:** *"55 rows: an 80/20 test set is 11 rows. Indicative only."* No statistical claim should rest on this dataset alone.
- **Its importance is disproportionate to its size**, because it is the dataset on which unconstrained pooled machine learning produced its worst failure: a coefficient of determination of **−868.6**.

## 10.9 D8 — Pioro, digitised

- **Rows:** 268. **Fluid:** R-134a. **Geometry:** horizontal and vertical tubes.
- **Groups:** 9 figure groups.

### What "digitised" means

This was asked directly, and the answer matters for interpreting this dataset's results.

**Digitising means extracting numerical data from a published graph.**

Many older papers — and many recent ones — publish their results only as plotted figures. The underlying numbers are not provided in any table and the authors may no longer have them. If you want the data, you must recover it from the picture.

**The process:**
1. Obtain a high-resolution image of the figure from the paper.
2. Load it into digitising software (common tools are WebPlotDigitizer, Engauge Digitizer, or GetData Graph Digitizer).
3. Calibrate the axes by clicking on at least two known tick marks on each axis, telling the software what numerical value each pixel position corresponds to.
4. Click on each plotted data point, or let the software auto-detect markers of a given colour and shape.
5. The software converts pixel coordinates back into data coordinates and exports them.

**Why the result is imperfect:**
- The click is made by a human eye, on a marker several pixels wide.
- The figure may be a low-resolution scan of a printed page.
- Axis calibration carries its own error.
- Logarithmic axes amplify small pixel errors into large value errors.
- Overlapping markers cannot be separated.

**The recorded uncertainty on this dataset is ±5–10%.**

### Why this dataset is the hardest in the corpus

**[OUR EXPERIMENT]** It fails even on its own random 80/20 split: coefficient of determination **−0.016**, with only **13.7%** of predictions within ±10%.

**[OUR INTERPRETATION]** A model trained on 80% of a dataset and tested on the remaining 20% — from the same source, same fluid, same geometry — should not fail. That it does means **the dataset is not internally self-consistent.** The digitisation noise is comparable to the physical signal. Three factors compound:
1. Digitisation error of ±5–10%.
2. It is the only R-134a data in the corpus, so there is nothing to learn the fluid from.
3. A constant diameter of 6.92 mm across all rows was **dropped because it has no variance** — a constant column carries no information.

**[OUR INTERPRETATION]** This is a *data* limitation, not a *model* limitation. We report it as such rather than presenting it as a modelling failure.

## 10.10 The merged corpus

- **Rows:** 28,582. **Fluids:** water, R-123, R-134a. **Geometries:** mixed.
- **Common schema** across all eight sources.
- **Grouping for the merged set is leave-one-dataset-out**, since "laboratory" is not comparable across sources.

## 10.11 What data is missing, and what we would want

**[OPEN QUESTION]** Recorded honestly:

1. **More fluids.** Three fluids, dominated by water, is thin for a model that claims fluid transferability.
2. **Surface condition.** No dataset records surface roughness, material, or wetting behaviour for flow boiling. **[LITERATURE]** These are known to matter.
3. **A genuinely external validation set.** Every dataset here has been used somewhere in development. There is **no** truly held-out corpus.
4. **Raw tabulated data for D8** rather than digitised values.
5. **Published diameters for D5** rather than inferred ones.

---

# PART XI — THE MODELS WE TRIED

## 11.1 Machine-learning terms defined before use

Because this document assumes no machine-learning background, the following terms are defined here and used freely afterward.

**Machine learning (ML).** A set of methods for building a mathematical function from examples rather than from theory. You supply many pairs of (inputs, correct answer), and an algorithm adjusts a function's internal numbers until it reproduces those answers well.

**Supervised regression.** The specific kind of machine learning used throughout this project: every training example has a known numerical answer (the Critical Heat Flux), and the task is to predict a continuous number for new inputs.

**Feature.** One input variable. Our honest feature set has five features (the five dimensionless groups).

**Training set / test set.** The data used to fit the model, and the data held back to evaluate it. The test set must never influence fitting, or the evaluation is meaningless.

**Overfitting.** When a model learns the noise and idiosyncrasies of the training data rather than the underlying relationship. It scores very well on training data and poorly on new data.

**Generalisation.** How well a model performs on data it has never seen. This is the only thing that matters in deployment.

**Hyperparameter.** A setting of the algorithm chosen before training — for example, how many trees a forest contains.

**Seed.** A number that initialises the random-number generator. Running with several different seeds and reporting the spread shows whether a result is stable or an accident of one particular random draw.

## 11.2 The thirteen machine-learning families we benchmarked

**[OUR EXPERIMENT]** Source: `results/prof_tasks/grid_results.csv`, 804 runs.

### Tree-based ensembles

**What a decision tree is.** A flowchart of yes/no questions about the input features. "Is the reduced pressure above 0.3? If yes, is the length-to-diameter ratio above 100? If yes, predict 2,400 kW/m²." Each leaf of the tree holds a predicted value.

**Why ensembles of trees.** A single tree is unstable and overfits. Building many trees and averaging their predictions dramatically improves reliability.

| Model | How it works | Why we tried it |
|---|---|---|
| **Random Forest** | Many trees, each trained on a random subset of rows and considering a random subset of features at each split. Predictions averaged. | The standard robust baseline for tabular data. |
| **Extra Trees** (Extremely Randomised Trees) | Like Random Forest, but split thresholds are also chosen at random rather than optimised. More randomness, less overfitting, faster. | **This is the model used in the final pipeline.** |
| **Extreme Gradient Boosting (XGBoost)** | Trees built *sequentially*, each new tree correcting the errors of all previous ones. | Usually the strongest performer on tabular competition data. |
| **LightGBM** | Gradient boosting with a leaf-wise growth strategy and histogram-based splitting. Faster on large data. | Speed on our 24,443-row dataset. |
| **CatBoost** | Gradient boosting with ordered boosting, designed to reduce a subtle form of overfitting. | Reported strong on small datasets — relevant to our small sets. |
| **Histogram Gradient Boosting (HistGB)** | Scikit-learn's own fast boosting implementation. | Built-in, fast, no extra dependency. |

**The critical shared weakness of all tree methods.** A tree predicts a **constant** within each leaf. Outside the range of the training data, that constant is simply held flat forever. A tree ensemble asked about conditions beyond its training envelope produces a **flat step function** — it cannot extrapolate at all. This is the structural reason pooled tree models produced coefficients of determination of −868.

### Kernel and distance methods

| Model | How it works | Why we tried it |
|---|---|---|
| **Gaussian Process Regression, Matérn kernel (GPR-Matérn)** | Places a probability distribution over all possible smooth functions and conditions it on the observed data. Returns a prediction *and* an uncertainty. | Excellent on small, smooth datasets, and provides uncertainty natively. |
| **Support Vector Regression, Radial Basis Function kernel (SVR-RBF)** | Finds a function that stays within a tolerance tube of the data while being as flat as possible, using a kernel to handle non-linearity. | Robust to outliers; strong on small data. |
| **k-Nearest Neighbours, k = 5** | Prediction is the average of the five most similar training rows. | The simplest possible baseline. If it performs well, the problem is mostly interpolation. |

**Note.** A variant `GPR_Matern_sub2000` was needed because Gaussian Process Regression costs time proportional to the cube of the number of training rows, making it infeasible on 24,443 rows; that variant subsamples to 2,000.

### Neural and linear methods

| Model | How it works | Why we tried it |
|---|---|---|
| **Multi-Layer Perceptron (MLP)** | A neural network: layers of simple units, each computing a weighted sum of its inputs followed by a non-linear function. | The standard neural approach and the one most published Critical Heat Flux papers use. |
| **Ridge regression (Linear)** | A straight-line fit with a penalty on large coefficients. | The floor. Any method that cannot beat this is not worth using. |
| **Polynomial degree-2 Ridge (Poly2_Ridge)** | Ridge regression on the features plus all their squares and pairwise products. | Tests whether mild curvature is enough. |

## 11.3 The nine physics and hybrid approaches

**[OUR EXPERIMENT]** Source: `results/prof_tasks/physics_results.csv`, 157 runs.

| Approach | Category | What it does |
|---|---|---|
| `PHYS_KattoOhno` | Pure physics, **zero fitted parameters** | Katto–Ohno correlation applied directly. |
| `PHYS_Biasi` | Pure physics, zero fitted parameters | Biasi correlation applied directly. |
| `PHYS_LUT2006` | Pure physics, zero fitted parameters | Trilinear interpolation of the 2006 look-up table. |
| `PIML_dimensionless` | Physics-informed machine learning | Gradient boosting trained in dimensionless space to predict the **boiling number**, then converted back to Critical Heat Flux. |
| `BOUND_Katto_x1.5` | Bounded correction | Machine-learning correction on the Katto baseline, clipped to at most a factor of 1.5. |
| `BOUND_Katto_x2.0` | Bounded correction | Same, clipped at factor 2.0. |
| `BOUND_Katto_x3.0` | Bounded correction | Same, clipped at factor 3.0. |
| `BOUND_Katto_unbounded` | Control | Same correction with **no** clip. The control that demonstrates what the bound is buying. |
| `ITER_closure_GB` | Iterative closure | A local-form model solved **jointly with the energy balance from inlet conditions only** — the way reactor codes actually use a look-up table. This makes a local-form model legitimately predictive rather than circular. |

**Why the `BOUND_*` family exists.** It is a direct, deliberate sweep of the single design parameter that became the core of the final pipeline: *how far should the machine-learning model be allowed to move the physics answer?* Running 1.5×, 2.0×, 3.0×, and unbounded side by side measures the trade-off rather than assuming it.

## 11.4 Total benchmark size

**[OUR EXPERIMENT]**

```
grid_results.csv    :  804 runs, 13 machine-learning model families
physics_results.csv :  157 runs,  9 physics and hybrid approaches
                      ---------
combined            :  961 runs, 22 distinct approaches
```

Across **10 datasets**, **3 formulations** (local, inlet, reduced), **4 split types**, and multiple random seeds.

> **A correction to an earlier figure.** An earlier draft of the project's résumé bullet points stated *"21 model families over 1,001 runs."* The verified figures are **22 approaches over 961 runs**. Use the verified numbers.

## 11.5 Deep-learning architectures: considered, not adopted

**[OUR PROPOSED IDEA]** — these were considered and explicitly **not implemented**. They are listed so nobody mistakes a plan for a result.

| Architecture | Why it was considered | Why it was not adopted |
|---|---|---|
| **Convolutional Neural Network (CNN)** — a network that learns local patterns via sliding filters, designed for images | Used in the literature for bubble-image analysis | Our data is tabular, with five features. There is no spatial structure for convolution to exploit. |
| **Long Short-Term Memory network (LSTM)** — a recurrent network that maintains a memory across a sequence | Genuinely relevant to the non-uniform-heating dataset, where upstream axial history matters | Only one of eight datasets has axial structure, and it has 888 rows. Not enough to train a recurrent network. |
| **Transformer encoder** — a network using attention to weigh relationships between all elements of an input | Considered for the merged multi-dataset corpus | 28,582 rows with five features is orders of magnitude too small. Transformers need vastly more data. |
| **Low-Rank Adaptation (LoRA)** — a technique for cheaply fine-tuning a large pretrained model | Considered for adapting a look-up-table-pretrained model to each dataset | Requires a large pretrained model we do not have. |
| **Mixture of Experts** — several specialist sub-models plus a gate that routes each input to one | This *was* attempted, in the form of the router | **It failed twice.** See Part XIII. |
| **Deep ensembles** — several independently trained networks, whose spread estimates uncertainty | Attempted for uncertainty quantification | **Failed badly.** Nominal 95% intervals achieved 9–22% actual coverage. See Section 14.6. |

**[OUR INTERPRETATION]** The honest summary is that this is a **small-data, low-dimensional, physics-constrained** problem. Deep learning's advantages — automatic feature extraction from high-dimensional raw signals — do not apply. Five well-chosen dimensionless groups already encode the physics that a deep network would have to rediscover from far more data than we have.

---

# PART XII — THE BENCHMARK: ANSWERING THE SUPERVISOR'S QUESTIONS

## 12.1 The questions posed

The project supervisor set three tasks:

1. **Are there specific models, architectures, or techniques that work for specific types of datasets?**
2. **Is there a technique, model, or architecture that performs well overall?**
3. **Opinion on a pipeline that classifies incoming input data as *seen* or *unseen* and routes accordingly.**

With two standing constraints:
- Always report **both** an 80/20 and a 70/30 train-test split.
- Always report **how many predictions fall within ±10%** of the actual Critical Heat Flux.
- Exclude all pool-boiling datasets and models.

## 12.2 Experiment 2 — The full model grid

**Objective.** Answer Questions 1 and 2 with measurement rather than opinion.

**Hypothesis.** Different datasets would favour different models, and one family (probably gradient boosting) would be best overall.

**Setup.** Every combination of 22 approaches × 10 datasets × 3 formulations × 4 splits × multiple seeds = **961 evaluated runs**.

### Results — Question 1: best model per dataset

**[OUR EXPERIMENT]** Honest (reduced) features, 80/20 split. Source: `results/prof_tasks/ANSWERS_Q1_Q2.txt`.

| Dataset | Rows | Fluid | Geometry | Best model | R² | ±10% | MAPE |
|---|---|---|---|---|---|---|---|
| D5 Helical R-123 | 257 | R-123 | helical coil | **CatBoost** | 0.963 | 79.9% | 7.0% |
| D6 Hardik helical | 156 | water | helical coil | **Extra Trees** | 0.955 | 86.7% | 6.5% |
| D7 Hardik straight | 55 | R-123 | horizontal tube | **GPR-Matérn** | 0.943 | 84.5% | 5.9% |
| D4 KAERI non-uniform | 888 | water | tube | **Random Forest** | 0.928 | 56.5% | 17.9% |
| D1 Nuclear Reg. Comm. | 24,443 | water | vertical tube | **Random Forest** | 0.925 | 46.9% | 16.5% |
| M merged | 28,583 | mixed | mixed | **Random Forest** | 0.918 | 47.2% | 17.2% |
| D3 KAERI uniform | 651 | water | tube | **Extra Trees** | 0.812 | 43.9% | 26.6% |
| D2 Zhao | 1,865 | water | tube/annulus/plate | **HistGB** | 0.715 | 40.0% | 18.9% |
| D8 Pioro | 268 | R-134a | horiz/vert tube | **GPR-Matérn** | 0.271 | 13.7% | 68.2% |
| D9 Look-Up Table | 11,088 | water | 8 mm tube | **SVR-RBF** | −0.003 | 9.0% | 365.2% |

**Interpretation — the answer to Question 1 is "yes, but weakly, and it depends on size."**

A clear pattern emerges, and it is a pattern about **dataset size, not about physics**:

- **Small, smooth datasets (55–257 rows)** favour **Gaussian Process Regression** and **CatBoost**. These methods impose smoothness, which is a form of regularisation that matters enormously when data is scarce.
- **Large, heterogeneous datasets (888–28,583 rows)** favour **tree ensembles** — Random Forest and Extra Trees. With enough data, flexibility beats smoothness.

**But the honest caveat, which is the more important finding:** the spread between the best and third-best model on any dataset is *far smaller* than the spread between splits on the *same* model. Changing the validation protocol moves the result by more than changing the model does.

> **Choice of validation matters more than choice of model.**

**On D9, the Look-Up Table, R² = −0.003:** this is **not** a model failure. As established in Section 4.7, the theoretical ceiling from pressure and mass flux alone is **0.1538**. The table's third indexing variable has been deliberately removed to test the honest formulation, and without it the mapping is one-to-many. The measured −0.003 is what "no usable signal" looks like.

### Results — Question 2: is one model good overall?

**[OUR EXPERIMENT]** The answer is **no — no single *model* is**.

Tree ensembles (Random Forest, Extra Trees) are the most consistent performers across datasets. But:

| Split | Mean best R² across datasets (honest features) |
|---|---|
| 80/20 random | 0.743 |
| 70/30 random | 0.741 |
| Leave-one-source-out | **−3.115** |

**Every model family, without exception, goes negative on a held-out laboratory or a held-out dataset.**

The consistency of the 80/20 and 70/30 numbers (0.743 vs 0.741) is worth noting on its own: the random-split result is not an artefact of the particular ratio. It is stable and it is stably misleading.

**Decision.** This result forced the conclusion that the answer to Question 2 is not a *model* but an *architecture*: a structure in which a model that fails gracefully is combined with something that cannot fail catastrophically. That is what Part XIV builds.

## 12.3 The formulation gap, restated

**[OUR EXPERIMENT]** The single table that summarises Parts IV, V, and VI:

| Split | Formulation | Datasets | Mean best R² | Mean best ±10% |
|---|---|---|---|---|
| 80/20 | local (with quality) | 10 | 0.973 | 86.3% |
| 80/20 | inlet | 4 | 0.961 | 86.5% |
| 80/20 | **reduced (honest)** | 10 | **0.743** | **50.8%** |
| 70/30 | local | 10 | 0.973 | 85.3% |
| 70/30 | inlet | 4 | 0.961 | 83.1% |
| 70/30 | **reduced (honest)** | 10 | **0.741** | **48.9%** |
| LOSO | local | 8 | −1.144 | 33.3% |
| LOSO | inlet | 3 | −2.732 | 38.9% |
| LOSO | **reduced (honest)** | 8 | **−3.115** | **27.3%** |

**Read this table from top-left to bottom-right and you have the entire scientific argument of the project.** Going from the reported protocol (local features, random split, 0.973) to the honest protocol (reduced features, unseen source, −3.115) is a journey from apparent mastery to demonstrable failure — with no change to the models at all.

---

# PART XIII — THE ROUTING FAILURE (QUESTION 3)

## 13.1 What was proposed

The supervisor's third question proposed an appealing architecture:

> Build a pipeline that classifies incoming input data as **seen** or **unseen**, and routes accordingly — familiar data to a specialised, high-accuracy model; unfamiliar data to something more conservative.

This is a **Mixture of Experts** design. It is intuitive, it is well-established in machine learning generally, and on the face of it, it directly addresses the generalisation collapse.

**We built it. It failed twice. The reasons are the most scientifically interesting findings in the project.**

## 13.2 Failure 1 — Hard routing

**Objective.** Detect out-of-distribution inputs and switch to a different model.

**Setup.** A gate computes a novelty score for each incoming row based on its distance from the training data in dimensionless space. Above a threshold, the row is declared "unseen" and routed to a fallback.

**Result.** **[OUR EXPERIMENT]** Source: `results/prof_tasks/ANSWER_Q3_router.txt`, `router_results.csv`.

| Measurement | Value |
|---|---|
| Router coefficient of determination on an unseen fluid | **−1276** |
| An *oracle* router (one that magically always chooses correctly) would score | **0.831** |
| Our router actually scored | **0.099** |

**Diagnosis — and this is the important part.**

The gate is blind, and it is blind for a reason that is **structural, not fixable by tuning**.

Recall from Section 8.3 why we use dimensionless groups: they deliberately **erase fluid identity**. That erasure is the *entire mechanism* by which knowledge transfers from water to a refrigerant. The model never sees "R-134a"; it sees a density ratio and a reduced pressure.

But that is exactly what makes detection impossible. The R-134a rows from the Pioro dataset — a fluid appearing in **zero** training rows — land **inside** the training envelope in dimensionless space. Their density ratios and reduced pressures overlap with water's. **Not one of the 268 rows was flagged as novel.**

> **The representation that enables transfer is the same representation that blinds the novelty detector.**

These two objectives are in **direct tension**. Make the representation more fluid-specific and detection improves, but transfer breaks. Make it more abstract and transfer works, but detection fails.

**[OPEN QUESTION]** We have no solution to this. **[OUR INTERPRETATION]** We believe this is the strongest genuine research gap the project has identified, and we have not found it stated in this form in the literature we have read.

## 13.3 Failure 2 — Soft blending

**Objective.** If a hard switch is too brittle, try a soft blend: weight the prediction toward a free, unanchored model when the trust score is high, and toward the physics anchor when it is low.

**Hypothesis.** A continuous blend avoids the catastrophic cost of a discrete switching error.

**Result.** **[OUR EXPERIMENT]** Held-out dataset D7 (Hardik straight tubes) fell from a coefficient of determination of **0.792 to −206.9**.

**Diagnosis.** The trust score measures distance **in dimensionless physics space**. A new laboratory running conventional experiments produces rows that sit comfortably *inside* that space — the trust score rated D7 at **0.72**, i.e. high trust — while still being genuinely unseen data from a source the model had never encountered.

> **Trust detects unfamiliar *physics*. It does not detect an unfamiliar *source*.**

So it must never be used to hand rows over to an unanchored model.

**Decision.** The `blend_free` option **remains in the code but is switched off by default**, with the measurement written directly into the source as a comment so that nobody re-enables it without re-measuring. The comment reads: *"Keep `blend_free=False` unless re-measured."*

**[OUR INTERPRETATION]** Leaving a failed option in the code with its failure documented is deliberate. A deleted experiment is an experiment someone else will repeat.

## 13.4 The lesson that became the architecture

The two failures share one structure, and articulating it was the turning point of the project:

> **Routing is a switch. Every switching error costs a factor of a thousand.**
> **Bounding is a dial. Every setting has a measured, monotonic trade-off.**

A switch has no graceful failure mode: choose wrong and you get −1276. A dial does: turn it down and the answer degrades smoothly toward the physics correlation, which is never brilliant but is never insane.

**This single sentence is the design principle of the final pipeline.**

## 13.5 Our answer to Question 3, as delivered

**[OUR EXPERIMENT]** The supervisor's proposed architecture is sound in intent but **not implementable on this corpus**, for a measured reason: the novelty gate cannot see the novelty that matters. We therefore implemented the *intent* — treat unfamiliar data more conservatively — through a mechanism that degrades continuously instead of switching discretely.

---

# PART XIV — THE FINAL PIPELINE: DESIGN AND EVERY LAYER

## 14.1 The design principle

Stated in one sentence:

> **Let established physics produce the answer, and let machine learning supply only a correction whose size is limited by how much we trust the input.**

The architecture has five layers. Each is explained below with its input, output, purpose, mathematics, physical meaning, and failure modes.

## 14.2 Layer 0 — Property enrichment and dimensionless transformation

**Input.** A raw row: fluid name, pressure, mass flux, diameter, heated length, inlet temperature or inlet subcooling.

**Operation.**
1. Resolve the fluid name to a CoolProp substance. **Raise an error if unknown** — never substitute.
2. Look up saturation properties at that pressure: liquid density, vapour density, surface tension, latent heat, saturated liquid enthalpy, saturation temperature, critical pressure.
3. Compute inlet subcooling enthalpy from whichever of the three routes is available (Section 8.3, Group 4).
4. Record whether geometry was actually supplied (`geom_known`) and warn if not.
5. Compute the five dimensionless groups.

**Output.** A row in dimensionless physics space.

**Purpose.** *Translate from the language of a particular experiment into the language of physics.* This is what makes an unseen fluid predictable at all.

**Failure modes.** Unknown fluid (raises). Pressure at or above critical (raises). Missing geometry (warns, and marks the physics baseline as unreliable).

## 14.3 Layer 1 — The physics backbone

**Input.** The enriched row.

**Operation.** Evaluate the available correlations, then select one. Selection is by **median absolute logarithmic error on training rows only**:

```
for each candidate arm:
    if the arm is valid on fewer than 90% of training rows: skip it
    error = median( |log( prediction / actual )| )
select the arm with the smallest error
```

**Output.** A physics prediction with **zero fitted parameters**.

**Why the median of absolute log error.** The logarithm makes the measure *relative* — being wrong by a factor of two counts the same whether the true value is 100 or 10,000 kW/m². The median makes it robust to a handful of extreme outliers.

**Why "valid on at least 90% of rows."** Because the look-up table and Biasi return "not a number" for non-water fluids. An arm that works on only half the rows would win on a favourable subset and then be unavailable at prediction time.

**Critically: this is one discrete choice, made on training rows only. It never sees a test row.**

**[OUR EXPERIMENT] Which backbone is actually selected, measured directly:**

| Dataset | Backbone selected | Does that backbone use vapour quality? |
|---|---|---|
| D1 Nuclear Regulatory Commission | `phys_katto` | **No** |
| D2 Zhao 2020 | `phys_lut` | **Yes** ⚠ |
| D3 KAERI uniform | `phys_katto` | **No** |
| D4 KAERI non-uniform | `phys_katto` | **No** |
| D5 Helical R-123 | `phys_katto` | **No** |
| D6 Hardik helical | `phys_katto` | **No** |
| D7 Hardik straight | `phys_katto` | **No** |
| D8 Pioro R-134a | `phys_katto` | **No** |

**A caveat that must be stated, and which was found while preparing this document.**

On **seven of eight** datasets the pipeline selects Katto–Ohno, which uses only pre-test quantities. Those results are fully honest.

On **D2 (Zhao)** the pipeline selects the look-up table backbone, **which is indexed on local vapour quality**. That means the reported Zhao results benefit from a quantity that is not known before the experiment.

**[OUR EXPERIMENT]** We quantified the effect by forcing the honest backbone on that dataset:

| D2 Zhao, 80/20, mean of 3 seeds | R² | ±10% |
|---|---|---|
| Backbone freely selected (`phys_lut`, uses quality) | 0.836 | 65.6% |
| Backbone forced to `phys_katto` (honest, no quality) | **0.650** | **41.5%** |

**[OUR INTERPRETATION]** The Zhao headline number is inflated by roughly 0.19 in coefficient of determination by the quality-dependent backbone. This does **not** affect the other seven datasets, and it does not affect the leave-one-dataset-out results that constitute our main claim (where Zhao scores −0.429 regardless). But it should be disclosed, and a `backbone='phys_katto'` constraint should be considered for any published version.

**This is recorded here rather than quietly omitted.**

## 14.4 Layer 2 — The bounded correction model

**Input.** The five dimensionless features, and the physics backbone's prediction.

**Operation.** A single **Extra Trees regressor with 300 trees**, trained on:

```
target  =  log( actual CHF  /  backbone prediction )
```

**This is the conceptual heart of the pipeline, and it deserves a careful explanation.**

The model's job is **not** "predict the Critical Heat Flux." Its job is **"predict how wrong the textbook correlation is at this condition."**

**Why this is better, in three respects:**

1. **The target is well-behaved.** The ratio of measurement to correlation hovers around 1. Its logarithm hovers around 0. It has a small dynamic range, unlike Critical Heat Flux itself which spans three orders of magnitude.

2. **The physics does the heavy lifting.** All the hard non-linearity — the effect of pressure through fluid properties, the length-to-diameter effect, the subcooling boost — is already captured by the correlation. The machine-learning model only has to learn the *remaining structured error*, which is a far easier function.

3. **Failure degrades to something sensible.** If the model has nothing useful to say and predicts zero, then `log(ratio) = 0`, so `ratio = 1`, so the pipeline returns **exactly the physics correlation**. A model that knows nothing produces the textbook answer rather than nonsense. *This property is the entire reason the architecture is safe.*

**Why one pooled model rather than one per dataset.** Because per-dataset routing failed twice (Part XIII). A single pooled model trained on everything avoids the need for a gate.

**Why Extra Trees rather than XGBoost or a neural network.** Extra Trees performed comparably in the benchmark, trains quickly, requires no hyperparameter tuning to be competitive, and its extra randomisation makes it less prone to overfitting on small data. **[ASSUMPTION]** This choice has **not** been tuned or systematically compared within the pipeline architecture — only in the standalone grid. It is a reasonable default, not a proven optimum.

## 14.5 Layer 3 — The trust score, and what it actually controls

**Input.** The dimensionless features of the incoming row, plus statistics recorded from the training data.

**Three signals are computed.**

### Signal 1 — Mahalanobis distance

**What it is.** A measure of how far a point is from a cloud of points, which accounts for the *shape* of the cloud rather than treating it as a sphere.

**Why not plain Euclidean distance.** Because the five dimensionless features are correlated and have wildly different scales. A point might be perfectly ordinary in every individual feature yet sit in a combination that never occurs — for instance, a high reduced pressure together with a very high length-to-diameter ratio. Mahalanobis distance catches that; Euclidean distance does not.

**How it is computed.**
```
1. Take the logarithm (base 10) of all features -- they span orders of magnitude
2. mu  = the median of the training cloud
3. cov = the covariance matrix of the training cloud (plus a tiny ridge for stability)
4. inv = the pseudo-inverse of cov
5. d   = sqrt( (x - mu)^T  inv  (x - mu) )
```

**Normalisation.** The raw distance is mapped to a 0–1 scale using the 50th and 95th percentiles of the distances *within the training data itself*. A row at the training median scores 0 (completely familiar); a row at the 95th percentile or beyond scores 1 (novel).

### Signal 2 — Correlation disagreement (`corr_spread`)

**What it is.** The standard deviation of the logarithms of all available physics correlations' predictions for this row.

**The idea in plain words:**

> *When Katto–Ohno, Biasi, and the look-up table all agree, we are in well-charted territory where sixty years of empirical work is consistent. When they disagree with each other, we are in a region where the physics itself is uncertain — and a machine-learning correction built on top of that should not be trusted far.*

**[OUR INTERPRETATION]** This is an elegant signal because it is completely independent of the training data. It measures the uncertainty of the *physics*, not of the *model*.

### Signal 3 — Unseen fluid flag

A binary indicator: is this fluid absent from the training set? Simple, and the one signal that is reliable when it fires.

### Combining them

```
trust  =  1  -  max( distance_signal, spread_signal, fluid_new_signal )
```

**Note carefully: the `max`, not the average.** This is deliberately **pessimistic**. If *any* signal says "unfamiliar," trust drops. A row must look familiar on *every* measure to be trusted.

### What trust controls — the bound

```
cap  =  log(1.5)  +  trust * ( log(10) - log(1.5) )

prediction  =  backbone * exp( clip( model_output, -cap, +cap ) )
```

**In plain words:**

| Trust | The correction may change the physics answer by at most |
|---|---|
| 1.0 (fully familiar) | a factor of **10** |
| 0.5 | a factor of about **3.9** |
| 0.0 (completely novel) | a factor of **1.5** |

**This is the crucial design decision, and it is what to emphasise to a reviewer:**

> **The trust score never chooses a model. It never flips a switch. It tightens a leash.**

A low-trust prediction **cannot** be catastrophic, because low trust means the answer is nearly pure physics. The worst case is bounded by the correlation's own worst case, which is bad but never −868.

### An honest finding about the trust score

**[OUR EXPERIMENT]** An ablation study measured each trust signal in isolation. Median coefficient of determination across held-out datasets:

- Using distance alone, spread alone, or the fluid flag alone: **0.736 to 0.753**.

**The three signals are near-interchangeable.**

**[OUR INTERPRETATION]** This is worth stating plainly rather than hiding: **the bounded-correction *structure* does the work, not the cleverness of the gate.** Any reasonable novelty signal produces essentially the same result.

This is not a disappointing finding — it is a **reassuring** one. It means the pipeline's robustness does not depend on a fragile, finely tuned component. That is exactly why it succeeds where the router (which depended entirely on gate quality) failed.

## 14.6 Layer 4 — Calibrated prediction intervals

**Input.** A point prediction and a trust score.

**The problem this solves.** A prediction without an uncertainty estimate is not usable in a safety application. But uncertainty estimates can themselves be badly wrong.

**[OUR EXPERIMENT]** **This project's deep ensembles produced nominal 95% intervals whose actual coverage was 9–22%.** An interval claimed to contain the truth 95% of the time contained it 9% of the time. That is not merely imprecise; it is actively dangerous, because it conveys false confidence.

**The method we use instead: split conformal prediction.**

**How it works.**
1. Set aside 20% of the training data as a **calibration set**. It is never used to fit the model.
2. Fit the model on the remaining 80%.
3. Predict on the calibration set and record the absolute logarithmic errors.
4. The interval half-width is the (1 − α) quantile of those errors. For a 90% target, take the 90th percentile.
5. For a new prediction, the interval is `prediction × exp(±width)`.

**Why this is trustworthy where deep ensembles were not.** Conformal prediction makes **no assumption about the distribution of errors**. It does not assume Gaussian noise. It does not rely on the model's internal notion of its own confidence. It simply **measures how wrong the model actually was on data it had not seen**, and uses that empirical distribution.

**Widening with low trust.**
```
width  =  q_global * ( 1 + 2 * (1 - trust) )
```
A fully trusted prediction gets the base width; a completely untrusted one gets three times that width.

**A calibration-leakage bug that was found and fixed.** [OUR EXPERIMENT] The Mahalanobis reference cloud was originally built from **all** training rows — including the conformal calibration slice. That meant a calibration row's novelty was being measured against a cloud that **contained that row itself**. The result: trust came out systematically optimistic, and therefore the conformal quantile came out too tight. The fix: build the reference cloud from the **fit rows only**. This is documented in a comment in the source.

**Two conformal modes, and why the tighter one is off.** A "Mondrian" variant computes a separate quantile per trust band. It is tighter in-distribution, but **worse on a held-out laboratory** — because trust rates a new laboratory as high-trust and therefore hands it the *tightest* interval, which is precisely backwards. The global mode is the default.

**Three-level output label.**

| Trust | Label | What it tells the user |
|---|---|---|
| ≥ 0.66 | `interpolating` | Inside well-sampled territory; the correction is fully active. |
| 0.33 – 0.66 | `extrapolating` | Outside the dense region; the correction is limited. |
| < 0.33 | `correlation fallback` | Essentially the physics correlation; treat as a correlation estimate. |

## 14.7 The deployment interface

The function `predict_one()` is the honest interface. It asks **only for what an experimenter knows before the test**:

```python
predict_one(fluid, P_kPa, G_kg_m2s, D_mm, L_mm, Tin_C=None, dHin_sub_kJkg=None)
```

**It never asks for outlet quality. That is the whole point.**

**It returns:**
```python
{ "CHF_kW_m2":        2903.5,
  "interval_kW_m2":   (2180.1, 3866.2),
  "confidence":       "interpolating",
  "trust":            0.812,
  "physics_only_kW_m2": 2611.0,     # what the correlation alone said
  "backbone":         "phys_katto",
  "warnings":         [] }
```

**Returning `physics_only_kW_m2` alongside the prediction is deliberate.** It lets the user see exactly how much the machine learning moved the answer, and judge for themselves.

**Input validation — every check exists because a real bug made it necessary:**

| Check | Behaviour | Why |
|---|---|---|
| Non-finite inputs | Raise | Silent "not a number" propagation. |
| Diameter or length ≤ 0 | Raise | Physically impossible. |
| Pressure ≤ 0 | Raise | Physically impossible. |
| Pressure ≥ critical pressure | Raise | Critical Heat Flux is not defined above the critical point. |
| **Mass flux = 0** | **Raise** | This is pool boiling, explicitly out of scope. |
| Unknown fluid | Raise | Prevents silently returning water's properties. |
| Mass flux > 10,000 | Warn | Beyond the training range (maximum about 8,000). |
| Diameter outside 1–50 mm | Warn | Beyond the training range. |
| Fluid not in training set | Warn | Prediction rests on the correlation using that fluid's real properties. |

## 14.8 The nine regression tests

**[OUR EXPERIMENT]** `scripts/prof_tasks/test_pipeline.py`. Each test pins a bug that was once real:

1. Rejects pool boiling (mass flux = 0).
2. Rejects impossible geometry.
3. Rejects supercritical pressure.
4. Rejects an unknown fluid.
5. An unseen fluid uses **its own** properties, not water's.
6. The interval brackets the point prediction.
7. Never returns zero or a negative Critical Heat Flux.
8. `blend_free` stays off by default.
9. Save and load round-trip produces identical predictions.

**A bug that test 9 exists because of.** The `save()` method originally **truncated the reference cloud in place**, meaning calling `save()` silently shrank the live object. Predictions happened to match in testing, but a method that mutates the object it is serialising is not something to hand to anyone. It now copies first.

---

# PART XV — COMPLETE RESULTS

All results in this part are **[OUR EXPERIMENT]**, sourced from `results/prof_tasks/FINAL_results.csv` and `final_review.log`.

## 15.1 Split 1 — 80/20 random, duplicate-aware, 10 seeds

| Dataset | n | R² | sd | ±10% | ±20% | MAPE | RMSE | Coverage |
|---|---|---|---|---|---|---|---|---|
| D1 Nuclear Reg. Comm. | 24,443 | **0.994** | 0.001 | **96.2%** | 99.0% | 3.2% | 128.0 | 0.95 |
| D4 KAERI non-uniform | 888 | 0.973 | 0.009 | 81.4% | 93.1% | 16.2% | 338.8 | 0.93 |
| D3 KAERI uniform | 651 | 0.928 | 0.037 | 88.5% | 92.6% | 31.3% | 355.4 | 0.93 |
| D6 Hardik helical | 156 | 0.901 | 0.057 | 77.8% | 90.0% | 9.3% | 186.5 | 0.94 |
| D5 Helical R-123 | 257 | 0.889 | 0.041 | 64.0% | 84.4% | 10.1% | 26.5 | 0.96 |
| D7 Hardik straight | 55 | 0.883 | 0.071 | 58.2% | 81.8% | 11.8% | 22.7 | 0.96 |
| D2 Zhao 2020 | 1,864 | 0.851 | 0.028 | 63.5% | 84.7% | 10.9% | 784.9 | 0.96 |
| D8 Pioro R-134a | 268 | −0.016 | 0.143 | 13.7% | 25.5% | 78.1% | 92.5 | 0.95 |

## 15.2 Split 2 — 70/30 random, duplicate-aware, 10 seeds

| Dataset | n | R² | sd | ±10% | ±20% | MAPE | RMSE | Coverage |
|---|---|---|---|---|---|---|---|---|
| D1 Nuclear Reg. Comm. | 24,443 | **0.993** | 0.001 | **95.8%** | 99.0% | 3.3% | 130.5 | 0.95 |
| D4 KAERI non-uniform | 888 | 0.970 | 0.011 | 81.9% | 93.6% | 14.6% | 346.5 | 0.94 |
| D3 KAERI uniform | 651 | 0.935 | 0.034 | 88.4% | 92.3% | 31.1% | 352.0 | 0.92 |
| D5 Helical R-123 | 257 | 0.896 | 0.045 | 62.2% | 86.3% | 9.9% | 24.7 | 0.95 |
| D6 Hardik helical | 156 | 0.891 | 0.037 | 73.2% | 88.3% | 10.1% | 200.6 | 0.95 |
| D7 Hardik straight | 55 | 0.852 | 0.075 | 55.3% | 79.4% | 12.9% | 26.4 | 0.96 |
| D2 Zhao 2020 | 1,864 | 0.843 | 0.025 | 62.8% | 84.5% | 11.0% | 788.5 | 0.96 |
| D8 Pioro R-134a | 268 | −0.007 | 0.140 | 12.1% | 23.9% | 74.9% | 92.2 | 0.94 |

**Interpretation.** The 80/20 and 70/30 results agree to within about 0.01 in coefficient of determination on every dataset. **The results are not an artefact of the split ratio.** The standard deviations across 10 seeds are small (0.001 on the large dataset, up to 0.14 on the smallest), confirming the numbers are stable rather than lucky draws.

## 15.3 Split 3 — Grouped hold-out (unseen laboratory / geometry / pressure range)

| Dataset | What the group is | Folds | Mean R² | Median R² | **Worst R²** | sd | ±10% | Coverage |
|---|---|---|---|---|---|---|---|---|
| D1 Nuclear Reg. Comm. | unseen lab (60 refs) | 10 | **0.976** | 0.982 | **0.953** | 0.014 | 83.9% | 0.77 |
| D4 KAERI non-uniform | unseen lab (11 sources) | 10 | 0.464 | 0.529 | −1.044 | 0.565 | 38.1% | 0.76 |
| D6 Hardik helical | unseen coil geometry (6) | 6 | 0.332 | 0.664 | −1.403 | 0.878 | 50.0% | 0.80 |
| D2 Zhao 2020 | unseen lab (10 authors) | 10 | −0.083 | 0.270 | −1.948 | 0.910 | 31.5% | 0.88 |
| D5 Helical R-123 | unseen pressure range (2) | 2 | −0.126 | −0.126 | −0.563 | 0.618 | 16.3% | 0.84 |
| D8 Pioro R-134a | unseen figure group (9) | 9 | −0.231 | −0.094 | −0.867 | 0.301 | 14.1% | 0.96 |
| D3 KAERI uniform | unseen lab (10 sources) | 10 | **−8.401** | 0.108 | **−73.464** | 24.436 | 44.3% | 0.61 |
| D7 Hardik straight | — | — | *no group column — single campaign, not evaluable* | | | | | |

**Interpretation — this table is important and uncomfortable.**

On the large dataset (D1), holding out an entire laboratory barely hurts: 0.976 mean, and crucially the **worst** of ten folds is still **0.953**. With 60 laboratories and 407 rows each, the model has seen enough variety that one more laboratory is not a surprise.

On the small datasets it collapses. D3 has a **mean of −8.4** but a **median of 0.108** — meaning most folds are mediocre and one or two are catastrophic (worst: −73.5). **Reporting only the mean here would be misleading in one direction, and reporting only the median would be misleading in the other.** Both are given, plus the worst case, which is the number that matters for a safety application.

**[OUR INTERPRETATION]** This is the confounding finding of Part VI appearing in the pipeline's own results. Datasets with few rows per laboratory cannot support laboratory-level generalisation, because laboratory identity is most of the signal.

## 15.4 Split 4 — Leave-One-Dataset-Out: the headline result

Train on seven datasets. Test on the entire eighth, never seen in any form. 5 seeds.

| Held-out dataset | Pure physics | **Pipeline** | Pooled ML | Pipeline ±10% |
|---|---|---|---|---|
| D4 KAERI non-uniform | 0.797 | **0.970** | 0.841 | 47.5% |
| D1 Nuclear Reg. Comm. | 0.926 | **0.944** | 0.242 | 47.0% |
| D7 Hardik straight | 0.636 | **0.793** | **−868.6** | 33.8% |
| D3 KAERI uniform | 0.620 | **0.735** | 0.852 | 58.9% |
| D6 Hardik helical | −0.161 | **0.687** | −1.214 | 29.7% |
| D5 Helical R-123 | 0.051 | **0.449** | **−307.7** | 14.8% |
| D8 Pioro R-134a | −0.097 | **−0.065** | **−423.6** | 16.0% |
| D2 Zhao 2020 | −0.437 | **−0.429** | −1.433 | 7.7% |
| **Median** | **0.336** | **0.711** | **−1.323** | |
| **Worst case** | −0.437 | **−0.429** | **−868.6** | |

### This is the single most important table in the project

Read it as three columns:

- **Pooled machine learning** (a strong model, honest features, no physics anchor): median **−1.323**, worst case **−868.6**. Unusable.
- **Pure physics** (Katto–Ohno, zero fitted parameters): median **0.336**. Never brilliant, never catastrophic.
- **The pipeline**: median **0.711**, worst case **−0.429**.

**Three claims follow, each directly supported:**

1. **The pipeline roughly doubles the accuracy of pure physics** — median 0.711 versus 0.336 — and **beats physics on all eight datasets**.
2. **The pipeline reduces the worst case from −868.6 to −0.429**, a factor of roughly 2,000.
3. **The pipeline cannot produce the −868 failure**, not because it happened not to, but because the bound forbids it. At zero trust the correction is limited to a factor of 1.5, so the answer can be at most 1.5× away from the correlation.

**[OUR INTERPRETATION]** Claim 3 is the one to emphasise. The first two are empirical results that could conceivably not replicate on new data. The third is a *structural guarantee* of the architecture.

## 15.5 Where the uncertainty quantification fails

**[OUR EXPERIMENT]** Source: `results/prof_tasks/pipeline_coverage.csv`. Target coverage: **90%**.

| Held-out dataset | Actual coverage |
|---|---|
| D1 Nuclear Regulatory Commission | 99.5% |
| D6 Hardik helical | 73.1% |
| D3 KAERI uniform | 65.9% |
| D7 Hardik straight | 60.0% |
| D4 KAERI non-uniform | 54.8% |
| D5 Helical R-123 | 48.6% |
| D8 Pioro R-134a | 36.9% |
| **D2 Zhao 2020** | **12.4%** |

**This is a real failure and we report it as one.**

**Diagnosis.** Conformal prediction guarantees coverage **only under exchangeability** — the assumption that calibration data and test data are drawn from the same distribution. A completely new laboratory is **not** exchangeable with the old ones. The mathematical guarantee is void in exactly the situation where we most need it.

**[OUR INTERPRETATION]** We do not have a fix. Note that within-distribution coverage (the 80/20 and 70/30 splits, Sections 15.1–15.2) is excellent — 0.92 to 0.96 against a 0.95 target. The method works exactly as advertised when its assumption holds, and fails when it does not. That is honest behaviour from the method; the problem is that our deployment case violates the assumption.

**[OPEN QUESTION]** How to produce honest uncertainty for genuinely out-of-distribution inputs. This is an open problem in the conformal-prediction literature generally, not something specific to us.

## 15.6 Physical sanity checks

**[OUR EXPERIMENT]** Source: `final_review.log`. The pipeline was probed by varying one input at a time and checking that the response has the physically correct sign.

| Check | Result | Measured sequence (kW/m²) |
|---|---|---|
| Critical Heat Flux rises with mass flux | **PASS** | 1466.3 → 2092.7 → 2903.5 → 3998.8 → 5073.9 |
| Critical Heat Flux falls as the inlet gets hotter | **PASS** | 3948.3 → 3460.5 → 2903.5 → 2279.6 → 1727.8 |
| Critical Heat Flux falls with heated length | **PASS** | 4359.1 → 2903.5 → 1798.1 → 1022.9 |

**Why this matters.** These are not accuracy tests. They test whether the model's *response surface* is physically coherent. A model can achieve a high coefficient of determination while having a locally wrong derivative — predicting, for example, that faster coolant flow makes burnout more likely. Such a model is dangerous even when it is accurate on average, because engineers use models to reason about *changes*.

**[OUR INTERPRETATION]** The pipeline passes these because the physics backbone provides the trend and the bounded correction cannot overturn it. An unconstrained model has no such guarantee. **This is a direct, testable benefit of the bounded architecture**, independent of any accuracy metric.

## 15.7 Full stress-test results

**[OUR EXPERIMENT]** Eleven automated checks, all passing:

| Check | Result |
|---|---|
| Metrics independently recomputed from raw predictions | PASS |
| Interval brackets the point prediction, all positive | PASS |
| No "not a number" or infinity in any prediction | PASS |
| Trust score within [0, 1] | PASS |
| CHF rises with mass flux | PASS |
| CHF falls as inlet gets hotter | PASS |
| CHF falls with heated length | PASS |
| Refuses impossible inputs | PASS (5 of 5) |
| Duplicate-grouped versus naive split gap measured | 0.847 vs 0.864, gap −0.017 |

---

# PART XVI — ELEVEN ATTEMPTS TO IMPROVE THE PIPELINE

## 16.1 Why this part exists

After the pipeline worked, the obvious question was whether it could be made better. We ran eleven structured variants. **None beat the baseline.** This part records them because the failures are scientifically more informative than the successes.

## 16.2 The variants

| Variant | What it changes |
|---|---|
| `weighted` | Sample weighting to counteract dataset size imbalance |
| `abs_loss` | Absolute-error loss instead of squared-error |
| `stacked` | A second-level model combining several base predictions |
| `weighted+stacked` | Both |
| `weighted+abs` | Both |
| `all_three` | All three |
| `stack_ind` | Stacking plus a missing-value indicator feature |
| `stack_noind` | Stacking without the indicator |
| `we_d` | A different weighting scheme |
| `we_d+stack_ind` | Both |
| `bo_target` | Predict the boiling number instead of the log ratio |
| `bb_ens` | Ensemble the three physics backbones |

## 16.3 Results

**[OUR EXPERIMENT]** Sources: `improve_ablation.csv`, `improve_round2.csv`. Median across held-out datasets.

| Variant | Median R² | Median ±10% |
|---|---|---|
| **baseline (shipped)** | **0.777** | **38.9%** |
| `bb_ens` | 0.777 | 38.9% ← *identical; see below* |
| `weighted` | 0.767 | 29.3% |
| `weighted+abs` | 0.754 | 32.4% |
| `all_three` | 0.747 | 27.0% |
| `stacked` | 0.738 | 27.0% |
| `we_d` | 0.731 | 30.9% |
| `stack_noind` | 0.727 | 24.1% |
| `bo_target` | 0.703 | 27.4% |
| `abs_loss` | 0.700 | 30.0% |
| `stack_ind` | 0.660 | 14.0% |
| `we_d+stack_ind` | 0.648 | 13.5% |
| *pure physics, for reference* | 0.336 | 29.2% |

**The baseline wins on both metrics. Every added sophistication made it worse.**

*(Note: the baseline median here is 0.777 rather than the 0.711 quoted in Section 15.4. The two come from different runs — the ablation used a single seed and a slightly different configuration, while Section 15.4 used five seeds. The ablation is internally consistent: all variants in the table above were run under identical conditions, so the comparison between them is valid. The 0.711 is the definitive figure for the pipeline itself.)*

## 16.4 Three failures worth studying

### Failure A — Premature enthusiasm

**[OUR EXPERIMENT]** After running only three datasets, the early signal suggested `abs_loss` and `stacked` were clear improvements. Across all eight, the baseline won on both metrics.

**[OUR INTERPRETATION]** This is precisely the trap the project exists to document — drawing a conclusion from a favourable subset. **We fell into it ourselves, during a project about not falling into it.** Recording this is more useful than pretending it did not happen.

### Failure B — A wrong diagnosis, confidently stated

When stacking failed, we predicted that adding a missing-value indicator would fix it, reasoning that the model was confused by the look-up table being unavailable on some datasets.

**Result:** `stack_ind` scored **0.660**, substantially **worse** than both the baseline (0.777) and plain stacking (0.727). On ±10% it was catastrophic: **14.0% versus 38.9%**.

**Diagnosis.** The missing-value indicator is a **fluid label in disguise**. The look-up table is unavailable precisely when the fluid is not water. So the indicator handed the model exactly the fluid-identity information that the dimensionless representation was carefully designed to remove — destroying transferability.

**[OUR INTERPRETATION]** This is a clean illustration of a general principle: *a feature that encodes "why data is missing" can leak the very grouping variable you are trying to generalise across.*

### Failure C — A silent no-op

`bb_ens` returned **byte-identical** results to the baseline. Investigation showed the look-up table arm is unavailable on four of eight datasets, so the ensemble always dropped it and reduced to the single backbone.

**[OUR INTERPRETATION]** This was caught only because the numbers were *identical* rather than merely similar. Had the variant contained a trivial source of noise, it would have produced slightly different numbers and been recorded as "a marginal change" rather than "not running at all."

## 16.5 What we conclude

**[OUR INTERPRETATION]** The pipeline appears to sit at a genuine local optimum for this corpus. The structure — physics backbone, ratio target, bounded correction, trust-scaled bound — is doing the work. Adding machine-learning machinery on top consistently *reduces* generalisation, because generalisation here is limited by **data coverage and source confounding**, not by model capacity.

> **When the bottleneck is the data, adding model complexity makes things worse, not better.**

---

# PART XVII — DIFFICULTIES AND SHORTCOMINGS

Each difficulty is given as: Problem → Evidence → Diagnosis → Attempted solution → Result → Final resolution → Remaining limitation.

## 17.1 Circularity of the standard feature set

- **Problem.** Reported accuracy came from an algebraic identity, not learned physics.
- **Evidence.** Zero-parameter energy balance: R² = 0.99977, 100% within ±10%, 77.3% within ±1% on 24,443 rows.
- **Diagnosis.** Outlet quality is algebraically linked to the target and unknowable before the test.
- **Attempted solution.** Remove quality; define an honest "reduced" formulation.
- **Result.** Ceiling fell from 0.973 to 0.743. Correct, and expected.
- **Final resolution.** All headline results use honest features. Local formulation retained only as a labelled diagnostic.
- **Remaining limitation.** The Zhao backbone still selects the quality-indexed look-up table (Section 14.3). **Disclosed, not yet fixed.**

## 17.2 Data leakage from raw-measurement columns

- **Problem.** Columns containing electrical power and geometry reconstruct the target exactly.
- **Evidence.** R² = 0.9999934, 100% within ±10% on KAERI datasets.
- **Diagnosis.** Critical Heat Flux is *derived* from power and geometry, not measured.
- **Solution.** Hard leakage whitelist enforced at load time.
- **Result.** Leak closed for all known cases.
- **Remaining limitation.** **[OPEN QUESTION]** The whitelist covers leaks we found. There may be others we have not.

## 17.3 Source-level confounding

- **Problem.** Laboratory identity, not physics, drives much of the apparent signal.
- **Evidence.** Intraclass correlation 0.815 on the surface set; Simpson reversals inverting the roughness effect in 5 of 5 studies.
- **Diagnosis.** Databases are compilations of small campaigns; rows within a campaign are not independent.
- **Solution.** Four split types including grouped and leave-one-dataset-out.
- **Result.** Revealed the true generalisation gap.
- **Remaining limitation.** We can *measure* confounding but not *remove* it. It is a property of how the field collects data.

## 17.4 The fabricated 8 mm diameter

- **Problem.** A hard-coded default diameter applied to all 1,864 Zhao rows.
- **Evidence.** The dataset names its column `D_e [mm]`; our code looked for `D_mm`, failed, and substituted silently.
- **Diagnosis.** A default with no announcement is indistinguishable from a measurement.
- **Solution.** Explicit column mapping, plus a `geom_known` flag and a warning.
- **Result.** Fixed. Corpus total corrected from 28,583 to 28,582 after also dropping a zero-mass-flux row.
- **Remaining limitation.** None for this case, but it is a warning about all silent defaults.

## 17.5 Unknown fluids returning water's properties

- **Problem.** An unrecognised fluid name silently fell back to water.
- **Evidence.** Ammonia returned exactly water's value (1960.0).
- **Diagnosis.** A permissive dictionary lookup with a default.
- **Attempted solution 1.** A downstream guard. **Failed** — the identical value reappeared.
- **Attempted solution 2.** Fix at source in `physics_arms.cp_name` — raise on unknown.
- **Result.** Fixed, and pinned by regression test 5.
- **Remaining limitation.** None known.

## 17.6 Duplicate rows inflating random splits

- **Problem.** Near-duplicate rows split across train and test inflate scores.
- **Evidence.** 578 duplicates identified; the Pioro dataset flipped sign depending on handling.
- **Solution.** Duplicate key on dimensionless features plus target; `GroupShuffleSplit`.
- **Result.** Measured gap on Zhao: −0.017.
- **Honest note.** Our hypothesis that a diameter bug created the duplicates was **wrong** — only 48 of 770 were artefacts.

## 17.7 Split mislabelling

- **Problem.** A split described as "unseen laboratory" was not always grouping on laboratory.
- **Solution.** A `GROUP_MEANING` dictionary stating explicitly what each dataset's group actually is — reference, author, coil, pressure range, or figure group.
- **Result.** Every table in this document names what the group is. Note that D5's "groups" are only 2 pressure ranges, which is why its grouped result rests on 2 folds and is weak evidence.

## 17.8 Calibration leakage

- **Problem.** The novelty reference cloud included the conformal calibration slice.
- **Diagnosis.** A calibration row's novelty was measured against a cloud containing itself — optimistic trust, over-tight intervals.
- **Solution.** Build the cloud from fit rows only.
- **Remaining limitation.** Fixed, but coverage still fails out-of-distribution for the separate reason in Section 15.5.

## 17.9 Uncertainty quantification out-of-distribution

- **Problem.** Coverage of 12.4% against a 90% target on a held-out laboratory.
- **Evidence.** Full table in Section 15.5.
- **Diagnosis.** Conformal prediction requires exchangeability, which a new laboratory violates.
- **Attempted solution.** Widening intervals as trust falls. **Insufficient.**
- **Attempted solution.** Mondrian per-band quantiles. **Worse** — a new laboratory is rated high-trust and given the *tightest* interval.
- **Final resolution.** **Unsolved.** Reported as a limitation.
- **[OPEN QUESTION]** This is the most important unsolved problem for deployment.

## 17.10 Two datasets that remain unpredictable

- **Problem.** D2 Zhao (−0.429) and D8 Pioro (−0.065) stay negative on unseen data.
- **Diagnosis for Pioro.** Digitisation noise of ±5–10%, a unique fluid, and a constant diameter with no variance. It fails on its *own* 80/20 split (−0.016), which means the data is not internally self-consistent. **This is a data limitation, not a model limitation.**
- **Diagnosis for Zhao.** **[OPEN QUESTION]** Genuinely unclear. It mixes three geometries (tubes, annuli, plates) in one dataset, and our hydraulic-diameter treatment may be inadequate for annuli and plates. **Not established.**

## 17.11 The model was never tuned

- **[ASSUMPTION]** Extra Trees with 300 trees and default settings. No hyperparameter search was performed *within the pipeline architecture*.
- **[OUR INTERPRETATION]** Given that eleven structural improvements all failed, we expect tuning to yield little. But this is an expectation, **not a measurement**.

## 17.12 Katto's constant is unverified

- **[ASSUMPTION]** `KATTO_C_SMALL` and `KATTO_C_LARGE` are tagged `[S] UNVERIFIED` in the source. They have not been checked against the 1984 paper.
- **Impact.** Propagates into most results, since Katto–Ohno is the selected backbone on seven of eight datasets.
- **Mitigation available but not run.** The `return_branches` option reports when `C` actually mattered.
- **This must be resolved before publication.**

## 17.13 No external validation

- **[OPEN QUESTION]** Every dataset was used somewhere in development. There is **no** corpus that was held back entirely and touched once.
- **[OUR INTERPRETATION]** Leave-one-dataset-out is a strong proxy, but it is not the same thing. The architecture was chosen partly *in response to* leave-one-dataset-out results, which means those results are no longer fully independent of the design.

---

# PART XVIII — NOVELTY: WHAT WE MAY CLAIM, AND WHAT WE MAY NOT

This part is deliberately conservative. Over-claiming is the fastest way to lose a reviewer.

## 18.1 What already exists in the literature — we claim none of this

**[LITERATURE]**

| Existing idea | Where it comes from |
|---|---|
| The energy-balance identity relating Critical Heat Flux to quality | Hall & Mudawar (2000b), Equation 8; and every thermal-hydraulics textbook |
| Empirical correlations for flow-boiling Critical Heat Flux | Katto & Ohno (1984), Biasi et al. (1967), Bowring (1972) |
| The Critical Heat Flux look-up table | Groeneveld and collaborators, 1995 and 2006 editions |
| Residual / deviation learning on a physics prior | Established in physics-informed machine learning |
| Dimensionless feature engineering | Standard practice, and explicitly studied in the physics-informed literature |
| Conformal prediction | Established statistics literature |
| Mahalanobis distance for novelty detection | Standard multivariate statistics |
| Look-up-table-guided grey-box networks | Published |
| Transfer learning pretrained on the look-up table | Published |
| Group-aware cross-validation | Standard machine-learning practice |

## 18.2 What we adopted unchanged

- Katto–Ohno, Biasi, and the 2006 look-up table, implemented directly as backbones.
- CoolProp / IAPWS for thermophysical properties.
- Split conformal prediction, standard form.
- Extra Trees, standard implementation.
- `GroupShuffleSplit` and `GroupKFold` from scikit-learn.

## 18.3 What we modified or combined

**[OUR PROPOSED IDEA — supported by our own measurements, novelty requires verification]**

| Element | What is different from the standard version |
|---|---|
| **Trust-scaled bound** | Residual learning normally uses either a fixed clip or none. We make the clip a **continuous function of a novelty score**, from 1.5× at zero trust to 10× at full trust. We have not found this specific mechanism in the literature we have read, but **we have not performed a systematic novelty search.** |
| **Correlation disagreement as a trust signal** | Using the spread among *independent physics correlations* as a novelty measure. It is independent of the training data — it measures uncertainty in the physics rather than in the model. **Potential novelty — requires verification.** |
| **Bounding instead of routing** | Reframing out-of-distribution handling as a continuous dial rather than a discrete switch, **justified by measured failure of the switch** (−1276 versus a bounded worst case of −0.429). |
| **Backbone selection on training rows only** | Selecting among correlations by median absolute log error, as a single discrete choice that never sees test data. |

## 18.4 The methodological finding — and exactly how far it can be pushed

This deserves its own section because it was raised as a direct question: *are we claiming the published literature is wrong, and are we certain enough to say so to a supervisor?*

### What we have actually demonstrated — this is solid

**[OUR EXPERIMENT]** On **our corpus**, with **our datasets**, under **our protocol**:

1. A zero-parameter algebraic identity reconstructs Critical Heat Flux at R² = 0.99977 with 100% of 24,443 predictions within ±10%. **Verified, reproducible, from a published equation.**
2. That identity uses a variable, outlet quality, which does not exist until after the event being predicted. **This is a matter of definition, not opinion.**
3. Removing that variable drops the achievable ceiling from 0.973 to 0.743. **Measured.**
4. Moving from a random split to a held-out source drops it from 0.743 to −3.115. **Measured.**
5. Laboratory identity alone explains 34.5% to 81.5% of variance depending on the dataset. **Measured.**

**All five of these are defensible in front of any reviewer. They are about our own data and our own experiments.**

### What we have *not* demonstrated

**[OPEN QUESTION]** We have **not** audited the published papers.

- We have **not** read the full text of most of the machine-learning-for-Critical-Heat-Flux papers and confirmed which feature set each used.
- We have **not** confirmed which used a random split versus a grouped one.
- Our own literature comparison table (Section 7.3) is explicitly flagged as assembled by **title and topic matching, pending full-text confirmation**. **It is not evidence.**

### Therefore, the precise claim that is safe to make

**Do not say:** *"The published literature is wrong."* You cannot support that, and a reviewer who has written one of those papers will ask you which papers, and you will not have an answer.

**Do say:**

> *"A reported coefficient of determination near 0.99, obtained using local outlet quality as an input under a random split, is not by itself evidence that a model has learned boiling physics — because on our corpus a zero-parameter energy-balance identity achieves 0.9998 on the same inputs, and because removing that variable and holding out an entire source drops the achievable ceiling to negative. Any study reporting such a number should state which feature set and which splitting protocol it used, so the result can be interpreted correctly."*

**Why this version is stronger, not weaker.** It is a statement about **what a metric can and cannot demonstrate**, supported entirely by our own measurements. It requires no claim about any specific paper. It is unfalsifiable by "well, my paper didn't do that" — because the response is simply "then your paper should say so, which is exactly what we are recommending."

**[OUR INTERPRETATION]** Framed this way, the finding is a **methodological recommendation**, which is a legitimate and publishable contribution, rather than an **accusation**, which would require evidence we do not have.

### What would be needed to make the stronger claim

A systematic review: obtain the full text of N published machine-learning Critical Heat Flux papers, and for each record (a) the exact feature list, (b) the splitting protocol, (c) whether source identity was controlled. Then report the proportion using local quality under random splits. **That study has not been done. It would be a genuine contribution and is a natural next step.**

## 18.5 Summary of novelty by category

| Category | Status |
|---|---|
| **Methodological novelty** | *Potential.* The demonstration that the standard feature set is algebraically circular, quantified with a zero-parameter reconstruction, plus the four-split protocol. **Requires a systematic literature audit to establish that this has not been stated before.** |
| **Dataset novelty** | *Modest.* All datasets are from public or published sources. The assembly into a single harmonised 28,582-row corpus with documented leakage controls has value, but the data is not new. |
| **Application novelty** | *Limited.* Machine learning for Critical Heat Flux is well-populated. |
| **Physics-integration novelty** | *Potential.* The trust-scaled bound and correlation-disagreement signal. **Requires verification.** |
| **Engineering novelty** | *Real but modest.* A deployable pipeline with input validation, calibrated intervals, an honest interface, and nine regression tests. |
| **Workflow novelty** | *Real.* The negative-result discipline — eleven improvement variants reported with all eleven rejected, two routing failures reported in full, four bugs documented with their evidence. |

## 18.6 The strongest single research gap

**[OUR INTERPRETATION]** If one contribution is to be emphasised, it is the tension identified in Section 13.2:

> **A novelty gate cannot detect an unseen fluid, because the dimensionless representation that makes cross-fluid transfer possible is precisely what erases fluid identity.**

**Supporting evidence:** R-134a appears in zero training rows, yet **not one of its 268 rows** is flagged as novel by either dimensionless-space distance or raw-unit distance. An oracle router would score 0.831; ours scored 0.099.

**Why this is a good research gap:** it is specific, it is falsifiable, it is supported by a measurement, and it states a genuine tension between two objectives rather than merely noting that something is difficult.

---

# PART XIX — THE FINAL PIPELINE, STEP BY STEP

## 19.1 The complete flow

```
  [ USER REQUEST ]
  fluid, pressure, mass flux, diameter, heated length, inlet temperature
        |
        v
  +--------------------------------------------------+
  |  VALIDATION                                       |
  |  reject: G=0 (pool boiling), P >= P_crit,          |
  |          D or L <= 0, unknown fluid, non-finite    |
  |  warn:   G > 10000, D outside 1-50 mm,             |
  |          fluid not in training set                 |
  +--------------------------------------------------+
        |
        v
  +--------------------------------------------------+
  |  LAYER 0 : PROPERTY ENRICHMENT                    |
  |  CoolProp -> rho_l, rho_g, sigma, h_fg, h_f,       |
  |              T_sat, P_crit                         |
  |  compute inlet subcooling enthalpy                 |
  |  compute 5 dimensionless groups                    |
  +--------------------------------------------------+
        |
        v
  +--------------------------------------------------+
  |  LAYER 1 : PHYSICS BACKBONE                       |
  |  evaluate Katto-Ohno / Biasi / 2006 LUT            |
  |  (arm was chosen at FIT time on training rows)     |
  |  -> baseline prediction, ZERO fitted parameters    |
  +--------------------------------------------------+
        |                              |
        |                              v
        |                    +----------------------+
        |                    | corr_spread =        |
        |                    | std(log of all arms) |
        |                    +----------------------+
        v                              |
  +--------------------------------------------------+
  |  LAYER 3a : TRUST                                 |
  |  s1 = Mahalanobis distance from training cloud     |
  |  s2 = corr_spread (physics disagreement)           |
  |  s3 = is the fluid unseen?                         |
  |  trust = 1 - max(s1, s2, s3)       <- pessimistic  |
  +--------------------------------------------------+
        |
        v
  +--------------------------------------------------+
  |  LAYER 2 : CORRECTION                             |
  |  ExtraTrees(300) predicts log(actual / baseline)   |
  +--------------------------------------------------+
        |
        v
  +--------------------------------------------------+
  |  LAYER 3b : BOUNDED APPLICATION                   |
  |  cap = log(1.5) + trust*(log(10) - log(1.5))       |
  |  CHF = baseline * exp( clip(correction, +/-cap) )  |
  +--------------------------------------------------+
        |
        v
  +--------------------------------------------------+
  |  LAYER 4 : CONFORMAL INTERVAL                     |
  |  width = q_global * (1 + 2*(1 - trust))            |
  |  interval = CHF * exp(+/- width)                   |
  |  label: interpolating / extrapolating /            |
  |         correlation fallback                       |
  +--------------------------------------------------+
        |
        v
  [ OUTPUT ]
  CHF, interval, confidence label, trust score,
  physics-only value, backbone name, warnings
```

## 19.2 Every stage in detail

| Stage | Input | Output | Purpose | Operation | Failure mode | Validation |
|---|---|---|---|---|---|---|
| **Validation** | Raw request | Pass, or an exception | Prevent confident wrong answers outside validity | Range and type checks | Silently accepting nonsense | Regression tests 1–4 |
| **Layer 0** | Validated request | 5 dimensionless groups | Translate to physics space | CoolProp lookups plus 5 ratio formulas | Unknown fluid; missing geometry | Regression test 5 |
| **Layer 1** | Dimensionless row | Baseline Critical Heat Flux | Provide an answer that is never insane | Closed-form correlation, zero fitted parameters | Outside correlation validity range | Grid benchmark, 157 physics runs |
| **Layer 3a** | Dimensionless row plus training statistics | Trust in [0,1] | Measure familiarity | `1 − max` of three normalised signals | **Blind to an unseen source** (Section 13.3) | Trust bounded [0,1] check |
| **Layer 2** | Dimensionless row | log correction | Learn the correlation's structured error | Extra Trees, 300 trees | Overfitting to source identity | Leave-one-dataset-out |
| **Layer 3b** | Baseline, correction, trust | Point prediction | Limit the correction by familiarity | Clip at a trust-scaled cap | None — this is the safety mechanism | Physical monotonicity checks |
| **Layer 4** | Point prediction, trust | Interval and label | Communicate uncertainty | Conformal quantile, widened by low trust | **Fails out-of-distribution** (Section 15.5) | Coverage table |

## 19.3 Deployment artefacts

| Item | Detail |
|---|---|
| Serialised model | `results/prof_tasks/chf_pipeline.pkl.gz` |
| File size | 272 MB compressed (the raw pickle is approximately 900 MB — 300 trees over 28,000 rows) |
| Compression | gzip, level 6 |
| Reference cloud | Subsampled to 5,000 rows **for the file only**; the live object is not mutated |
| Entry point | `CHFPipeline.predict_one(...)` |
| Tests | 9, in `scripts/prof_tasks/test_pipeline.py` |

---

# PART XX — THE ENTIRE SYSTEM IN PLAIN ENGLISH

*No equations. No abbreviations. This section is written so it can be read aloud.*

Imagine you are an engineer designing a cooling channel. You have a metal tube. Water will flow through it. The tube will be heated. You need to know: **how much heat can I apply before the cooling suddenly fails and the tube burns out?**

You know some things in advance. You know what fluid you are using. You know the pressure. You know how fast the fluid will flow. You know how wide the tube is and how long the heated part is. You know how cold the water is when it enters.

You do **not** know when it will burn out. That is the question.

**Step one — translate into the language of physics.**

The pipeline takes your tube and your fluid and looks up the fluid's real physical properties at your pressure: how dense the liquid is, how dense the steam is, how strongly the liquid surface holds together, how much energy it takes to boil it. It gets these from an internationally agreed reference database, the same one steam tables are built from.

Then it converts everything into five pure ratios — numbers with no units. One ratio compares how hard the flowing fluid pushes against how strongly the liquid film holds together. One compares steam density to liquid density, which tells you how close the fluid is to the point where liquid and steam become the same thing. One is simply the tube's length divided by its width. One says how much spare cooling capacity the cold incoming water is carrying. One says what fraction of the way to its critical pressure the fluid is.

**Why bother with ratios?** Because they let the pipeline compare completely different situations. Water at a hundred atmospheres in a wide pipe and a refrigerant at ten atmospheres in a narrow pipe look nothing alike — but if their ratios match, they behave the same way. This is the same trick that lets engineers test a small model aeroplane in a wind tunnel and trust the result for a full-size aircraft.

**Crucially, the pipeline never sees the name of your fluid. It only sees its properties.** That is why it can say something sensible about a fluid it has never encountered.

**Step two — ask the textbook.**

The pipeline applies a classical formula developed by heat-transfer researchers over sixty years of experiments. This formula has no adjustable numbers. It was not trained on anything. It simply encodes what physicists worked out about boiling.

The textbook answer is never brilliant. But it is never absurd either, because it comes from physics rather than from patterns in a spreadsheet.

**Step three — ask how much the textbook is usually wrong here.**

Now the machine learning enters, and its job is narrower than you might expect.

It is **not** asked "what is the answer?" It is asked: **"at conditions like these, how far off is the textbook usually, and in which direction?"**

This is a much easier question, and it has a very useful property. If the machine has nothing helpful to say — if it has never seen anything like your situation — then it says "the textbook is not wrong at all," and the pipeline simply returns the textbook answer. **A model that knows nothing gives you the classical result rather than nonsense.**

**Step four — decide how much to let it change the answer.**

Before applying the correction, the pipeline asks itself three questions:

1. *Have I seen conditions like these before?* It measures how far your situation sits from everything it was trained on.
2. *Do the classical formulas even agree with each other here?* It runs several different published correlations. If they all agree, this is well-charted territory. If they disagree, the physics itself is uncertain here.
3. *Have I ever seen this fluid?*

It takes the **most pessimistic** of those three answers and produces a confidence score between zero and one.

That score does one thing, and it is the key idea of the whole design: **it decides how far the machine learning is allowed to move the textbook answer.**

- Full confidence: the correction may change the answer by up to ten times.
- No confidence: the correction may change it by at most one and a half times.

**Think of it as a leash.** When the pipeline recognises where it is, the leash is long and the machine learning can do its job properly. When it is somewhere unfamiliar, the leash is short, and the answer stays close to the textbook.

**This is why it cannot fail catastrophically.** In an unfamiliar situation the machine learning simply is not allowed to move the answer far, so the worst possible outcome is roughly the textbook's own worst outcome. A model without this restraint, tested on data from a laboratory it had never seen, produced answers that were wrong by a factor in the hundreds. Ours cannot, by construction.

**Step five — say how sure you are, honestly.**

The pipeline sets aside a portion of its training data that it never learns from, purely to measure how wrong it actually turns out to be. From that measured record of its own mistakes, it produces a range around the answer. If confidence is low, it widens the range.

This matters because earlier in the project we tried a fashionable method that produced ranges claimed to be right ninety-five percent of the time, and they were actually right nine percent of the time. A confident wrong answer is worse than no answer at all.

**Step six — hand back everything, not just a number.**

You get the prediction, a range, a plain-language label saying whether the pipeline considers itself to be interpolating, extrapolating, or falling back on the correlation, the confidence score, **and what the classical formula alone said** — so you can see exactly how much the machine learning changed things and judge that for yourself.

**And what it refuses to do.**

If you ask it about a stationary pool of liquid with no flow, it refuses, because that is a different physical problem it was not built for. If you ask about a pressure above the point where liquid and vapour stop being distinguishable, it refuses, because burnout does not exist there. If you give it a fluid it cannot look up, it refuses, rather than quietly giving you water's answer.

**In one sentence:**

> *Let sixty years of physics give the answer, let machine learning correct it, and let the correction be limited by how much the system honestly recognises the situation.*

---

# PART XXI — CURRENT PROJECT STATUS

## 21.1 Completed and experimentally validated

| Item | Evidence |
|---|---|
| Corpus assembled: 8 flow-boiling datasets, 28,582 rows | `datasets.py`, schema table |
| Leakage audit with hard whitelist | `leakage_evidence.csv` |
| Circularity demonstrated with a zero-parameter reconstruction | `exhibitA_energy_balance.csv` |
| Source confounding quantified | `CONFOUNDING_FINDINGS.md` |
| Model benchmark: 961 runs, 22 approaches | `grid_results.csv`, `physics_results.csv` |
| Four-split protocol implemented and reported | `FINAL_results.csv` |
| Routing failure measured and diagnosed | `router_results.csv` |
| Pipeline implemented, 5 layers | `pipeline.py` |
| Leave-one-dataset-out evaluation | `pipeline_loso.csv` |
| Coverage evaluation | `pipeline_coverage.csv` |
| Eleven improvement variants, all rejected | `improve_ablation.csv`, `improve_round2.csv` |
| Physical monotonicity checks | `final_review.log` |
| Nine regression tests | `test_pipeline.py` |
| Five figures | `results/prof_tasks/figures_outline/` |
| Report document | `docs/CHF_Pipeline_Report.docx` |
| Shareable dataset package | `share/`, 31 files, plus a 1.5 MB archive |

## 21.2 Partially completed

| Item | What is done | What is not |
|---|---|---|
| Backbone honesty | 7 of 8 datasets use a quality-free backbone | D2 Zhao uses the quality-indexed look-up table; measured (0.836 → 0.650) but not yet constrained |
| Uncertainty quantification | Works in-distribution (0.92–0.96 vs 0.95 target) | Fails out-of-distribution (12.4% vs 90%) |
| Literature review | Approximately 35 papers held; core physics papers read | The comparison table is title/topic-matched, **not full-text verified** |
| Katto implementation | Implemented and used as default backbone | The constant `C` is `[S] UNVERIFIED` against the 1984 source |

## 21.3 Proposed but not implemented

**[OUR PROPOSED IDEA]** — none of these has been run. Do not present any as a result.

- Correlation-disagreement as a *routing* gate (as opposed to a trust signal, where it is already used)
- Transfer learning pretrained on the look-up table
- Recurrent modelling of axial power profiles for the non-uniform dataset
- Hyperparameter optimisation within the pipeline architecture
- Per-geometry hydraulic-diameter treatment for annuli and plates

## 21.4 Unknown

- Why the Zhao dataset resists generalisation. Mixed geometry is a hypothesis, **not established**.
- Whether the trust-scaled bound is novel. **Requires a systematic literature search.**
- Whether tuning would help. Expected to be minor, **not measured**.

## 21.5 Explicitly deferred by the supervisor

- **The "clean feature re-run"** — drop the length-to-diameter ratio and use a categorical coil label for the helical dataset, to test whether those features carry residual geometric leakage. The supervisor said *"skip that clean feature re-run for now, note it for later."* **Noted here.**

## 21.6 Recommended next steps, in priority order

1. **Verify Katto's constant `C`** against the 1984 paper. Cheap, and it affects most results.
2. **Constrain the Zhao backbone** to `phys_katto` and re-report, or disclose the difference prominently.
3. **Full-text audit of the literature table.** Required before any manuscript submission, and it is the study that would convert our methodological observation into a supported claim about the field.
4. **Obtain a genuinely external validation dataset** never touched during development.
5. **Attack the out-of-distribution coverage problem.** This is the most valuable open problem.
6. Run the deferred clean-feature re-run.

---

# PART XXII — WHAT WE KNOW VERSUS WHAT WE BELIEVE

## Established facts (measured, reproducible)

1. The energy-balance identity reconstructs Critical Heat Flux at R² = 0.99977 on 24,443 rows with zero fitted parameters, 100% within ±10%.
2. Power and geometry columns reconstruct Critical Heat Flux at R² = 0.9999934 on the KAERI datasets.
3. Removing outlet quality drops the mean best R² from 0.973 to 0.743 under an 80/20 split.
4. Moving from a random split to a held-out source drops it from 0.743 to −3.115.
5. Study identity alone explains 81.5% of variance in the surface set, 65.9% in the pin-fin set, 34.5% in the Nuclear Regulatory Commission set.
6. Simpson reversals occur in the sparse sets and not in the large one.
7. Pooled machine learning on unseen datasets: median R² −1.323, worst −868.6.
8. Pure physics on unseen datasets: median R² 0.336.
9. The pipeline on unseen datasets: median R² 0.711, worst −0.429.
10. The pipeline beats pure physics on all eight held-out datasets.
11. Out-of-distribution coverage ranges from 12.4% to 99.5% against a 90% target.
12. Eleven improvement variants were tested; none beat the baseline.
13. The theoretical R² ceiling on the look-up table from pressure and mass flux alone is 0.1538.
14. The pipeline selects Katto–Ohno on 7 of 8 datasets and the look-up table on Zhao.
15. Forcing Katto–Ohno on Zhao changes its 80/20 R² from 0.836 to 0.650.

## Direct observations

- The router scored −1276 on an unseen fluid; an oracle would score 0.831; ours scored 0.099.
- Zero of 268 R-134a rows were flagged as novel by any distance measure.
- Enabling `blend_free` dropped D7 from 0.792 to −206.9.
- The Pioro dataset fails on its own 80/20 split at −0.016.
- The three trust signals give near-identical results in isolation (0.736–0.753).

## Our interpretations (could be wrong)

- The Pioro failure is a **data** limitation, not a model limitation, because it fails on its own internal split.
- The trust signals are interchangeable because the **bounded structure**, not the gate, does the work.
- The novelty gate's blindness is **structural**, arising from the transfer/detection tension, not a tuning problem.
- Zhao resists generalisation because of its mixed geometry.
- Adding model complexity fails because the bottleneck is data coverage, not model capacity.

## Hypotheses (not yet tested)

- Hyperparameter tuning would yield only marginal gains.
- A per-geometry hydraulic-diameter treatment would improve Zhao.
- The trust-scaled bound is novel.

## Assumptions the pipeline currently makes

- Katto's `C` constant values are correct. **Unverified.**
- The five dimensionless groups are sufficient for cross-fluid transfer.
- `dh_sub = 0` is an acceptable default when inlet conditions are unreported.
- The inferred D5 coil diameters are correct.
- Extra Trees with 300 default-configured trees is an adequate correction model.
- `LtoD_dim` does not carry problematic residual leakage. **Deferred test.**

## Open questions

- How to produce honest uncertainty out-of-distribution.
- How to detect an unseen fluid without destroying transferability.
- Whether the published literature systematically over-reports. **Not established by us.**
- Why Zhao fails.
- Whether other undetected leaks exist.

---

# PART XXIII — HOW THE RESEARCH QUESTION EVOLVED

## The original question

> *Can machine learning predict Critical Heat Flux better than classical correlations?*

## What replaced it

> *What does it actually take to make a Critical Heat Flux model that is trustworthy on data it has never seen — and how much of the reported accuracy in this field measures that?*

## The chain

| Stage | Question | Answer obtained | New question raised |
|---|---|---|---|
| 0 | Can machine learning beat correlations? | Apparently yes, R² 0.987 | What is the model using? |
| 1 | What is the model using? | An algebraic identity | Does this hold on every dataset? |
| 2 | Does it hold everywhere? | No — fails on non-uniform heating, inapplicable to the look-up table | What else leaks? |
| 3 | What else leaks? | Power and geometry columns, at R² 0.99999 | Is the data independent? |
| 4 | Is the data independent? | No — study identity explains up to 81.5% | Does the split matter? |
| 5 | Does the split matter? | Enormously: 0.743 → −3.115 | Can a model be fixed by tuning? |
| 6 | Can tuning fix it? | No — every family goes negative | Can routing fix it? |
| 7 | Can routing fix it? | No — failed twice, −1276 and −206.9 | Why did routing fail? |
| 8 | Why did routing fail? | Transfer and detection are in direct tension | Can we bound instead of switch? |
| 9 | Can bounding work? | **Yes** — median 0.711, worst −0.429 | Can it be improved? |
| 10 | Can it be improved? | No — eleven variants all failed | Are the intervals trustworthy? |
| 11 | Are the intervals trustworthy? | **In-distribution yes; out-of-distribution no** | **[OPEN]** |

## Answered with evidence

- Do specific models suit specific datasets? Yes, weakly, and it tracks dataset size.
- Is one model good overall? No — but one *architecture* is.
- Does the seen/unseen router work? No, and we can say precisely why.
- Does bounding beat routing? Yes, measurably.

## Partially answered

- Can physics anchoring rescue generalisation? Yes for the point prediction; **no** for the uncertainty.

## Still open

- Honest out-of-distribution uncertainty.
- Detecting an unseen fluid without breaking transfer.
- The true state of reporting practice in the published literature.

---

# PART XXIV — REPRODUCIBILITY

## Software

| Component | Purpose |
|---|---|
| Python 3 | Language |
| NumPy, pandas | Numerical arrays and dataframes |
| scikit-learn | Extra Trees, Random Forest, Histogram Gradient Boosting, Gaussian Process Regression, Support Vector Regression, k-Nearest Neighbours, Ridge, Multi-Layer Perceptron, `GroupShuffleSplit`, `GroupKFold` |
| XGBoost, LightGBM, CatBoost | Additional gradient-boosting implementations |
| SciPy | `RegularGridInterpolator` for look-up-table interpolation |
| **CoolProp** | Thermophysical properties (IAPWS for water) |
| matplotlib (Agg backend) | Figures |
| python-docx | Report generation |
| pytest | Regression tests |

## Configuration

| Setting | Value |
|---|---|
| Correction model | `ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=seed)` |
| Target | `log(CHF / backbone)` |
| Features | `["Katto_We_inv", "rho_ratio_gl", "LtoD_dim", "dh_sub_star", "P_red"]` |
| Conformal calibration fraction | 20%, minimum 30 rows |
| `alpha` (1 − target coverage) | 0.10 |
| `BOUND_MIN`, `BOUND_MAX` | 1.5, 10.0 |
| `TRUST_BANDS` | (0.33, 0.66) |
| Mahalanobis normalisation | 50th and 95th percentiles of training distances |
| `blend_free` | `False` |
| `conformal` | `'global'` |
| Seeds | 10 for random splits, 5 for leave-one-dataset-out |

## Code map

```
scripts/prof_tasks/
    datasets.py        loaders, leakage whitelist, group definitions
    physics_arms.py    CoolProp properties, dimensionless groups, 3 correlations
    pipeline.py        the CHFPipeline class -- the deliverable
    final_review.py    definitive evaluation, 4 splits, 11 stress checks
    test_pipeline.py   9 regression tests
    run_grid.py        the 804-run machine-learning benchmark
    analyze.py         answers to supervisor Questions 1 and 2
    router.py          the failed routing experiment (retained deliberately)
    improve_ablation.py, improve_round2.py   the 11 rejected variants
    outline_figures.py, make_figures.py      figures
    make_report_docx.py                      report generation
    verify_leakage.py                        the leakage audit

physics_pipeline/scripts/physics/correlations.py   katto_ohno_chf, katto_C
scripts/chf_physics.py                             biasi_chf, zuber_pool_boiling_chf
```

## Reproducibility checklist

| Item | Status |
|---|---|
| Source code | **Available** |
| Datasets | **Available** — `share/`, 31 files |
| Random seeds | **Available** — fixed and recorded |
| Model hyperparameters | **Available** |
| Feature definitions | **Available** |
| Physical assumptions | **Available** — documented in source |
| Evaluation procedure | **Available** |
| Result files | **Available** |
| Trained model | **Available** — `chf_pipeline.pkl.gz`, 272 MB |
| Regression tests | **Available** |
| Katto's constant `C` | **MISSING** — value not verified against source |
| Exact library versions | **MISSING** — no lock file or environment specification |
| D5 coil diameters | **PARTIAL** — inferred by inversion, not published |
| Literature table numbers | **MISSING** — title/topic matched, not full-text verified |
| External validation set | **MISSING** — none exists |

---

# PART XXV — DECISION LOG

| # | Decision | Why | Evidence | Alternative considered | Outcome |
|---|---|---|---|---|---|
| 1 | Start with the Nuclear Regulatory Commission database | Largest, standard, public | 24,443 rows | Start with small clean sets | Correct — size enabled statistical claims |
| 2 | Use the standard published feature set | It is what the literature uses | — | Inlet-only from the start | **Revised at decision 4** |
| 3 | Test the energy-balance identity | Feature importance looked suspicious | R² 0.99977, 0 parameters | Accept the 0.987 and publish | **The pivotal decision of the project** |
| 4 | Remove outlet quality from all headline results | It is unknowable before the test | Ceiling fell 0.973 → 0.743 | Keep it and caveat | Correct — makes the work defensible |
| 5 | Audit every column for leakage | Circularity implied more might exist | R² 0.9999934 on KAERI | Trust column names | Correct — found 4 leaks |
| 6 | Hard leakage whitelist at load time | Manual exclusion is forgettable | — | Document and remember | Correct — protection is automatic |
| 7 | Add grouped and leave-one-dataset-out splits | Data is a compilation, not a sample | 0.743 → −3.115 | Random splits only | **The second pivotal decision** |
| 8 | Duplicate-aware splitting | Near-duplicates inflate scores | Gap −0.017 on Zhao | Ignore duplicates | Correct, though the effect was smaller than predicted |
| 9 | Quantify source confounding | Needed to know *why* splits matter | Intraclass correlation 0.815 | Assert it qualitatively | Correct — gave a dose-response relationship |
| 10 | Benchmark 22 approaches, 961 runs | Answer supervisor Questions 1 and 2 with data | Full grid | Test 3 or 4 models | Correct |
| 11 | Build the seen/unseen router | Directly requested | — | Decline | Failed, **and the failure was informative** |
| 12 | Try soft blending after the router failed | A dial seemed safer than a switch | D7: 0.792 → −206.9 | Abandon immediately | Failed, and taught us trust ≠ source novelty |
| 13 | **Bound instead of route** | Switch errors cost 1000×; bounds degrade smoothly | Worst case −868.6 → −0.429 | Keep trying to fix the router | **The core design decision** |
| 14 | Physics backbone with zero fitted parameters | Need a floor that cannot be absurd | Physics median 0.336 vs pooled ML −1.323 | Learned baseline | Correct |
| 15 | Train on the log ratio, not on Critical Heat Flux | Well-behaved target; degrades to physics | — | Direct prediction | Correct — gives the safety property |
| 16 | Pooled correction, not per-dataset | Per-dataset routing failed twice | — | One model per dataset | Correct |
| 17 | Trust scales the bound, never selects a model | Bounds have measured monotonic trade-offs | `BOUND_*` sweep | Trust as a router | Correct |
| 18 | Conformal intervals | Deep ensembles gave 9–22% coverage at nominal 95% | — | Deep ensembles | Correct in-distribution; insufficient out |
| 19 | Global conformal, not per-band | Per-band gives a new lab the tightest interval | Coverage table | Mondrian bands | Correct |
| 20 | Build the novelty cloud from fit rows only | Calibration rows were being compared to themselves | — | All training rows | Correct — removed a real leak |
| 21 | Exclude pool boiling entirely | Supervisor directive | — | Include with a Zuber backbone | Followed |
| 22 | Raise, not substitute, on unknown fluid | Ammonia silently returned water's value | Identical value 1960.0 | Fall back to water | Correct |
| 23 | Run eleven improvement variants | Do not stop at the first thing that works | All eleven lost | Ship the first version | Correct — confirmed a local optimum |
| 24 | Report all failures in full | Negative results explain the design | — | Report only the final method | Correct |
| 25 | Defer the clean-feature re-run | Supervisor directive | — | Run it now | Deferred, recorded |

---

# PART XXVI — GLOSSARY

**Annulus.** A flow channel between two concentric tubes; the fluid flows in the ring-shaped gap.

**Boiling number.** A dimensionless group: applied heat flux divided by the maximum flux the flowing mass could absorb if fully vaporised. `Bo = CHF / (G · h_fg)`.

**Burnout.** Colloquial term for the boiling crisis, from the fact that the surface can physically burn out.

**CatBoost.** A gradient-boosting implementation using ordered boosting to reduce a subtle form of overfitting.

**Coefficient of determination (R²).** A goodness-of-fit measure. 1.0 is perfect. 0.0 means no better than always predicting the average of the data. **Negative means worse than always predicting the average** — the model is actively harmful.

**Conformal prediction.** A method for producing prediction intervals with a guaranteed coverage rate, requiring no assumption about the error distribution. It works by measuring the model's actual errors on held-out calibration data.

**CoolProp.** An open-source library of thermophysical properties, implementing internationally accepted equations of state.

**Coverage.** The fraction of times a prediction interval actually contains the true value. If a "90% interval" achieves 90% coverage, it is well calibrated.

**Critical Heat Flux (CHF).** The maximum heat flux at which efficient nucleate boiling can be sustained. Above it, a vapour film forms and cooling collapses.

**Critical point / critical pressure.** The pressure and temperature above which liquid and vapour become indistinguishable. Boiling does not exist there.

**Departure from Nucleate Boiling (DNB).** The boiling crisis occurring in subcooled or low-quality flow, by liquid-sublayer dryout under a vapour blanket. The pressurised-water-reactor failure mode.

**Digitising.** Extracting numerical data from a published graph image by calibrating the axes and clicking on plotted points. Introduces error of roughly ±5–10%.

**Dimensionless group.** A ratio of physical quantities whose units cancel, expressing a balance between competing effects. Allows physically different systems to be compared.

**Dryout.** The boiling crisis occurring in high-quality annular flow, by depletion of the liquid film on the wall. The boiling-water-reactor failure mode.

**Enthalpy.** The energy content of a fluid per unit mass, including the work done by pressure. Measured in joules per kilogram.

**Exchangeability.** The statistical assumption that data points are drawn from the same distribution and their order does not matter. **Conformal prediction's guarantee requires it, and a new laboratory violates it.**

**Extra Trees (Extremely Randomised Trees).** A tree ensemble in which both the features considered and the split thresholds are chosen randomly. **The correction model in our pipeline.**

**Feature.** One input variable to a model.

**Gaussian Process Regression (GPR).** A method placing a probability distribution over functions and conditioning it on observed data. Returns a prediction and an uncertainty. Costs time proportional to the cube of the number of training points.

**Gradient boosting.** Building models sequentially, each correcting the errors of the ensemble so far.

**GroupShuffleSplit / GroupKFold.** Splitting methods that keep all rows sharing a group label on the same side of the split.

**Hydraulic diameter.** An equivalent diameter for a non-circular channel: four times the cross-sectional area divided by the wetted perimeter. Equals the actual diameter for a circular tube.

**Intraclass Correlation Coefficient (ICC).** The fraction of total variance explained by group membership alone.

**Latent heat of vaporisation (h_fg).** Energy required to convert one kilogram of saturated liquid into vapour at constant temperature.

**Leakage.** When information about the answer reaches the model through its inputs, producing an inflated and meaningless evaluation.

**LightGBM.** A fast, histogram-based gradient-boosting implementation.

**Look-Up Table (LUT).** A tabulation of Critical Heat Flux indexed by pressure, mass flux, and quality. The Groeneveld 2006 edition is the international standard.

**LODO (Leave-One-Dataset-Out).** Train on all datasets but one; test on the excluded one entirely.

**LOSO (Leave-One-Source-Out).** The same, at the level of individual source publications.

**Mahalanobis distance.** A distance measure accounting for the shape and correlation structure of a data cloud, rather than treating it as a sphere.

**MAPE (Mean Absolute Percentage Error).** The average absolute error expressed as a percentage of the true value.

**Mass flux (G).** Mass flow rate per unit cross-sectional area, in kilograms per square metre per second.

**Mixture of Experts.** An architecture with several specialist models and a gate routing each input to one. **Attempted here; failed.**

**Mondrian conformal prediction.** Computing a separate conformal quantile for each of several defined groups.

**Multi-Layer Perceptron (MLP).** A feed-forward neural network of stacked layers, each a weighted sum followed by a non-linear function.

**Nucleate boiling.** Efficient boiling in which discrete bubbles form at surface sites, detach, and stir the liquid. The desired operating regime.

**Pool boiling.** Boiling in stationary liquid with no forced flow. **Explicitly out of scope here.**

**Quality (x).** The mass fraction of the flowing fluid that is vapour. Negative values indicate subcooled liquid by convention.
- **Inlet quality** — at the channel entrance. **Known in advance.**
- **Outlet / local quality** — at the burnout point. **NOT known in advance. This is the circularity problem.**

**Random Forest.** A tree ensemble using random row and feature subsets, with predictions averaged.

**Reduced pressure.** Operating pressure divided by the fluid's critical pressure. The key variable for comparing different fluids.

**Residual / deviation learning.** Training a model to predict the *error* of a baseline rather than the target itself.

**RMSE (Root-Mean-Square Error).** The square root of the average squared error, in the original units. Penalises large errors heavily.

**Saturation.** The state at which liquid and vapour coexist in equilibrium. Properties at saturation are determined entirely by pressure.

**Seed.** The initialiser of a random-number generator. Varying it and reporting the spread shows whether a result is stable.

**Simpson reversal.** When a relationship in pooled data points opposite to the relationship within every subgroup.

**Split conformal prediction.** The specific conformal variant used here: hold out a calibration set, measure errors on it, use their quantile as the interval width.

**Subcooling.** How far below its boiling point a liquid is. Expressed here as an enthalpy deficit in joules per kilogram.

**Support Vector Regression (SVR).** A method fitting a function that stays within a tolerance band of the data while remaining as flat as possible.

**Surface tension (σ).** The force per unit length resisting the breaking of a liquid surface. Falls to zero at the critical point.

**Trust score.** Our composite novelty measure in [0, 1]. **It scales the bound on the correction; it never selects a model.**

**Weber number.** The ratio of inertial force to surface-tension force. We use its inverse.

**XGBoost (Extreme Gradient Boosting).** A widely used, high-performance gradient-boosting implementation.

**Zuber correlation.** A 1959 pool-boiling Critical Heat Flux expression derived from hydrodynamic instability. Pressure-only; **out of scope** for flow boiling.

---

# PART XXVII — SCIENTIST REFERENCE TABLE

| Scientist | Field & era | What they did | Key principle | Why it matters to us | How we used it |
|---|---|---|---|---|---|
| **Yoshiro Katto** (University of Tokyo) | Two-phase flow and boiling heat transfer, 1960s–1980s | Developed a generalised flow-boiling Critical Heat Flux correlation from dimensionless groups, valid across fluids rather than fitted to water alone | Identify the competing failure mechanisms, write an expression for each, and take the **minimum** — a system fails by its weakest route | It is the only one of our three correlations that needs **no vapour quality**, so it is honest by our standard | **Directly implemented** as `phys_katto`. **Selected as the pipeline backbone on 7 of 8 datasets.** Its constant `C` is `[S] UNVERIFIED`. |
| **Hiroshi Ohno** | Heat transfer, 1980s | Co-developed the 1984 refinement of the Katto correlation, including the density-ratio regime gate at 0.15 | Different density regimes are governed by different mechanisms, and the boundary is physical rather than tuned | The regime gate is why the correlation spans water and refrigerants | Implemented as the `dr < 0.15` branch selection |
| **L. Biasi and colleagues** (CISE, Italy) | Reactor thermal-hydraulics, 1960s | Fitted a two-branch empirical correlation for water in vertical tubes | Two algebraic expressions — one low-quality, one high-quality — cover the practical range; take the larger | Provides a second independent physics opinion, which feeds our **correlation-disagreement** trust signal | **Directly implemented** as `phys_biasi`. Water-only; requires quality. A real bug was fixed in its per-diameter exponent selection. |
| **Derek C. Groeneveld** (AECL / University of Ottawa) | Reactor thermal-hydraulics, 1980s–present | Led the international Critical Heat Flux Look-Up Table, consolidating over 30,000 experimental points into a standardised table | Stop arguing about formulas; pool all the data, smooth it, and publish an international reference | It is our reference standard, one of our three backbones, and the object whose structure explains why quality is unavoidable *there* but circular *for prediction* | **Directly implemented** as `phys_lut` via trilinear interpolation. Also the subject of the ceiling analysis in Section 4.7. |
| **John Hall** and **Issam Mudawar** (Purdue University) | Boiling and two-phase cooling, 2000 | Assembled a consolidated water Critical Heat Flux database and systematically assessed existing correlations against it | Before proposing a new correlation, assemble the data honestly and evaluate everything on equal terms | **Their Equation 8 is the energy-balance identity on which our entire methodological finding rests** | Used **not as a predictor but as a diagnostic instrument** — to prove the standard feature set is circular. **The equation is theirs, not ours.** |
| **Novak Zuber** | Thermal-hydraulics, 1959 | Derived a pool-boiling Critical Heat Flux expression from a hydrodynamic (Helmholtz) instability argument | The boiling crisis occurs when the counterflow of rising vapour and falling liquid becomes unstable | Embedded in the look-up table we use — its authors used Zuber to build the zero-flow skeleton and extrapolate to 21 MPa | **Not used directly.** Pool boiling is explicitly out of scope; the pipeline **refuses** zero-mass-flux requests and names Zuber in the error message. |
| **Weisman & Pei; Lee & Mudawar** | Mechanistic boiling models | Formulated the liquid-sublayer dryout model for Departure from Nucleate Boiling | Burnout occurs when a vapour blanket prevents turbulent core liquid from replenishing a thin sublayer | Explains the *physical mechanism* our correlations describe empirically | **Conceptual only.** Not implemented — these models need empirical closure constants recalibrated per regime. |
| **Hewitt & Govan** | Mechanistic boiling models | Developed three-fluid film-dryout models balancing film flow, droplet entrainment, and deposition | Burnout in annular flow occurs when the liquid film is depleted faster than droplets replenish it | Explains the high-quality dryout mechanism, and why the inverse Weber number is the right group | **Conceptual only.** Not implemented. |

---

# PART XXVIII — EQUATION REFERENCE TABLE

| # | Equation | Name | Variables | Physical meaning | Where used |
|---|---|---|---|---|---|
| 1 | `q_CHF = G·D·(x·h_fg + ΔH_in,sub) / (4L)` | Energy balance (Hall & Mudawar 2000b Eq. 8) | q_CHF W/m²; G kg/(m²s); D m; L m; x —; h_fg J/kg; ΔH_in,sub J/kg | Heat in equals energy carried away. **Exact, zero parameters.** | **Diagnostic only.** Proved the circularity (R² 0.99977, 100% within ±10%). Never used to predict. |
| 2 | `q_CHF = q_co · (1 + K·Δh_sub/h_fg)` | Katto–Ohno (1984) | q_co W/m² saturated-inlet value; K — subcooling coefficient | Compute the saturated-inlet value, then boost it for cold inlet | **The pipeline's default backbone** (7 of 8 datasets) |
| 3 | `q_co = min(q2,q3,q4)` if `ρg/ρl<0.15` else `min(q2,q5,q13)` | Katto regime gate | Five candidate expressions | Failure occurs by whichever mechanism is weakest | Inside the Katto backbone |
| 4 | `q = max(q_low, q_high)` | Biasi (1967) | P atm; G g/(cm²s); D cm; X — | Two branches for low and high quality; take the larger | Backbone candidate; water only |
| 5 | `CHF = LUT(P, G, x)` | Groeneveld 2006 | P kPa; G kg/(m²s); x — | Trilinear interpolation of the international standard table | Backbone candidate; **selected on Zhao** |
| 6 | `Katto_We_inv = σ·ρ_l/(G²·L)` | Inverse Weber number | σ N/m; ρ_l kg/m³; G kg/(m²s); L m | Surface tension holding the film together versus inertia tearing it apart | Model feature 1 |
| 7 | `rho_ratio_gl = ρ_g/ρ_l` | Density ratio | Both kg/m³ | Distance from the critical point; how much vapour expands on boiling | Model feature 2 — **the main carrier of fluid transferability** |
| 8 | `LtoD_dim = L/D` | Length-to-diameter ratio | Both m | Channel shape; how much vapour can accumulate | Model feature 3 |
| 9 | `dh_sub_star = Δh_sub/h_fg` | Dimensionless subcooling | Both J/kg | Stored cooling capacity versus the cost of boiling | Model feature 4 |
| 10 | `P_red = P/P_crit` | Reduced pressure | Both Pa | Corresponding states — the key to cross-fluid comparison | Model feature 5 |
| 11 | `Bo = CHF/(G·h_fg)` | Boiling number | CHF W/m²; G kg/(m²s); h_fg J/kg | Applied flux versus maximum absorbable flux | Target of `PIML_dimensionless` and `bo_target` |
| 12 | `d = sqrt((x−μ)ᵀ Σ⁻¹ (x−μ))` | Mahalanobis distance | x point; μ median; Σ covariance | Distance accounting for the cloud's shape | Trust signal 1 |
| 13 | `corr_spread = std(log of all arms)` | Correlation disagreement | — | When published correlations disagree, the physics is uncertain here | Trust signal 2 — **potentially novel** |
| 14 | `trust = 1 − max(s1, s2, s3)` | Trust score | Three normalised signals | Deliberately pessimistic: any one alarm lowers trust | Layer 3a |
| 15 | `cap = log(1.5) + trust·(log(10) − log(1.5))` | **Trust-scaled bound** | trust in [0,1] | **How far machine learning may move the physics: 1.5× when unfamiliar, 10× when familiar** | **Layer 3b — the core of the design** |
| 16 | `CHF = base·exp(clip(correction, ±cap))` | Bounded correction | base from Layer 1 | Physics answer, adjusted within a leash | Layer 3b |
| 17 | `width = q_global·(1 + 2(1 − trust))` | Trust-widened conformal interval | q_global = the (1−α) quantile of calibration errors | Less confidence means a wider stated range | Layer 4 |

---

# PART XXIX — MASTER EXPERIMENT TABLE

| # | Stage | Objective | Dataset | Approach | Physics? | Key result | Interpretation | Decision |
|---|---|---|---|---|---|---|---|---|
| 1 | Baseline | Can machine learning beat correlations? | D1 | 13 families, local features, 80/20 | No | **R² 0.987**, 78.1% within ±10% | Looked excellent | Investigate what the model uses |
| 2 | **Circularity** | Is the accuracy real? | D1 | Energy balance, **0 parameters** | Pure | **R² 0.99977**, **100%** within ±10%, 77.3% within ±1% | **The identity beats the trained model** | **Remove quality from all headlines** |
| 3 | Circularity scope | Does it hold everywhere? | D1,D3,D4,D9 | Energy balance per dataset | Pure | Holds on D1, D3; **fails** on D4 (non-uniform); **inapplicable** to D9 | Real but not universal | Report the scope honestly |
| 4 | Look-up-table ceiling | Why does the table need quality? | D9 | Oracle analysis | — | **Ceiling from (P,G) alone = 0.1538**; with quality = 1.000; CHF spans ~200× per cell | **−0.003 is not a model failure** | Explain the table's structure |
| 5 | **Leakage audit** | Do columns contain the answer? | D3,D4,D6 | Direct reconstruction | — | **R² 0.9999934**, **100%** within ±10% | Power and geometry are the answer | **Hard whitelist at load time** |
| 6 | Quality semantics | Is D4's quality inlet or outlet? | D4 | Correlation with reconstruction | — | **r = 0.999999** on all 888 rows — it is **inlet** | An earlier claim was **wrong** | Rename to `X_inlet`, drop the local form |
| 7 | **Confounding** | Is the data independent? | 3 families | Intraclass correlation, Simpson | — | ICC **0.815 / 0.659 / 0.345**; roughness reverses sign in **5 of 5** studies | Laboratory identity dominates sparse sets | **Add grouped and LODO splits** |
| 8 | **Model grid** | Which model for which data? | 10 | 22 approaches, 961 runs | Both | Best per dataset varies; gap 0.973 → 0.743 → **−3.115** | Validation matters more than model | Physics anchoring is required |
| 9 | **Router** | Does seen/unseen routing work? | All | Hard gate | Both | **R² −1276**; oracle 0.831, ours **0.099**; **0 of 268** R-134a rows flagged | Transfer and detection are in tension | **Abandon routing** |
| 10 | Blending | Does soft blending work? | D7 held out | Trust-weighted blend | Both | **0.792 → −206.9** | Trust ≠ source novelty | `blend_free=False`, documented in source |
| 11 | Bound sweep | How tight should the bound be? | All | 1.5×, 2×, 3×, unbounded | Both | Monotonic trade-off measured | A dial, not a switch | **Adopt the trust-scaled bound** |
| 12 | **Pipeline LODO** | Does the architecture generalise? | All 8 | Pipeline vs physics vs pooled ML | Both | Median **0.711** vs 0.336 vs **−1.323**; worst **−0.429** vs **−868.6** | **The headline result** | Ship it |
| 13 | Random splits | Supervisor's requirement | All 8 | 80/20 and 70/30, 10 seeds | Both | D1 **0.994 / 96.2%**; agreement to ~0.01 between ratios | Not a split-ratio artefact | Report both |
| 14 | Grouped splits | Unseen laboratory | 7 | Grouped hold-out | Both | D1 mean 0.976, **worst 0.953**; D3 mean **−8.401**, worst **−73.5** | Large sets survive; small ones collapse | Report mean, median **and worst** |
| 15 | Coverage | Are the intervals honest? | All 8 | Conformal, 90% target | Both | **12.4% to 99.5%** | **Fails out-of-distribution** | Report as a limitation |
| 16 | Monotonicity | Is the response surface physical? | Synthetic | One-at-a-time sweeps | Both | All 3 **PASS** | Bounded structure preserves physical trends | Adopt as a standing check |
| 17 | Improvements R1 | Can it be improved? | All 8 | 6 variants | Both | **Baseline wins**, 0.777 median | Early 3-dataset signal was misleading | Reject all |
| 18 | Improvements R2 | Second attempt | All 8 | 5 more variants | Both | **Baseline still wins**; `stack_ind` **0.660** | The indicator is a fluid label in disguise | Reject all |
| 19 | No-op detection | Why is `bb_ens` identical? | All 8 | Investigation | — | Look-up table is unavailable on 4 of 8; arm always dropped | **A silent no-op** | Document |
| 20 | **Backbone audit** | Does the backbone use quality? | All 8 | Direct inspection | — | **7 of 8 use Katto (no quality); Zhao uses the look-up table (uses quality)**; forcing Katto on Zhao: **0.836 → 0.650** | An honesty caveat on one dataset | **Disclose** |
| 21 | Independent check | Is D1's 0.994 reproducible? | D1 | 3 fresh seeds | Both | **0.9932 / 0.9951 / 0.9932**; physics-only 0.928–0.930 | Confirmed; features contain **no quality** | Confirmed for publication |
| 22 | Stress tests | Is it deployable? | — | 11 checks, 9 regression tests | — | **All PASS** | Deployable | Ship |

---

# PART XXX — ONE-PAGE SUMMARY

**What problem are we solving?**
Predicting Critical Heat Flux — the heat load at which boiling suddenly stops cooling a surface and its temperature jumps hundreds of degrees in under a second. It sets the power limit of every water-cooled nuclear reactor. Under-predict it and you waste generating capacity for sixty years; over-predict it and you melt a core.

**Why is it difficult?**
No closed-form theory exists. The failure mechanism itself changes between regimes. Experiments are destructive, so data accumulates slowly in small campaigns at different laboratories over decades.

**What did we initially try?**
Standard machine learning on the standard published feature set: pressure, mass flux, quality, diameter, heated length, subcooling. Random 80/20 split. We obtained **R² = 0.987**.

**Why did that not work?**
Three discoveries, in order:
1. **Circularity.** A zero-parameter energy-balance identity — Hall & Mudawar (2000b) Equation 8 — reconstructs the target at **R² = 0.99977** with **100% of 24,443 predictions within ±10%**. It uses outlet quality, which does not exist until after burnout. The model was re-deriving arithmetic. Removing quality drops the ceiling to **0.743**.
2. **Leakage.** Power and geometry columns reconstruct the target at **R² = 0.9999934**.
3. **Confounding.** These databases are compilations. Laboratory identity alone explains **34.5% to 81.5%** of variance. Holding out a source drops the ceiling to **−3.115**.

**What did we learn?**
A high coefficient of determination on a random split, using local quality, is not evidence of learned physics. **Choice of validation matters more than choice of model.**

**Why did we introduce physics?**
Because no model fixes −3.115. Pooled machine learning on an unseen dataset reached **R² = −868.6**. The model needed a floor that cannot be absurd.

**What literature did we investigate?**
Classical correlations (Katto–Ohno 1984, Biasi 1967), the Groeneveld 2006 Look-Up Table, Hall & Mudawar's database work, and the physics-informed and transfer-learning literature. The most useful single finding: **residual learning on a physics prior diverges out-of-domain** — which is precisely what our bound prevents.

**What datasets did we use?**
Eight flow-boiling datasets, **28,582 points**, three fluids, five geometries, plus the 11,088-entry look-up table as a reference. Pool boiling excluded by direction.

**What models did we try?**
**22 approaches over 961 runs** — thirteen machine-learning families and nine physics and hybrid approaches — across ten datasets, three formulations, and four split types.

**What finally worked?**
A five-layer pipeline: dimensionless physics features → a zero-parameter correlation backbone → an Extra Trees model predicting the *log ratio* of measurement to correlation → a trust score that **scales the bound on that correction** (1.5× when unfamiliar, 10× when familiar) → conformal prediction intervals.

**The result, on completely unseen datasets:**

| | Pure physics | **Pipeline** | Pooled ML |
|---|---|---|---|
| Median R² | 0.336 | **0.711** | −1.323 |
| Worst case | −0.437 | **−0.429** | **−868.6** |

The pipeline beats physics on **all eight** held-out datasets and reduces the worst case by a factor of roughly 2,000. On its own data it reaches **R² 0.994 with 96.2% of predictions within ±10%** (80/20) and **0.993 / 95.8%** (70/30).

**What is potentially novel?**
The trust-scaled bound; correlation disagreement as a trust signal; and reframing out-of-distribution handling as a **dial rather than a switch**, justified by the measured failure of the switch. **All flagged as requiring a systematic novelty search.** The methodological demonstration of circularity is our strongest contribution.

**What are the biggest limitations?**
1. **Uncertainty fails out-of-distribution** — 12.4% actual coverage against a 90% target on one held-out laboratory.
2. **Two datasets stay negative**: Zhao (−0.429, cause unknown) and Pioro (−0.065, digitisation noise).
3. **No external validation set exists.**
4. **Katto's constant `C` is unverified** against its source.
5. **One backbone selection (Zhao) uses quality**, inflating that dataset's number by ~0.19.
6. **We have not audited the published literature** and therefore cannot claim it systematically over-reports.

**What remains to be done?**
Verify Katto's constant. Constrain or disclose the Zhao backbone. Perform the full-text literature audit. Obtain a genuinely external validation set. Attack the out-of-distribution coverage problem — which, together with the transfer/detection tension, is the most valuable open research gap this project has identified:

> **A novelty gate cannot detect an unseen fluid, because the dimensionless representation that makes cross-fluid transfer possible is exactly what erases fluid identity.** R-134a appears in zero training rows, yet not one of its 268 rows is flagged as novel by any distance measure.

---

*End of document.*

*Every number labelled **[OUR EXPERIMENT]** is traceable to a named file in the project repository. Every claim labelled **[LITERATURE — UNVERIFIED]**, **[ASSUMPTION]**, **[OUR PROPOSED IDEA]**, or **[OPEN QUESTION]** is explicitly not established and must not be presented as a result.*
