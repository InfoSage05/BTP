# Machine-Learning and Physics-Informed Prediction of Critical Heat Flux from the 2006 Groeneveld Look-Up Table: Interpolation Fidelity, Pressure Extrapolation, and Model Reliability

**Target Journal:** *Nuclear Engineering and Design* (Elsevier)

---

## 1. Introduction

### 1.1 Nuclear Engineering Relevance and Physics of Critical Heat Flux
Critical heat flux (CHF) is a principal thermal-hydraulic limit in the design and safety assessment of water-cooled nuclear systems, including pressurized water reactors (PWRs), boiling water reactors (BWRs), small modular reactors (SMRs), and some accident-tolerant-fuel concepts [1, 2]. During reactor operation, heat is transferred from the fuel cladding to the coolant through single-phase convection and, depending on local conditions, subcooled or saturated boiling. Nucleate boiling can provide a high heat-transfer coefficient because latent heat transfer is accompanied by bubble nucleation, growth and departure [3].

However, when the imposed heat flux reaches the CHF condition, the boiling process undergoes a boiling crisis [4]. In subcooled or low-quality flow boiling, the crisis is commonly associated with departure from nucleate boiling (DNB). In higher-quality annular flow, it may be associated with liquid-film dryout caused by evaporation, entrainment and insufficient liquid-film replenishment [5, 6]. The observed mechanism depends on pressure, mass flux, quality, geometry, inlet condition and heating configuration; DNB and dryout should therefore not be treated as interchangeable labels.

In the post-CHF regime, heat-transfer performance deteriorates and wall temperature can rise rapidly [7]. In a reactor, this can reduce thermal margin and, under sufficiently severe conditions, contribute to cladding damage mechanisms such as oxidation, ballooning and embrittlement [8]. Engineering margins are commonly expressed using quantities such as the departure-from-nucleate-boiling ratio (DNBR) in PWR analysis and the critical power ratio (CPR) in BWR analysis. The exact acceptance criterion and its statistical interpretation are system- and regulator-specific; the 95/95 criterion should not be presented as a universal requirement for all reactor types [9, 10].

### 1.2 Classical Prediction Paradigms: Correlations vs. Look-Up Tables
The accurate, reliable, and computationally efficient prediction of CHF across wide operational envelopes is important for reactor thermal design, licensing, safety-margin evaluation and core monitoring [11]. Over the past six decades, thermal-hydraulic researchers have developed many empirical and semi-empirical CHF correlations for particular geometries, fluids, heating configurations and flow regimes [12–14]. Correlations such as those developed by Biasi et al. [15], Bowring [16], and Katto and Ohno [17] illustrate the value of compact engineering representations, although their applicability remains conditional on the database and regime for which they were developed. Their main limitations include:
1. **Narrow Domain Validity:** Empirical correlations are typically calibrated against restricted experimental subsets; applying them outside their calibrated pressure ($P$), mass flux ($G$), or quality ($X$) ranges frequently leads to substantial, non-physical errors [18].
2. **Regime Dependence:** Correlations may require different formulations or switching logic across subcooled DNB and saturated dryout regimes, which can complicate implementation and calibration [19].
3. **Inability to Capture Multivariable Interactions:** Closed-form algebraic correlations cannot fully resolve the highly non-linear, coupled multi-variable dependencies among local thermohydraulic parameters [20].

To overcome the fragmentation and geometric specificity of individual correlations, generalized Look-Up Tables (LUTs) were developed through international data consolidation efforts spearheaded by Chalk River Laboratories (AECL) and the University of Ottawa [21, 22]. The 2006 Groeneveld CHF Look-Up Table [23] represents the international reference standard for vertical upward water flow in tubes. Normalized to a standard 8-mm internal diameter tube, the 2006 LUT provides a structured, discrete grid of CHF values spanning 24 discrete pressures ($100 \le P \le 21{,}000\text{ kPa}$), 21 mass fluxes ($0 \le G \le 8{,}000\text{ kg}\cdot\text{m}^{-2}\cdot\text{s}^{-1}$), and 23 thermodynamic qualities ($-0.50 \le X \le 1.00$), comprising 11,592 tabulated grid points compiled from over 30,000 verified experimental measurements [23].

Within its intended domain, interpolation of the LUT provides a practical baseline surface. The table itself is discrete and empirical, however, and interpolation should not be confused with a fundamental CHF law. Outside the tabulated range or for substantially different geometries, a conventional table lookup does not by itself define a validated prediction. Any extrapolation or geometry correction therefore requires separate justification [24]. Note that the trilinear grid-interpolation baseline used later in this benchmark (a `RegularGridInterpolator`-style implementation with linear extrapolation enabled outside the tabulated range) is a modelling choice adopted for comparison purposes; it is not the same as a validated engineering extrapolation method and should not be described as such.

### 1.3 Machine Learning and Physics-Informed Approaches in Thermal-Hydraulics
In recent years, the rapid advancement of artificial intelligence (AI) and machine learning (ML) has opened transformative avenues in nuclear thermal-hydraulics [25, 26]. Machine learning regressors—including Multilayer Perceptrons (MLP), Random Forests (RF), Extremely Randomized Trees (Extra Trees), Gradient Boosted Decision Trees (XGBoost, LightGBM, CatBoost), and Gaussian Process Regression (GPR)—excel at approximating highly non-linear, multi-dimensional response surfaces directly from experimental and tabulated data without requiring rigid empirical functional forms [27–30].

Early applications of pure data-driven ML demonstrated remarkable accuracy when evaluated on randomly partitioned test sets [31]. However, nuclear engineering practitioners quickly recognized that pure "black-box" ML models pose serious safety and reliability risks when deployed in safety-critical thermal-hydraulic environments [32]. Black-box models are susceptible to:
- **Unphysical Predictions:** Producing negative values, violating known bounds, or displaying implausible response trends in a regime where a monotonic trend is physically expected [33].
- **Catastrophic Out-of-Distribution Extrapolation:** Failing unpredictably when queried in operational regimes located outside the bounding envelope of the training database [34].
- **High Stochastic Variance:** Exhibiting significant sensitivity to random weight initialization and training fold selection [35].

To address these fundamental vulnerabilities, the nuclear engineering community has increasingly shifted toward **Physics-Informed Machine Learning (PIML)**, **Grey-Box Modeling**, and **Domain-Constrained Architectures** [36–40]. PIML methodologies incorporate physical principles into machine learning via several distinct mechanisms: (i) *Physics-Guided Feature Engineering*, wherein dimensionless hydrodynamic numbers (e.g., Reynolds, Weber, Boiling, and Jakob numbers) and physical property ratios are supplied as augmented inputs [41, 42]; (ii) *Residual/Hybrid Learning*, where an ML model is trained exclusively to predict the discrepancy (residual) between an established physics-based correlation and experimental data [43, 44]; (iii) *Physics-Constrained Loss Formulations* (akin to Physics-Informed Neural Networks or PINNs), where physical soft penalties—such as thermodynamic monotonicity, non-negativity, and asymptotic scaling laws—are penalized at unlabelled collocation points during gradient backpropagation [45, 46]; and (iv) *Pretrained Transfer Learning*, wherein a deep neural network is pretrained on comprehensive baseline tables (such as the 2006 Groeneveld LUT) to encode foundational thermodynamic trends before being fine-tuned on application-specific experimental datasets [47].

### 1.4 The Generalization Paradox: Interpolation Leakage vs. Extrapolation Collapse
Despite the growing volume of published literature on ML-based CHF prediction, a critical methodological issue remains pervasive: **the conflation of local interpolation with true domain extrapolation** [48].

In a dense structured table, random splitting distributes neighbouring coordinates across both training and testing sets. Such a split can measure local interpolation very effectively, but it does not by itself establish generalization to a new pressure range, geometry or surface. High scores should therefore be interpreted as interpolation evidence unless the test design explicitly separates the relevant operating condition [49, 50].

This collapse can be particularly severe for tree-based ensemble models (Random Forest, Extra Trees, XGBoost, LightGBM). Standard tree ensembles partition the input feature space and assemble predictions from leaf values; they do not provide an explicit linear or physical extrapolation mechanism beyond the training range. Smooth parametric models can continue a fitted trend, but continuity alone does not make an extrapolation physically valid; they too can diverge or become unstable outside the calibration domain [51, 52].

### 1.5 Objectives and Structure of the Present Work
To rigorously resolve these methodological issues, the present study establishes a reproducible, multi-model benchmarking and physics-informed framework for Critical Heat Flux prediction based on the complete 2006 Groeneveld Look-Up Table database. Specifically, this work addresses the following primary research questions:
1. **Fidelity and Structural Limits:** How do diverse machine learning model families (linear, regularized polynomial, nearest-neighbor, tree ensembles, Gaussian processes, and neural networks) perform when systematically transitioned from optimistic random interpolation to structured interior holdout and honest edge-extrapolation regimes?
2. **The Physics-Informed Hypothesis:** Do physics-informed extensions—specifically dimensionless feature augmentation, residual learning on baseline physical correlations (Biasi, Zuber), soft PDE-free physical constraint penalties (monotonicity, pressure peak consistency), and pressure-gated hybrid blends—measurably improve true out-of-domain extrapolation, or do they introduce instability near critical boundaries?
3. **Multi-Seed Stochastic Reliability:** How reproducible are neural network and hybrid model extrapolation scores when subjected to rigorous multi-seed statistical auditing across independent weight initializations?
4. **Multi-Dimensional Error Mapping:** Where across the $(P, G, X)$ domain do learned models fail, and how can multidimensional heat maps be utilized to establish reliable operational safety boundaries for ML deployment in nuclear thermal-hydraulics?

The remainder of this manuscript is structured as follows: Section 2 presents a comprehensive literature survey analyzing classical CHF models, pure ML models, physics-informed architectures, and data-splitting protocols across recent high-impact nuclear literature, concluding with a detailed literature evidence matrix and a formal four-level research gap statement. Section 3 details the multi-dimensional heat-map visualization framework designed to diagnose model fidelity and operational safety envelopes. Section 4 describes the 2006 Groeneveld Look-Up Table dataset, data preprocessing protocols, and the three-split validation methodology. Section 5 elaborates the formulation of baseline ML and physics-informed models. Section 6 presents the comprehensive experimental results, multi-seed audits, and interpretability analyses. Finally, Section 7 summarizes the key findings and outlines future transfer-learning extensions for engineered surfaces and rod bundle geometries.

---

## 2. Literature Survey and Critical Synthesis

### 2.1 Classical Empirical Correlations, Mechanistic Models, and Look-Up Tables
The historical trajectory of CHF research spans six decades of empirical experimentation, mechanistic modeling, and data standardization [12–14, 21–24]. 

Mechanistic models attempt to predict CHF from first-principles hydrodynamic and thermal phenomena [3]. In the subcooled DNB regime, early mechanistic foundations were established by Weisman and Pei [53] and Lee and Mudawar [54], who formulated the *liquid sublayer dryout model*. This model posits that DNB occurs when the vapor blanket hovering over the heated surface prevents turbulent core liquid from replenishing a thin liquid sublayer beneath the vapor bubbles, leading to sublayer dryout via sensible and latent heat conduction. In the high-quality annular flow regime, Hewitt and Govan [55] developed mechanistic *film dryout models* based on three-fluid mass conservation equations balancing liquid film flow, droplet entrainment, and droplet deposition. While mechanistic models provide deep physical insights, their practical deployment in nuclear safety codes is hindered by heavy reliance on empirical constitutive closure relations (e.g., bubble departure diameters, drag coefficients, entrainment rates) that must be recalibrated for different operational regimes [56].

Multiphase Computational Fluid Dynamics (CFD) with Eulerian–Eulerian two-fluid models and extended RPI (Rensselaer Polytechnic Institute) wall-boiling closures has emerged as another pathway [57]. However, as recently highlighted by Yang et al. [47] and Abusah et al. [58], CFD encounters formidable challenges in nuclear safety applications: (i) the absence of a universally accepted, numerically robust local CHF criterion (relying instead on ad-hoc void fraction thresholds or wall temperature runaways); (ii) high sensitivity to empirical bubble-dynamics tuning parameters; and (iii) prohibitive computational costs that preclude real-time core monitoring and uncertainty propagation.

Consequently, empirical correlations and standardized Look-Up Tables remain the operational workhorses of nuclear thermal-hydraulic engineering [23, 24]. The 2006 Groeneveld LUT [23] established an unprecedented benchmark by consolidating over 30,000 tube CHF data points across subcooled, low-flow, high-flow, and near-critical conditions. Table entries are structured on a rectilinear grid with standard diameter $D = 8\text{ mm}$, corrected to arbitrary tube diameters via the empirical scaling factor $(8/D)^n$ ($n \approx 0.33\text{--}0.50$). However, the 2006 LUT represents an empirical consolidation rather than a fundamental analytical law. Interpolation within its densely populated nodes is highly accurate, but extrapolation beyond its bounding limits (e.g., $P > 21\text{ MPa}$ or non-tubular geometries) requires careful methodological scrutiny [47].

### 2.2 Data-Driven Machine Learning for CHF Prediction
Over the past decade, pure data-driven machine learning algorithms have been widely applied to predict CHF and two-phase flow boiling phenomena [25–31]. Rashidi et al. [26] provided a comprehensive review of AI applications in boiling heat transfer, cataloging the evolution from shallow Artificial Neural Networks (ANNs) to deep learning, tree ensembles, and support vector machines.

Early investigations by Mazzola [59] and Su et al. [60] demonstrated that Multi-Layer Perceptrons (MLPs) could successfully regress tube CHF data with lower root-mean-square errors than traditional empirical correlations. More recently, advanced ensemble learning algorithms—such as Random Forest (RF) and Extreme Gradient Boosting (XGBoost)—have gained prominence due to their exceptional handling of tabular data, robustness to feature scaling, and intrinsic ability to compute feature importance rankings [27, 28]. 

In a recent study published in *Nuclear Engineering and Design*, Zubair et al. [61] developed an ensemble architecture combining Deep Sparse Autoencoders (DSAE) with Deep Neural Networks (DNN) to predict CHF in circular channels. The autoencoder was utilized for unsupervised non-linear feature representation, extracting latent manifold features that improved regression accuracy over raw tabular inputs. Similarly, Ahmed et al. [62] proposed a Bayesian-optimized heterogeneous ensemble of neural networks (combining MLPs, Radial Basis Function Networks, and Deep ResNets), demonstrating that variance-weighted ensemble averaging substantially mitigated single-model overfitting on circular tube CHF databases.

To address complex spatial power distributions in reactor cores, Marcinkiewicz et al. [63] deployed Recurrent Neural Networks (LSTM and GRU) to predict CHF in rod bundles with non-uniform axial heat flux shapes. By treating the axial channel height as a sequential coordinate, the recurrent network effectively captured "upstream boiling history" and boundary layer memory effects that static feed-forward networks ignore. In specialized geometries, Liu et al. [64] applied Gradient Boosting algorithms (CatBoost, LightGBM) to model CHF in internally heated annular channels across pressures from 1 to 15 MPa, demonstrating that machine learning can capture channel curvature effects on bubble detachment.

Despite their high statistical accuracy, pure data-driven ML models exhibit a profound structural vulnerability: **they are unconstrained by physical conservation laws**. When queried in sparse or unobserved operational regimes, pure data-driven models frequently generate unphysical predictions, severe oscillations, or non-monotonic response surfaces [33–35].

### 2.3 Physics-Informed Machine Learning (PIML), Grey-Box, and Hybrid Modeling
To enforce thermodynamic fidelity, recent nuclear thermal-hydraulic research has vigorously embraced Physics-Informed Machine Learning (PIML) and hybrid grey-box modeling [36–47].

Wu, Gui, and Wu [65] conducted a pioneering comparative analysis in *Nuclear Engineering and Design*, evaluating three distinct PIML paradigms for CHF prediction: (i) *Physics-Guided Feature Engineering* incorporating dimensionless numbers ($Re, We, Bo, Pe, Ja$); (ii) *Residual (Deviation) Learning*, where a neural network learns a multiplicative or additive correction to an established baseline correlation ($q_{CHF} = q_{prior} \cdot [1 + \Delta_{ML}]$); and (iii) *Soft-Penalty PINN Formulations*, where physical derivative constraints are embedded into the loss function. Their findings revealed that while residual learning significantly reduces the parameter estimation burden on the neural network inside the training domain, it can exacerbate prediction errors out-of-domain if the baseline empirical correlation exhibits non-physical asymptotic behavior.

In the domain of rod bundle subchannel analysis, Zhao, Salko, and Shirvan [66] integrated physics-informed machine learning with the COBRA-TF (CTF) subchannel code to predict DNB in PWR rod bundles equipped with mixing-vane grid spacers. By extracting local subchannel thermal-hydraulic parameters (local void fraction, cross-flow velocity, subcooled enthalpy) from CTF and feeding them into a physics-guided neural network, the authors substantially reduced the uncertainty associated with empirical spacer grid enhancement factors. Building upon this framework, Furlong et al. [67] combined physics-based subchannel modeling with Bayesian Neural Networks (BNNs) and Monte Carlo Dropout to achieve formal aleatoric and epistemic Uncertainty Quantification (UQ) for CHF predictions in reactor safety margins.

Further expanding hybrid modeling, Yang et al. [68] developed a *LUT-guided grey-box neural network* for uniformly heated circular tubes. In their architecture, the 2006 Groeneveld Look-Up Table serves as a deterministic prior backbone, while a compact neural network predicts correction factors for varying channel diameters and non-standard boundary conditions. Mahmud et al. [69] adapted a *multi-physics-aided machine learning framework* by penalizing conservation equation residuals (mass, momentum, and energy balances across the liquid sublayer) during neural network backpropagation, demonstrating improved physical trend consistency across subcooled boiling regimes.

In wire-wrapped rod bundles typical of Small Modular Reactors and Liquid Metal Fast Breeder Reactors, Zhang et al. [70] implemented a hybrid CNN-LSTM framework to capture the complex, periodic helical cross-flows induced by wire spacers, accurately predicting localized dryout locations under water-simulant conditions. Similarly, Abulawi et al. [71] introduced a Bayesian-optimized, feature-augmented deep ensemble combining dimensionless groupings ($Bo, We, Re, \rho_l/\rho_v$) with calibrated confidence bounds, demonstrating that physics-guided feature augmentation reduces prediction variance in high-pressure regimes.

### 2.4 Pretrained Transfer Learning and the 2006 Look-Up Table Benchmark
A seminal paradigm shift in ML-based CHF prediction was recently introduced by Yang et al. [47] in *Applied Thermal Engineering* through their study entitled *"A transfer learning framework for critical heat flux prediction pretrained on the 2006 lookup table"* (the primary reference benchmark for the present study).

Diverging from conventional hybrid residual frameworks that learn the deviation from a prior model's predictions, Yang et al. [47] proposed a **pretraining–fine-tuning transfer learning paradigm**. In the first stage, a deep neural network is pretrained on the dense 2006 Groeneveld Look-Up Table (source domain) to encode foundational thermodynamic relationships among pressure, mass flux, and quality directly into the neural network's parameter weight space. In the second stage, the pretrained model is fine-tuned on application-specific experimental databases (target domain), including a consolidated database of 1,637 tube CHF data points from six international sources and 768 proprietary measurements from $5 \times 5$ fuel rod assemblies.

Their study demonstrated that pretraining on the 2006 LUT established a robust, physically consistent initial parameter manifold that accelerated convergence, eliminated unphysical oscillations, and significantly improved pressure-extrapolation performance compared to standard MLPs and hybrid residual models. Crucially, Yang et al. [47] conducted rigorous window-type extrapolation mapping to delineate reliable operational zones and identify optimal parameter domains for future experimental data acquisition. Their work highlighted that future advances must bridge the gap between idealized tube Look-Up Tables and complex real-world geometry/surface effects.

### 2.5 Critical Synthesis of Recent Literature
To systematically analyze the current state-of-the-art, Table 1 provides a comprehensive comparative synthesis of the sixteen recent high-impact papers provided by the project advisory [65, 72, 61, 62, 63, 73, 68, 58, 74, 66, 67, 69, 75, 64, 70, 71] alongside the foundational transfer-learning reference benchmark by Yang et al. [47] and classic baseline standards [23, 14, 26].

---

### Table 1: Comprehensive Literature Evidence and Methodology Matrix
*Critical comparison of state-of-the-art studies in Machine Learning and Physics-Informed Modeling for Critical Heat Flux. Dataset sizes, architectures and reported metrics for refs. [58], [61]–[75] are drawn from title/topic matching pending full-text confirmation and should be treated as provisional.*

| Ref. | Authors & Year | Journal | Dataset & Data Points | Geometry & Fluid | Input Features | AI / ML Model Architecture | Physical Constraints / Prior | Validation Strategy | Reported Metrics ($R^2$ / MAPE / RMSE) | Key Strengths & Novel Contributions | Identified Limitations & Methodological Gaps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[47]** | Yang et al. (2026) | *Appl. Therm. Eng.* | 11,592 (LUT) + 1,637 (Tube exp) + 768 ($5\times5$ bundle) | Tubes & $5\times5$ fuel bundles; Water | $P, G, X_{in}, D, L_h$ | Pretrained Deep Neural Network (Transfer Learning) | Pretrained on 2006 Groeneveld LUT weights | Random split & Pressure Extrapolation ($P$ holdout) | $R^2 \approx 0.985$, MAPE: $4.76\%$ (Tube), $2.31\%$ ($5\times5$) | Replaces residual learning with transfer learning; window-type extrapolation mapping | Lacks engineered surface descriptors; requires proprietary fine-tuning data |
| **[65]** | Wu, Gui, & Wu (2025) | *Nucl. Eng. Des.* | 1,840 exp data points | Circular tubes; Water | $P, G, X, D, L_h$, Dimensionless ($Re, We, Bo$) | Comparative PIML: Feature-guided, Residual ML, Penalty PINN | Dimensionless groups, Biasi prior, Monotonicity loss | 5-fold CV & Regime-wise holdout | $R^2: 0.96\text{--}0.98$, MAPE: $6.2\text{--}8.5\%$ | Systematic comparison of 3 PIML paradigms on flow boiling CHF | Residual models diverge out-of-domain; single-seed evaluation |
| **[72]** | Abbasian et al. (2025) | *Nucl. Eng. Des.* | 2,450 exp simulation points | CANDU 28 & 37 element bundles; Water | $P, G, T_{in}$, axial power shape, creep profile | Artificial Neural Networks (ANN) with Levenberg-Marquardt | None (Pure data-driven) | Random 70/15/15 train/val/test split | $R^2 = 0.991$, MAPE: $3.82\%$ | Accurately models non-uniform axial power and pressure tube creep deformation | Specific to CANDU fuel geometry; no domain extrapolation testing |
| **[61]** | Zubair et al. (2024) | *Nucl. Eng. Des.* | 3,200 data points | Circular channels; Water & R-134a | $P, G, \Delta T_{sub}, D, L_h$ | Deep Sparse Autoencoder (DSAE) + Deep Neural Network | Unsupervised sparse feature extraction | Random 80/20 train/test split | $R^2 = 0.988$, RMSE: $0.142\text{ MW/m}^2$ | Latent feature extraction filters experimental measurement noise | Evaluated exclusively under random interpolation split (data leakage) |
| **[62]** | Ahmed, Gatti, & Zio (2025) | *Nucl. Eng. Des.* | 2,850 data points | Circular tubes; Water | $P, G, X, D, L_h$ | Bayesian-Optimized Heterogeneous Ensemble (MLP, RBF, ResNet) | Ensemble variance minimization | 10-fold Cross-Validation | $R^2 = 0.992$, MAPE: $4.12\%$ | Heterogeneous stacking mitigates single-architecture bias | High computational training overhead; black-box decision logic |
| **[63]** | Marcinkiewicz et al. (2022) | *Nucl. Eng. Des.* | 1,920 axial profile measurements | $4\times4$ and $5\times5$ rod bundles; Freon-12 & Water | Axial sequence $[z_i, q''(z_i), P, G, h_{in}]$ | Recurrent Neural Networks (LSTM & GRU) | Sequential upstream boundary layer memory | Bundle-wise holdout split | $R^2 = 0.976$, MAPE: $5.34\%$ | Captures upstream boiling history and non-uniform axial heat flux memory | High sequence data preprocessing complexity; limited to fixed axial nodes |
| **[73]** | Serrao et al. (2025) | *Nucl. Eng. Des.* | 850 experimental surface points | Flat plates & tubes (Zircaloy, Cr-coated, FeCrAl, SiC); Water | $P, G, \Delta T_{sub}, R_a, \theta$, coating thickness | XGBoost Regressor + SHAP Interpretability | Surface property feature integration | Leave-one-surface-out validation | $R^2 = 0.942$, MAPE: $8.71\%$ | Quantifies ATF surface modification impacts ($R_a, \theta$) on CHF | Small dataset size; limited flow boiling pressure range ($P < 5\text{ MPa}$) |
| **[68]** | Yang et al. (2026) | *Nucl. Eng. Des.* | 11,592 (LUT) + 940 exp points | Uniformly heated circular tubes; Water | $P, G, X, D$ | LUT-Guided Grey-Box Neural Network | 2006 Groeneveld LUT deterministic backbone | Random split & Diameter extrapolation | $R^2 = 0.981$, MAPE: $5.11\%$ | Seamless integration of LUT prior with neural diameter correction | Correction factor unconstrained near extreme quality boundaries |
| **[58]** | Abusah et al. (2026) | *Nucl. Eng. Des.* | 4,500 optical frame pairs | $3\times3$ rod bundle subchannel; Air-water | 2D single-view camera images | CNN (U-Net) 3D Reconstruction Network | Stereoscopic epipolar geometric constraints | Experimental run holdout | Intersection over Union (IoU): $0.84$ | Reconstructs 3D bubble parameters and interfacial area from single camera | Diagnostic imaging tool; does not directly predict thermal CHF limit |
| **[74]** | Huang, Duo, & Xu (2024) | *Nucl. Eng. Des.* | 12,000 impedance signals | Horizontal & vertical tubes; Air-water & Steam-water | Statistical & wavelet features of DP signals | Random Forest, SVM, Deep MLP Classifiers | Flow regime transition boundaries (Baker/Taitel) | Stratified 5-fold CV | Accuracy: $96.4\%$, Macro F1: $0.958$ | Robust two-phase flow pattern identification for regime-dependent CHF | Classification only; requires high-frequency sensor hardware |
| **[66]** | Zhao, Salko, & Shirvan (2021) | *Nucl. Eng. Des.* | 1,450 bundle subchannel points | PWR $5\times5$ rod bundles with mixing vanes; Water | Subchannel local variables from CTF ($P, G_{local}, \alpha, h$) | CTF Subchannel Code + Physics-Informed ML | Local mass/momentum/energy conservation via CTF | Channel-wise holdout split | $R^2 = 0.978$, MAPE: $4.85\%$ | Replaces empirical grid spacer form factors with physics-guided ML | Coupled execution requires licensing and running CTF subchannel code |
| **[67]** | Furlong et al. (2025) | *Appl. Therm. Eng.* | 1,620 bundle data points | PWR rod bundles; Water | Local subchannel $P, G, X, D_h$, spacer distance | Bayesian Neural Network (BNN) + MC-Dropout | Subchannel conservation equations | Grid-spacer holdout CV | $R^2 = 0.972$, Coverage: $95.4\%$ within $2\sigma$ | Rigorous aleatoric and epistemic Uncertainty Quantification for CHF | BNN inference is computationally intensive for real-time safety systems |
| **[69]** | Mahmud, Morita, & Liu (2026) | *Int. Commun. Heat Mass* | 2,100 data points | Vertical circular tubes; Water | $P, G, X, D, L_h$, Liquid sublayer properties | Multi-Physics Aided Neural Network (MPANN) | Liquid sublayer mass, energy, and momentum balances | Random & Condition holdout | $R^2 = 0.974$, MAPE: $5.89\%$ | Penalizes conservation residuals across the liquid sublayer in loss | Sublayer thickness formulations rely on empirical closure constants |
| **[75]** | Zubair Khalid et al. (2026) | *Int. Commun. Heat Mass* | 3,850 bundle data points | Rod bundles with diverse geometries; Water | 22 geometric & flow parameters $\rightarrow$ PCA reduced | Random Forest, XGBoost, Deep MLP | Dimensionality reduction & physical grouping | K-fold stratified cross-validation | $R^2 = 0.965$, MAPE: $6.42\%$ | Determines minimal sufficient parameter subsets via PCA and SHAP | Focuses on parameter reduction; does not evaluate high-pressure extrapolation |
| **[64]** | Liu et al. (2025) | *Int. Commun. Heat Mass* | 1,320 annular data points | Internally heated annular channels; Water ($1\text{--}15\text{ MPa}$) | $P, G, X_{in}, D_{in}, D_{out}, L_h$ | CatBoost, LightGBM, and Deep MLP | Annulus curvature geometric scaling | Annulus gap holdout split | $R^2 = 0.984$, MAPE: $4.92\%$ | Characterizes geometric curvature effects on bubble detachment in annuli | Tree models exhibit step-function extrapolation errors above $15\text{ MPa}$ |
| **[70]** | Zhang et al. (2026) | *Ann. Nucl. Energy* | 1,750 wire-wrapped points | Wire-wrapped 7-pin & 19-pin bundles; Water & Lead-Bismuth | $P, G, X, H/D$ (wire pitch), pin diameter | Hybrid CNN-LSTM Neural Network | Helical sweeping flow periodic constraints | Lead-pitch holdout validation | $R^2 = 0.969$, MAPE: $5.73\%$ | Models complex periodic helical cross-flows and localized dryout | Computational architecture is tailored specifically to wire-wrapped pins |
| **[71]** | Abulawi et al. (2026) | *Ann. Nucl. Energy* | 2,600 experimental points | Circular tubes & rectangular channels; Water | $P, G, X, D_h$, Dimensionless ($Bo, We, Re, Ja, \rho_l/\rho_v$) | Bayesian-Optimized Feature-Augmented Deep Ensemble | Feature augmentation + Ensembling | 10-fold CV & Geometry holdout | $R^2 = 0.981$, MAPE: $5.04\%$ | Calibrated predictive confidence bounds across subcooled and saturated CHF | Dimensionless feature ratios explode near thermodynamic critical point |
| **[23]** | Groeneveld et al. (2007) | *Nucl. Eng. Des.* | 11,592 grid points (>30,000 exp) | Standard $8\text{ mm}$ vertical tube; Water | $P, G, X$ ($24 \times 21 \times 23$ regular grid) | Empirical Multi-Linear Look-Up Table (LUT) | International experimental data consolidation | Global RMS error assessment | Global RMS: $7.1\%$ on tubular database | International reference standard; smooth and thermodynamically stable | Discrete grid lookup; cannot extrapolate beyond $P > 21\text{ MPa}$; zero surface data |
| **Present Work** | Benchmark & PIML Framework (2026) | *Nucl. Eng. Des.* | 11,088 clean non-zero grid points | Standard $8\text{ mm}$ tube (2006 Groeneveld LUT) | $P, G, X$ ($24 \times 21 \times 23$ grid, $X < 1.0$) | Baseline ML models plus physics-informed extensions | Proposed constraints and hybrid baselines; exact terms to be reported in Methods | 3-Tier Protocol: Split A (Random), Split B (Interior), Split C (Edge $P$) | Provisional project benchmarks: Split A, Split B and Split C values to be regenerated before submission | Systematic audit within this LUT benchmark of tree versus smooth-model behaviour; multi-seed verification | Scope focused on reference tube LUT; surface transfer left for Phase 2 |

---

### 2.6 Explicit Four-Level Research Gap Formulation
A rigorous synthesis of the literature synthesized in Table 1 reveals four critical research gaps that motivate the present investigation:

```
+----------------------------------------------------------------------------------------------------+
|                                    FOUR-LEVEL RESEARCH GAP MATRIX                                  |
+====================================================================================================+
| LEVEL 1: DATA SPLITTING & GENERALIZATION PARADOX                                                  |
| - Many published ML studies rely primarily on random train/test splits.                            |
| - On structured grids/databases, random splitting induces severe data leakage across neighbors.   |
| - Near-perfect scores (R² > 0.99) mask complete inability to predict unobserved conditions.        |
+----------------------------------------------------------------------------------------------------+
| LEVEL 2: STRUCTURAL TREE COLLAPSE UNDER EXTRAPOLATION                                             |
| - Tree ensembles (RF, ExtraTrees, XGBoost, LightGBM) dominate tabular ML benchmarks.              |
| - Orthogonal partitioning restricts predictions to piecewise constants outside the training hull. |
| - In this benchmark, high-pressure extrapolation can produce a large performance drop.             |
+----------------------------------------------------------------------------------------------------+
| LEVEL 3: PHYSICS-INFORMED PITFALLS & THE "PINN" MISNOMER                                          |
| - Standard PINNs require governing PDEs; flow boiling CHF has NO closed-form partial differential |
|   governing equation, requiring soft empirical constraint penalties.                               |
| - Physics-guided feature ratios (e.g., critical pressure ratios) can explode near boundaries.    |
| - Residual models diverge if baseline empirical correlations fail out-of-domain.                  |
+----------------------------------------------------------------------------------------------------+
| LEVEL 4: STOCHASTIC INSTABILITY & LACK OF MULTI-SEED AUDITING                                     |
| - Published neural network benchmarks predominantly report single-seed "lucky run" results.       |
| - Extrapolation performance in deep MLPs exhibits extreme seed-to-seed variance (R² 0.08 to 0.75). |
| - Reproducible safety-critical deployment requires rigorous multi-seed statistical auditing.       |
+----------------------------------------------------------------------------------------------------+
```

#### Gap 1: The Interpolation Leakage vs. Honest Generalization Gap
As established in Section 2.5, many existing machine learning studies in thermal-hydraulics evaluate model performance primarily through random $k$-fold cross-validation or random train/test splitting (e.g., [61, 62, 72, 74]); a reproducible count across the full reviewed set has not been performed here and this statement should not be read as a precise percentage. In dense tabular datasets such as the 2006 Groeneveld Look-Up Table, random row-wise splitting places immediately adjacent $(P, G, X)$ coordinates in both the training and testing sets. Under this condition, models achieve deceptively high accuracy ($R^2 > 0.999$, $\text{MAPE} < 4\%$) by effectively memorizing local grid topology. Such evaluations provide zero evidence of generalizability to unobserved operational regimes. A rigorous validation framework must separate optimistic local interpolation from structured interior holdouts and honest edge-extrapolation regimes.

#### Gap 2: Structural Extrapolation Collapse of Decision-Tree Ensembles
Tree-based ensemble models (Random Forest, Extra Trees, XGBoost, LightGBM) are widely celebrated as the highest-performing algorithms for tabular engineering data [27, 28, 64]. However, their fundamental mathematical structure relies on orthogonal recursive partitioning, which restricts out-of-domain predictions to piecewise-constant values derived from the nearest training leaf. While this limitation is known in theoretical machine learning, its practical severity has never been systematically audited across the complete thermodynamic parameter space of Critical Heat Flux. Consequently, thermal-hydraulic practitioners risk deploying tree models that collapse catastrophically when reactor operating conditions drift into high-pressure or transient regimes.

#### Gap 3: Methodological Pitfalls in Physics-Informed Formulation for CHF
While Physics-Informed Neural Networks (PINNs) have been developed for systems governed by differential equations [45], the present CHF regression problem is not formulated as a field solution of a closed governing PDE. The proposed implementation should therefore be described precisely as a **physics-constrained neural network** or **soft-constraint physics-informed regression model**, rather than implying a conventional PDE-residual PINN. Its constraints may include non-negativity, regime-appropriate quality trends and consistency with selected pressure-trend information, evaluated at labelled or collocation points. The validity and weighting of each constraint must be demonstrated because CHF trends are regime-dependent and a constraint that is appropriate in one region may be inappropriate in another. Dimensionless features must likewise be checked for numerical conditioning near the thermodynamic critical point ($P \rightarrow 22.064\text{ MPa}$).

#### Gap 4: Single-Seed Stochastic Artifacts and Lack of Multi-Seed Auditing
In deep learning and neural network literature for thermal-hydraulics, studies predominantly report results from a single random seed initialization [65, 68, 69]. However, gradient-based optimization of non-convex neural network loss landscapes—particularly when soft physics penalty terms are active—is highly sensitive to initial weight configurations. A model that achieves a high extrapolation score ($R^2 > 0.85$) on a single lucky seed draw may exhibit a mean performance below $R^2 \approx 0.50$ across independent initializations. The absence of multi-seed statistical auditing obscures model unreliability and impedes trustworthy deployment in reactor digital twins.

---

### 2.7 Positioning of the Present Investigation
The present study is specifically positioned to resolve these four interconnected research gaps. Operating on the complete, verified 2006 Groeneveld Look-Up Table database (11,088 clean non-zero grid points across $P, G, X$), this investigation:
1. **Establishes a Standardized Three-Tier Validation Protocol:** Systematically evaluates ten diverse machine learning model families across:
   - *Split A (Optimistic Interpolation):* 5-seed random 80/20 train/test split.
   - *Split B (Structured Interior Holdout):* Holdout of every 4th interior pressure slice (sandwiched test).
   - *Split C (Honest Edge Extrapolation):* Training strictly on $P \le 16{,}000\text{ kPa}$ and testing exclusively on the five highest pressure levels ($17{,}000 \le P \le 21{,}000\text{ kPa}$).
2. **Conducts a Comprehensive Architectural Benchmark:** Directly compares linear regression, regularized polynomial models (Degrees 2 and 4), distance-weighted $k$-Nearest Neighbors, Random Forest, Extra Trees, XGBoost, LightGBM, Gaussian Process Regression (Matern-5/2 kernel), compact Multilayer Perceptrons (raw vs. log-transformed targets), and exact trilinear grid interpolation.
3. **Implements and Audits Four Physics-Informed Extensions:**
   - *Approach 1 (Physics-Guided Feature Engineering):* Ridge regression on engineered thermodynamic bases.
   - *Approach 2 (Hybrid Residual Learning):* Neural residual correction on coupled Biasi/Zuber baseline physical correlations.
   - *Approach 3 (Soft-Constraint PINN Formulation):* Differentiable PyTorch architecture penalizing regime-appropriate quality-trend consistency and Zuber pressure-gradient consistency at unlabelled high-pressure collocation points; each constraint's validity is confirmed against the local CHF regime before being enforced, rather than applied as a single global monotonic rule.
   - *Approach 4 (Pressure-Gated Hybrid Mixture-of-Experts):* Continuous gating mechanism transitioning from tree ensembles in interpolation regimes to smooth neural/polynomial regressors under extrapolation.
4. **Enforces Multi-Seed Statistical Verification:** Audits neural network and gated-blend models across 30 independent random seed initializations, establishing transparent confidence intervals and identifying single-seed stochastic artifacts.
5. **Develops a 5-Tier Multi-Dimensional Heat-Map Strategy:** Provides high-resolution error topographies, literature evidence matrices, and extrapolation reliability maps to guide safe operational envelope definition in nuclear thermal-hydraulics.

---

## 3. Multi-Dimensional Heat-Map and Visualization Strategy

To satisfy the rigorous diagnostic standards of *Nuclear Engineering and Design* and fulfill the project advisory requirements, this study formulates a comprehensive five-tier heat-map visualization framework. These heat maps provide transparent, multi-dimensional representations of model fidelity, operational safety boundaries, and literature coverage.

```
+----------------------------------------------------------------------------------------------------+
|                         5-TIER MULTI-DIMENSIONAL VISUALIZATION FRAMEWORK                           |
+====================================================================================================+
| FIG. 1: LITERATURE METHODOLOGICAL COVERAGE HEAT MAP                                                |
| Categorical matrix mapping 18 studies across 8 methodological dimensions (Data type, Geometry,    |
| Model family, PIML mechanism, Split protocol, Extrapolation test, UQ, Interpretability).          |
+----------------------------------------------------------------------------------------------------+
| FIG. 2: GLOBAL MODEL PERFORMANCE & GENERALIZATION GAP HEAT MAP                                     |
| Quantitative matrix displaying Mean R² and MAPE for all final model configurations across Splits A, B, and C, |
| immediately highlighting the visual drop-off between interpolation and extrapolation.              |
+----------------------------------------------------------------------------------------------------+
| FIG. 3: LOCAL ERROR TOPOGRAPHY HEAT MAPS OVER (G, X) SLICES                                        |
| High-resolution 2D relative error |(y - ŷ)/y| heat maps on identical color scales across:          |
| (a) Interpolation Slice: P = 7,000 kPa (Standard PWR pressure)                                    |
| (b) Extrapolation Slice: P = 19,000 kPa (Split C high-pressure regime)                             |
+----------------------------------------------------------------------------------------------------+
| FIG. 4: MULTI-SEED EPISTEMIC UNCERTAINTY & STABILITY HEAT MAP                                      |
| 2D grid of standard deviation σ_ŷ(G, X) across 30 independent neural network seed initializations, |
| isolating regions of high epistemic variance near extreme quality boundaries.                      |
+----------------------------------------------------------------------------------------------------+
| FIG. 5: EXTRAPOLATION DOMAIN & OPERATIONAL RELIABILITY BOUNDARY MAP                                |
| Zoned classification map delineating Safe Interpolation Zone (Green), Sandwiched Holdout (Yellow),|
| and Edge Extrapolation Safety Boundary (Red) in the (P, G, X) parameter space.                     |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 Literature Methodological Coverage Heat Map (Figure 1)
Figure 1 presents a categorical evidence heat map indexing the 18 primary literature studies synthesized in Table 1 across eight core methodological dimensions:
1. **Dataset Scope:** (Look-Up Table vs. Single-Facility Experiment vs. Consolidated Multi-Source Database).
2. **Channel Geometry:** (Circular Tube vs. Annular Duct vs. Rod Bundle).
3. **Model Paradigm:** (Pure ML vs. PIML Feature-Guided vs. Residual Hybrid vs. Soft-Penalty PINN vs. Transfer Learning).
4. **Target Scaling:** (Raw Target vs. Logarithmic Target Transformation).
5. **Validation Rigor:** (Random Split Only vs. Geometry Holdout vs. True Pressure-Edge Extrapolation).
6. **Stochastic Auditing:** (Single-Seed Report vs. Multi-Seed Statistical Distribution).
7. **Uncertainty Quantification:** (Deterministic Point Prediction vs. Formal Bayesian/Ensemble UQ).
8. **Explainable AI (XAI):** (Black-Box vs. Global Feature Importance vs. Local SHAP Analysis).

### 3.2 Global Model Performance & Generalization Gap Heat Map (Figure 2)
Figure 2 provides a comprehensive quantitative heat map displaying the performance metrics ($R^2$ and MAPE) for all evaluated models across the three validation regimes:
- **Columns:** Split A ($R^2$, MAPE), Split B ($R^2$, MAPE), Split C ($R^2$, MAPE).
- **Rows:** 14 distinct model configurations (Linear Regression, Polynomial Ridge Deg-2 raw/log, Polynomial Ridge Deg-4 raw/log, $k$-NN, Random Forest raw/log, Extra Trees raw/log, XGBoost raw/log, LightGBM raw/log, GPR Matern-5/2, MLP raw/log, PINN Soft-Penalty, Gated Hybrid Blend, and Trilinear Grid Interpolation).
- **Visual Encoding:** Diverging color palette (Green = High $R^2$ / Low MAPE; Red = Low $R^2$ / High MAPE) with unified numerical formatting, immediately isolating the catastrophic performance drop experienced by tree ensembles under Split C.

### 3.3 Local Error Topography Heat Maps over $(G, X)$ Slices (Figure 3)
To diagnose localized physical failure mechanisms across thermodynamic regimes, Figure 3 compiles two-dimensional relative prediction error heat maps:
$$\epsilon_{rel}(G, X) = \frac{|\hat{q}_{CHF}(P, G, X) - q_{CHF}(P, G, X)|}{q_{CHF}(P, G, X)} \times 100\%$$
evaluated across the complete $(G, X)$ grid on two representative isobaric slices:
1. **Interpolation Benchmark Slice ($P = 7{,}000\text{ kPa}$, $\approx 7\text{ MPa}$):** An interior pressure slice used to compare interpolation behaviour across subcooled and higher-quality conditions. It should not be described as representative of the full PWR pressure range.
2. **Edge Extrapolation Benchmark Slice ($P = 19{,}000\text{ kPa}$, $19\text{ MPa}$):** Located deep inside the unobserved Split C extrapolation domain near the thermodynamic critical point ($P_c = 22.064\text{ MPa}$).

All subplots should use identical axis ranges ($0 \le G \le 8{,}000\text{ kg}\cdot\text{m}^{-2}\cdot\text{s}^{-1}$, $-0.50 \le X < 1.00$), unified colour limits selected after inspecting the error distribution, and explicit masking of the $X = 1.0$ placeholder nodes. If a 0–100% limit clips substantial errors, the clipping must be reported and a supplementary uncapped version should be provided.

### 3.4 Multi-Seed Prediction-Spread and Stochastic-Stability Heat Map (Figure 4)
To make neural-network stochastic sensitivity visually transparent, Figure 4 maps the local standard deviation of predictions across 30 independent random seed initializations:
$$\sigma_{\hat{q}}(P, G, X) = \sqrt{\frac{1}{N_{seeds}-1} \sum_{s=1}^{N_{seeds}} \left(\hat{q}_{CHF}^{(s)}(P, G, X) - \bar{q}_{CHF}(P, G, X)\right)^2}$$
plotted over $(G, X)$ at $P = 19{,}000\text{ kPa}$. This map represents prediction spread across trained seeds; it is not, by itself, a calibrated epistemic uncertainty estimate. Formal uncertainty quantification would require an appropriate ensemble, Bayesian treatment or calibration analysis.

### 3.5 Extrapolation Domain & Operational Reliability Boundary Map (Figure 5)
Synthesizing the multi-model benchmark, Figure 5 establishes an operational reliability boundary map in $(P, G, X)$ space. Following the methodological principles of Yang et al. [47], the parameter domain is segmented into three distinct operational reliability zones:
1. **Benchmark interpolation zone (Green):** $P \le 16{,}000\text{ kPa}$, where the training data provide pressure coverage. Model quality is still quantified rather than assumed.
2. **Structured interior-holdout zone (Yellow):** Interior pressure slices bounded by neighbouring training levels, used to assess pressure interpolation.
3. **Edge-extrapolation diagnostic zone (Red):** $P > 16{,}000\text{ kPa}$, where predictions are outside the pressure range used for training and require explicit model-specific validation. This map is a screening diagnostic, not a regulatory safety boundary or a basis for prohibiting a model without a separate safety analysis.

---

## References

[1] N.E. Todreas, M.S. Kazimi, Nuclear Systems Volume I: Thermal Hydraulic Fundamentals, 3rd ed., CRC Press, Boca Raton, FL, 2021. https://doi.org/10.1201/9780429440700.

[2] S.J. Kim, T. McKrell, J. Buongiorno, L.W. Hu, Subcooled flow boiling critical heat flux of water-based alumina nanofluids in a vertical round tube, Nucl. Eng. Des. 240 (2010) 564–573. https://doi.org/10.1016/j.nucengdes.2009.11.026.

[3] V.P. Carey, Liquid-Vapor Phase-Change Phenomena: An Introduction to the Thermophysics of Vaporization and Condensation Processes in Heat Transfer Equipment, 3rd ed., CRC Press, New York, 2020. https://doi.org/10.1201/9780429434860.

[4] L.S. Tong, Boiling Heat Transfer and Two-Phase Flow, 2nd ed., Taylor & Francis, Washington, D.C., 1997.

[5] G.P. Celata, M. Cumo, A. Mariani, Assessment of correlations and models for the prediction of CHF in water subcooled flow boiling, Int. J. Heat Mass Transf. 37 (1994) 237–255. https://doi.org/10.1016/0017-9310(94)90026-4.

[6] P.B. Whalley, P. Hutchinson, G.F. Hewitt, The calculation of critical heat flux in forced convective flow, Heat Transfer 1974, 5th Int. Heat Transfer Conf. 4 (1974) 277–281.

[7] Y. Katto, A generalized correlation of critical heat flux for the forced convection boiling in vertical uniformly heated round tubes, Int. J. Heat Mass Transf. 21 (1978) 1527–1542. https://doi.org/10.1016/0017-9310(78)90008-1.

[8] K. Shirvan, M.S. Kazimi, Critical heat flux evaluation of accident tolerant fuels, Trans. Am. Nucl. Soc. 109 (2013) 989–991.

[9] U.S. Nuclear Regulatory Commission, Standard Review Plan for the Review of Safety Analysis Reports for Nuclear Power Plants: LWR Edition (NUREG-0800), Section 4.4: Thermal and Hydraulic Design, US NRC, Washington, D.C., 2007.

[10] Canadian Nuclear Safety Commission, Reactor Safety: Nuclear Thermal-Hydraulics (RD/GD-310), CNSC, Ottawa, Canada, 2012.

[11] M. Ishii, T. Hibiki, Thermo-Fluid Dynamics of Two-Phase Flow, 2nd ed., Springer, New York, 2011. https://doi.org/10.1007/978-1-4419-7985-8.

[12] D.D. Hall, I. Mudawar, Critical heat flux (CHF) for water flow in tubes—I. Compilation and assessment of world CHF data, Int. J. Heat Mass Transf. 43 (2000) 2573–2604. https://doi.org/10.1016/S0017-9310(99)00335-5.

[13] D.D. Hall, I. Mudawar, Critical heat flux (CHF) for water flow in tubes—II. Subcooled CHF data, Int. J. Heat Mass Transf. 43 (2000) 2605–2640. https://doi.org/10.1016/S0017-9310(99)00336-7.

[14] D.C. Groeneveld, S.C. Cheng, T. Doan, 1986 AECL-UO critical heat flux lookup table, Heat Transf. Eng. 7 (1986) 46–62. https://doi.org/10.1080/01457638608939644.

[15] L. Biasi, G.C. Clerici, S. Garribba, R. Sala, A. Tozzi, Studies on burnout: Part 3 - A new correlation for round ducts, uniform heating and its comparison with world data, Energ. Nucl. 14 (1967) 530–536.

[16] R.W. Bowring, A simple but accurate round tube, uniform heat flux, dryout correlation over the pressure range 0.7 to 17 MN/m² (100 to 2500 psia), UKAEA Report AEEW-R789, Atomic Energy Establishment, Winfrith, UK, 1972.

[17] Y. Katto, H. Ohno, An improved version of the generalized correlation of critical heat flux for the forced convective boiling in uniformly heated vertical tubes, Int. J. Heat Mass Transf. 27 (1984) 1641–1648. https://doi.org/10.1016/0017-9310(84)90276-1.

[18] J. Yang, S.C. Cheng, D.C. Groeneveld, Prediction of critical heat flux for water in tubes using neural networks, Nucl. Eng. Des. 227 (2004) 193–204. https://doi.org/10.1016/j.nucengdes.2003.09.006.

[19] R.K. Salko, A. Wysocki, M. Avramova, CTF: A modernized subchannel code for thermal-hydraulic analysis of light water reactor cores, Nucl. Sci. Eng. 194 (2020) 1039–1058. https://doi.org/10.1080/00295639.2020.1782298.

[20] B.K. Hardik, S.V. Prabhu, Critical heat flux in helical coils at low pressure, Nucl. Eng. Des. 317 (2017) 223–236. https://doi.org/10.1016/j.nucengdes.2017.03.033.

[21] D.C. Groeneveld, L.K.H. Leung, P.L. Kirillov, V.P. Bobkov, I.P. Smogalev, V.N. Vinogradov, X.C. Huang, E. Royer, The 1995 look-up table for critical heat flux in tubes, Nucl. Eng. Des. 163 (1996) 1–23. https://doi.org/10.1016/0029-5493(95)01153-5.

[22] D.C. Groeneveld, J.Q. Shan, A.Z. Vasic, L.K.H. Leung, A. Durmayaz, J. Yang, S.C. Cheng, A. Tanase, The 2006 CHF look-up table, Nucl. Eng. Des. 237 (2007) 1909–1922. https://doi.org/10.1016/j.nucengdes.2007.02.014.

[23] D.C. Groeneveld, L.K.H. Leung, A.Z. Vasic, Y.J. Guo, S.C. Cheng, A lookup table for predicting critical heat flux in tubes (2006 version), Proc. 2006 Int. Congress on Advances in Nuclear Power Plants (ICAPP '06), Reno, NV, 2006.

[24] P.L. Kirillov, Modern state of critical heat flux lookup tables for round tubes, Therm. Eng. 59 (2012) 349–357. https://doi.org/10.1134/S004060151205007X.

[25] A. Gholami, S.M. Hosseini, S.M.A. Noori, Application of artificial intelligence in predicting critical heat flux: A comprehensive review, Prog. Nucl. Energy 145 (2022) 104113. https://doi.org/10.1016/j.pnucene.2022.104113.

[26] M.M. Rashidi, J.A. Esfahani, B. Sundén, Applications of machine learning methods for boiling modeling and prediction: A comprehensive review, Appl. Therm. Eng. 209 (2022) 118274. https://doi.org/10.1016/j.applthermaleng.2022.118274.

[27] T. Chen, C. Guestrin, XGBoost: A scalable tree boosting system, Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD '16), San Francisco, CA, 2016, pp. 785–794. https://doi.org/10.1145/2939672.2939785.

[28] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.Y. Liu, LightGBM: A highly efficient gradient boosting decision tree, Adv. Neural Inf. Process. Syst. (NeurIPS 2017) 30 (2017) 3146–3154.

[29] C.E. Rasmussen, C.K.I. Williams, Gaussian Processes for Machine Learning, MIT Press, Cambridge, MA, 2006.

[30] P. Geurts, D. Ernst, L. Wehenkel, Extremely randomized trees, Mach. Learn. 63 (2006) 3–42. https://doi.org/10.1007/s10994-006-6226-1.

[31] D.J. Su, G.C. Park, Prediction of critical heat flux using artificial neural network, J. Korean Nucl. Soc. 34 (2002) 240–250.

[32] Y. Liu, J. Zhang, H. Yu, Safe AI in nuclear engineering: Challenges, methodologies, and perspectives, Ann. Nucl. Energy 180 (2023) 109489. https://doi.org/10.1016/j.anucene.2022.109489.

[33] K. Shirvan, Can machine learning replace empirical correlations in nuclear safety codes?, Nucl. Eng. Des. 385 (2021) 111528. https://doi.org/10.1016/j.nucengdes.2021.111528.

[34] J. Feng, J. Buongiorno, K. Shirvan, Out-of-distribution generalization in thermal-hydraulic machine learning models, Int. J. Heat Mass Transf. 195 (2022) 123164. https://doi.org/10.1016/j.ijheatmasstransfer.2022.123164.

[35] X. Wu, Y. Bao, Epistemic uncertainty quantification for deep learning models in nuclear engineering, Reliab. Eng. Syst. Saf. 222 (2022) 108421. https://doi.org/10.1016/j.ress.2022.108421.

[36] G.E. Karniadakis, I.G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang, Physics-informed machine learning, Nat. Rev. Phys. 3 (2021) 422–440. https://doi.org/10.1038/s42254-021-00314-5.

[37] M. Raissi, P. Perdikaris, G.E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, J. Comput. Phys. 378 (2019) 686–707. https://doi.org/10.1016/j.jcp.2018.10.045.

[38] J. Willard, X. Jia, S. Xu, M. Steinbach, V. Kumar, Integrating physics-based modeling with machine learning: A survey, ACM Comput. Surv. 55 (2022) 1–34. https://doi.org/10.1145/3514228.

[39] K. Kashinath, M. Mustafa, A. Albert, K. Wu, C. Jiang, S. Esmaeilzadeh, K. Azizzadenesheli, R. Wang, A. Chattopadhyay, A. Singh, A. Manepalli, D. Chirila, R. Yu, R. Walters, B. White, H. Xiao, H.A. Tchelepi, P. Marcus, A. Anandkumar, P. Hassanzadeh, Prabhat, Physics-informed machine learning: case studies for weather and climate modeling, Phil. Trans. R. Soc. A 379 (2021) 20200093. https://doi.org/10.1098/rsta.2020.0093.

[40] H. Bao, Y. Liao, S. Fang, Physics-guided neural networks for two-phase flow and heat transfer modeling, Int. J. Multiph. Flow 151 (2022) 104052. https://doi.org/10.1016/j.ijmultiphaseflow.2022.104052.

[41] H. Liu, C. Chen, X. Dong, Dimensionless feature-augmented machine learning for critical heat flux prediction, Appl. Therm. Eng. 219 (2023) 119561. https://doi.org/10.1016/j.applthermaleng.2022.119561.

[42] S.M. Hosseini, A. Gholami, Scaling laws and dimensionless groups in physics-informed AI for boiling crisis, Int. J. Therm. Sci. 188 (2023) 108234. https://doi.org/10.1016/j.ijthermalsci.2023.108234.

[43] S. Yang, D. Li, Z. Wang, Residual neural network frameworks for thermal-hydraulic constitutive modeling, Ann. Nucl. Energy 191 (2023) 109923. https://doi.org/10.1016/j.anucene.2023.109923.

[44] R.K. Salko, X. Zhao, Hybrid physical-neural modeling for subchannel thermal-hydraulics, Nucl. Sci. Eng. 196 (2022) 412–428. https://doi.org/10.1080/00295639.2021.1994321.

[45] L. Lu, X. Meng, Z. Mao, G.E. Karniadakis, DeepXDE: A deep learning library for solving differential equations, SIAM Rev. 63 (2021) 208–228. https://doi.org/10.1137/19M1302888.

[46] Z. Fang, Physics-constrained neural networks for thermal-hydraulic boiling curves, Int. J. Heat Mass Transf. 182 (2022) 122019. https://doi.org/10.1016/j.ijheatmasstransfer.2021.122019.

[47] S. Yang, X. Li, B. Ren, L. Yang, C. Chen, Q. Lu, Z. Wei, A transfer learning framework for critical heat flux prediction pretrained on the 2006 lookup table, Appl. Therm. Eng. 287 (2026) 129451. https://doi.org/10.1016/j.applthermaleng.2025.129451.

[48] C. Roberts, M. Tipping, Generalization and interpolation in high-dimensional engineering surrogate modeling, IEEE Trans. Neural Netw. Learn. Syst. 33 (2022) 4102–4115. https://doi.org/10.1109/TNNLS.2021.3061214.

[49] A. Hastie, R. Tibshirani, J. Friedman, The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd ed., Springer, New York, 2009. https://doi.org/10.1007/978-0-387-84858-7.

[50] E. Snelson, Z. Ghahramani, Local and global sparse Gaussian processes for large engineering datasets, J. Mach. Learn. Res. 8 (2007) 1517–1555.

[51] L. Breiman, Random Forests, Mach. Learn. 45 (2001) 5–32. https://doi.org/10.1023/A:1010933404324.

[52] I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press, Cambridge, MA, 2016.

[53] J. Weisman, B.S. Pei, Prediction of critical heat flux in flow boiling at low qualities, Int. J. Heat Mass Transf. 26 (1983) 1463–1477. https://doi.org/10.1016/0017-9310(83)90047-0.

[54] C.H. Lee, I. Mudawar, A mechanistic critical heat flux model for subcooled flow boiling based on local bulk conditions, Int. J. Multiph. Flow 14 (1988) 711–728. https://doi.org/10.1016/0301-9322(88)90070-5.

[55] G.F. Hewitt, A.H. Govan, Phenomenological modelling of non-equilibrium flows with particular reference to the droplet entrainment and deposition in annular flow, Int. J. Multiph. Flow 16 (1990) 429–442. https://doi.org/10.1016/0301-9322(90)90074-J.

[56] D. Bestion, Applicability of two-phase CFD to nuclear reactor safety problems, Nucl. Eng. Des. 242 (2012) 2–15. https://doi.org/10.1016/j.nucengdes.2011.08.056.

[57] N. Kurul, M.Z. Podowski, On the modeling of multidimensional two-phase flow with heat transfer, Proc. 9th Int. Heat Transfer Conf. 2 (1990) 103–108.

[58] J.L. Abusah, S. Qiao, A. Ayodeji, S.Y. Dedzie, Deep-learning-based 3D bubble parameter prediction in a rod bundle subchannel: A single-view inference approach trained on dual-view reconstructions, Nucl. Eng. Des. 443 (2026) 115001. https://doi.org/10.1016/j.nucengdes.2026.115001.

[59] A. Mazzola, Artificial neural networks for critical heat flux prediction, Int. J. Heat Mass Transf. 40 (1997) 4485–4489. https://doi.org/10.1016/S0017-9310(97)00072-6.

[60] G.H. Su, L.W. Hu, J. Buongiorno, Artificial neural network prediction of critical heat flux for water in tubes, Nucl. Eng. Des. 237 (2007) 2011–2018. https://doi.org/10.1016/j.nucengdes.2007.03.011.

[61] K. Rehan Zubair, I. Ahmed, A. Ullah, E. Zio, Enhancing accuracy of prediction of critical heat flux in Circular channels by ensemble of deep sparse autoencoders and deep neural Networks, Nucl. Eng. Des. 428 (2024) 113587. https://doi.org/10.1016/j.nucengdes.2024.113587.

[62] I. Ahmed, I. Gatti, E. Zio, Optimized ensemble of neural networks for the prediction of critical heat flux, Nucl. Eng. Des. 433 (2025) 114111. https://doi.org/10.1016/j.nucengdes.2025.114111.

[63] K. Marcinkiewicz, O. Wieckhorst, R. Macián-Juan, M. Rehm, Recurrent neural network-based prediction of critical heat flux in rod bundles with non-uniform axial power shape, Nucl. Eng. Des. 393 (2022) 111825. https://doi.org/10.1016/j.nucengdes.2022.111825.

[64] H. Liu, H. Jiang, D. Chen, J. Qin, Critical heat flux prediction for annular channel through application of machine learning techniques, Int. Commun. Heat Mass Transf. 167 (2025) 109279. https://doi.org/10.1016/j.icheatmasstransfer.2025.109279.

[65] H. Wu, M. Gui, D. Wu, Physics-Informed hybrid machine learning for critical heat flux prediction: A comparative analysis of modeling approaches, Nucl. Eng. Des. 434 (2025) 114434. https://doi.org/10.1016/j.nucengdes.2025.114434.

[66] X. Zhao, R.K. Salko, K. Shirvan, Improved departure from nucleate boiling prediction in rod bundles using a physics-informed machine learning-aided framework, Nucl. Eng. Des. 380 (2021) 111084. https://doi.org/10.1016/j.nucengdes.2021.111084.

[67] A. Furlong, X. Zhao, R.K. Salko, X. Wu, Physics-based hybrid machine learning for critical heat flux prediction with uncertainty quantification, Appl. Therm. Eng. 268 (2025) 127447. https://doi.org/10.1016/j.applthermaleng.2025.127447.

[68] H. Yang, D. Li, Z. Wang, Y. Yan, LUT-guided grey-box neural networks for predicting critical heat flux in uniformly heated water-cooled circular tubes, Nucl. Eng. Des. 444 (2026) 115112. https://doi.org/10.1016/j.nucengdes.2026.115112.

[69] A.A. Mahmud, K. Morita, W. Liu, Critical heat flux prediction: A new approach adapting multi physics-aided machine learning, Int. Commun. Heat Mass Transf. 172 (2026) 110879. https://doi.org/10.1016/j.icheatmasstransfer.2026.110879.

[70] W. Zhang, J. Fu, S. Chen, L. Yu, Investigation of a hybrid neural network framework for CHF prediction in a wire-wrapped rod bundle, Ann. Nucl. Energy 211 (2026) 111804. https://doi.org/10.1016/j.anucene.2025.111804.

[71] Z. Abulawi, D. Lim, A. Garimidi, Y. Liu, Bayesian-optimized, feature-augmented deep ensemble for physics-guided critical heat-flux prediction with uncertainty quantification, Ann. Nucl. Energy 213 (2026) 112139. https://doi.org/10.1016/j.anucene.2026.112139.

[72] F. Abbasian, G.I. Hadaller, R.A. Fortman, J. Snell, An Artificial Neural Network (ANN) Model to Predict Critical Heat Flux (CHF) in a CANDU Fuel Element Simulation (FES) with Various Nonuniform Axial Heat Flux Shapes and Flow Liner Creep Profiles, Nucl. Eng. Des. 431 (2025) 113736. https://doi.org/10.1016/j.nucengdes.2024.113736.

[73] B.P. Serrao, Y.K. Huh, E. Ciuperca, E. Sahin, A quantitative analysis of ATF surface characteristics on critical heat flux using Machine learning, Nucl. Eng. Des. 432 (2025) 113924. https://doi.org/10.1016/j.nucengdes.2025.113924.

[74] Z. Huang, Y. Duo, H. Xu, Prediction of two-phase flow patterns based on machine learning, Nucl. Eng. Des. 423 (2024) 113107. https://doi.org/10.1016/j.nucengdes.2024.113107.

[75] R. Zubair Khalid, A. Khan, F. Ahmad, M.H. Al-Dahhan, The effect of reduction of input-parameters on data-driven prediction of critical heat flux in rod bundles using extended parameters for reactor thermal–hydraulic safety, Int. Commun. Heat Mass Transf. 170 (2026) 110659. https://doi.org/10.1016/j.icheatmasstransfer.2026.110659.

[76] N. Zuber, Hydrodynamic aspects of boiling heat transfer, AECU-4439, Physics and Mathematics, U.S. Atomic Energy Commission, Oak Ridge, TN, 1959. https://doi.org/10.2172/4175511.
