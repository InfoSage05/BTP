# Physics Ablation Results

## Read this first

**Arm A1 is NOT literally today's pipeline.** It shares the latent-heat
baseline, but it also carries the Stage-0 data repair, FC-72 saturation
properties, and `subcooling_kJkg` in place of the 0.8%-coverage
`subcooling_K`. `attribution_test.py` separates those data fixes from the
modelling ideas; on strategy 3 they account for 0.173 -> 0.711 on their
own, before any physics idea is applied. Credit the data, not the models.

**A5's out-of-domain numbers are the physics, not the model.** Trust decay
drives the learned correction to zero for 100% of strategy-3 test rows, so
A5 there reduces exactly to `PHYS_gated`. That is the designed behaviour
(fall back to physics off-manifold) but it means A5 can only MATCH pure
physics out of domain, never beat it. The decay is currently binary rather
than graded -- softening `trust_gamma` is the obvious next experiment.

Each rung adds one idea to the rung above it, so the contribution of
each is isolated. `PHYS_*` arms are the closed-form physics with no
learning at all -- the reference every learned arm must beat.

Constraint columns come from `physics/constraints.py`; C3/C4/C6/S1/S6/S7
are measured by probing each trained model on synthetic sweeps it never
saw in training.

## Accuracy by split

### Strategy 1 - random (optimistic)

| arm                                          |     R2 |    RMSE |      MAE |   MAPE_pct |   train_seconds |
|:---------------------------------------------|-------:|--------:|---------:|-----------:|----------------:|
| Physics only (Katto-Ohno), no learning       | 0.7791 | 770.658 | 342.671  |    19.2132 |          0      |
| Physics only (gated), no learning            | 0.7328 | 847.537 | 380.82   |    21.9698 |          0      |
| A0  no physics, raw features                 | 0.9792 | 236.583 | 118.993  |     6.5017 |          1.789  |
| A1  latent-heat baseline  [today's pipeline] | 0.9699 | 284.435 | 129.213  |     6.5416 |          1.043  |
| A2  + Katto-Ohno baseline  [idea 1]          | 0.9776 | 245.544 |  95.3871 |     4.6714 |          1.222  |
| A3  + dimensionless features  [idea 2]       | 0.9817 | 221.994 |  79.2177 |     3.9107 |          1.9132 |
| A4  + mechanism gating  [idea 4]             | 0.9785 | 240.331 |  89.4429 |     4.1062 |          2.8074 |
| A5  + bounded/monotone/trust  [idea 5]       | 0.9574 | 338.416 | 130.486  |     7.1424 |          3.032  |
| A1-ANN  latent baseline, MLP                 | 0.9637 | 312.337 | 118.382  |     5.6696 |          6.0333 |
| A5-ANN  constrained, MLP                     | 0.9595 | 329.92  | 119.958  |     5.9908 |          8.1617 |

### Strategy 2 - condition-wise (pressure extrapolation)

| arm                                          |     R2 |    RMSE |     MAE |   MAPE_pct |   train_seconds |
|:---------------------------------------------|-------:|--------:|--------:|-----------:|----------------:|
| Physics only (Katto-Ohno), no learning       | 0.5075 | 904.735 | 423.981 |    23.775  |          0      |
| Physics only (gated), no learning            | 0.8085 | 564.167 | 335.491 |    24.561  |          0      |
| A0  no physics, raw features                 | 0.9033 | 400.825 | 228.071 |    16.9051 |          2.9557 |
| A1  latent-heat baseline  [today's pipeline] | 0.8771 | 451.861 | 259.346 |    15.9467 |          2.9886 |
| A2  + Katto-Ohno baseline  [idea 1]          | 0.8123 | 558.455 | 271.88  |    17.3154 |          3.045  |
| A3  + dimensionless features  [idea 2]       | 0.7897 | 591.185 | 280.69  |    16.2447 |          2.9005 |
| A4  + mechanism gating  [idea 4]             | 0.8266 | 536.755 | 256.734 |    15.4215 |          2.9837 |
| A5  + bounded/monotone/trust  [idea 5]       | 0.8538 | 492.953 | 281.85  |    21.745  |          1.4096 |
| A1-ANN  latent baseline, MLP                 | 0.645  | 768.085 | 454.609 |    29.6715 |          4.0924 |
| A5-ANN  constrained, MLP                     | 0.8005 | 575.727 | 305.142 |    21.7189 |          6.5219 |

### Strategy 3 - surface-wise (held-out surface types)

| arm                                          |        R2 |     RMSE |      MAE |   MAPE_pct |   train_seconds |
|:---------------------------------------------|----------:|---------:|---------:|-----------:|----------------:|
| Physics only (Katto-Ohno), no learning       |    0.5051 |  251.432 |  167.49  |    42.2757 |          0      |
| Physics only (gated), no learning            |    0.5063 |  251.121 |  165.042 |    43.8668 |          0      |
| A0  no physics, raw features                 |  -17.9965 | 1557.71  | 1145.14  |   585.286  |          5.6265 |
| A1  latent-heat baseline  [today's pipeline] |    0.7112 |  192.05  |  123.269 |    36.6459 |          3.3805 |
| A2  + Katto-Ohno baseline  [idea 1]          |    0.6057 |  224.415 |  160.709 |    52.3892 |          4.7619 |
| A3  + dimensionless features  [idea 2]       |    0.2519 |  309.127 |  213.97  |    63.7111 |          4.4002 |
| A4  + mechanism gating  [idea 4]             |    0.315  |  295.808 |  189.62  |    53.1503 |          3.4805 |
| A5  + bounded/monotone/trust  [idea 5]       |    0.5063 |  251.121 |  165.042 |    43.8668 |          3.2174 |
| A1-ANN  latent baseline, MLP                 | -362.957  | 6818.29  | 4291.92  |  1187.03   |          9.9292 |
| A5-ANN  constrained, MLP                     |    0.5063 |  251.121 |  165.042 |    43.8668 |         12.7559 |

### Strategy 4 - leave-one-source-out (pooled)

| arm                                          |      R2 |     RMSE |      MAE |   MAPE_pct | train_seconds   |
|:---------------------------------------------|--------:|---------:|---------:|-----------:|:----------------|
| Physics only (Katto-Ohno), no learning       |  0.7851 |  799.572 |  349.905 |    18.2686 | -               |
| Physics only (gated), no learning            |  0.7511 |  860.554 |  384.474 |    20.7873 | -               |
| A0  no physics, raw features                 |  0.7187 |  914.85  |  587.868 |    49.492  | -               |
| A1  latent-heat baseline  [today's pipeline] |  0.8003 |  770.796 |  499.258 |    30.5883 | -               |
| A2  + Katto-Ohno baseline  [idea 1]          | -0.797  | 2312.23  | 1187.48  |    57.1465 | -               |
| A3  + dimensionless features  [idea 2]       | -0.4667 | 2088.93  | 1180.44  |    59.7016 | -               |
| A4  + mechanism gating  [idea 4]             |  0.6198 | 1063.5   |  710.8   |    44.2859 | -               |
| A5  + bounded/monotone/trust  [idea 5]       |  0.7103 |  928.409 |  484.8   |    25.5119 | -               |
| A1-ANN  latent baseline, MLP                 | -4.1451 | 3912.47  | 1158.4   |   108.636  | -               |
| A5-ANN  constrained, MLP                     |  0.7201 |  912.585 |  491.86  |    26.9109 | -               |

## R2 across all splits

|                                              |   strategy1 |   strategy2 |   strategy3 |   strategy4_pooled_oof |
|:---------------------------------------------|------------:|------------:|------------:|-----------------------:|
| Physics only (Katto-Ohno), no learning       |      0.7791 |      0.5075 |      0.5051 |                 0.7851 |
| Physics only (gated), no learning            |      0.7328 |      0.8085 |      0.5063 |                 0.7511 |
| A0  no physics, raw features                 |      0.9792 |      0.9033 |    -17.9965 |                 0.7187 |
| A1  latent-heat baseline  [today's pipeline] |      0.9699 |      0.8771 |      0.7112 |                 0.8003 |
| A2  + Katto-Ohno baseline  [idea 1]          |      0.9776 |      0.8123 |      0.6057 |                -0.797  |
| A3  + dimensionless features  [idea 2]       |      0.9817 |      0.7897 |      0.2519 |                -0.4667 |
| A4  + mechanism gating  [idea 4]             |      0.9785 |      0.8266 |      0.315  |                 0.6198 |
| A5  + bounded/monotone/trust  [idea 5]       |      0.9574 |      0.8538 |      0.5063 |                 0.7103 |
| A1-ANN  latent baseline, MLP                 |      0.9637 |      0.645  |   -362.957  |                -4.1451 |
| A5-ANN  constrained, MLP                     |      0.9595 |      0.8005 |      0.5063 |                 0.7201 |

## MAPE % across all splits

|                                              |   strategy1 |   strategy2 |   strategy3 |   strategy4_pooled_oof |
|:---------------------------------------------|------------:|------------:|------------:|-----------------------:|
| Physics only (Katto-Ohno), no learning       |       19.21 |       23.78 |       42.28 |                  18.27 |
| Physics only (gated), no learning            |       21.97 |       24.56 |       43.87 |                  20.79 |
| A0  no physics, raw features                 |        6.5  |       16.91 |      585.29 |                  49.49 |
| A1  latent-heat baseline  [today's pipeline] |        6.54 |       15.95 |       36.65 |                  30.59 |
| A2  + Katto-Ohno baseline  [idea 1]          |        4.67 |       17.32 |       52.39 |                  57.15 |
| A3  + dimensionless features  [idea 2]       |        3.91 |       16.24 |       63.71 |                  59.7  |
| A4  + mechanism gating  [idea 4]             |        4.11 |       15.42 |       53.15 |                  44.29 |
| A5  + bounded/monotone/trust  [idea 5]       |        7.14 |       21.75 |       43.87 |                  25.51 |
| A1-ANN  latent baseline, MLP                 |        5.67 |       29.67 |     1187.03 |                 108.64 |
| A5-ANN  constrained, MLP                     |        5.99 |       21.72 |       43.87 |                  26.91 |

## Physics-consistency scorecard

Measured on strategy 3 (the hardest split). Lower is better for the
violation fractions; `S1_peak_reduced_pressure` should be near 0.35;
`S2_pool_K_median` should sit in the Zuber band 0.119-0.157.

| arm                                          |   C1_nonpositive_frac | C3_quality_violation_frac   | C4_massflux_violation_frac   | C6_satisfied   | S1_peak_reduced_pressure   | S6_sign_reversal_captured   | S7_crossover_captured   |   S2_pool_K_median |
|:---------------------------------------------|----------------------:|:----------------------------|:-----------------------------|:---------------|:---------------------------|:----------------------------|:------------------------|-------------------:|
| Physics only (Katto-Ohno), no learning       |                     0 | -                           | -                            | -              | -                          | -                           | -                       |              0.131 |
| Physics only (gated), no learning            |                     0 | -                           | -                            | -              | -                          | -                           | -                       |              0.131 |
| A0  no physics, raw features                 |                     0 | 0.042                       | 0.083                        | yes            | 0.223                      | no                          | no                      |              0.917 |
| A1  latent-heat baseline  [today's pipeline] |                     0 | 0.375                       | 0.125                        | no             | 0.201                      | no                          | no                      |              0.227 |
| A2  + Katto-Ohno baseline  [idea 1]          |                     0 | 0.583                       | 0.000                        | yes            | 0.223                      | no                          | no                      |              0.221 |
| A3  + dimensionless features  [idea 2]       |                     0 | 0.500                       | 0.000                        | yes            | 0.133                      | no                          | no                      |              0.095 |
| A4  + mechanism gating  [idea 4]             |                     0 | 0.333                       | 0.000                        | yes            | 0.088                      | no                          | yes                     |              0.105 |
| A5  + bounded/monotone/trust  [idea 5]       |                     0 | 0.125                       | 0.000                        | yes            | 0.313                      | no                          | yes                     |              0.131 |
| A1-ANN  latent baseline, MLP                 |                     0 | 0.958                       | 0.000                        | yes            | 0.065                      | no                          | no                      |             10.719 |
| A5-ANN  constrained, MLP                     |                     0 | 0.667                       | 0.000                        | yes            | 0.313                      | no                          | yes                     |              0.131 |

## Leave-one-source-out, per fold (R2)

|                                              |   helical_coil_r123 |   kaeri_nonuniform |   kaeri_uniform |   mentor_master |   nrc_groeneveld_24579pt |   pinfin_chf_water_fc72 |   zhao2020 |
|:---------------------------------------------|--------------------:|-------------------:|----------------:|----------------:|-------------------------:|------------------------:|-----------:|
| Physics only (Katto-Ohno), no learning       |               0.051 |              0.573 |           0.146 |          -0.902 |                    0.926 |                   0.285 |     -0.362 |
| Physics only (gated), no learning            |               0.093 |             -0.433 |           0.462 |          -0.902 |                    0.884 |                   0.285 |      0.22  |
| A0  no physics, raw features                 |            -678.452 |              0.015 |           0.743 |          -0.462 |                    0.769 |                   0.521 |      0.601 |
| A1  latent-heat baseline  [today's pipeline] |              -0.192 |              0.511 |           0.768 |           0.333 |                    0.839 |                   0.662 |      0.446 |
| A2  + Katto-Ohno baseline  [idea 1]          |              -2.407 |              0.323 |           0.623 |          -0.083 |                   -1.286 |                   0.571 |      0.207 |
| A3  + dimensionless features  [idea 2]       |              -1.902 |              0.324 |           0.721 |          -1.155 |                   -0.821 |                   0.087 |      0.077 |
| A4  + mechanism gating  [idea 4]             |              -0.901 |             -0.172 |           0.794 |          -0.896 |                    0.656 |                   0.055 |      0.4   |
| A5  + bounded/monotone/trust  [idea 5]       |               0.093 |             -0.6   |           0.722 |          -0.902 |                    0.829 |                   0.285 |      0.28  |
| A1-ANN  latent baseline, MLP                 |           -1568.68  |             -0.77  |           0.707 |          -0.628 |                    0.751 |               -3703.06  |    -36.404 |
| A5-ANN  constrained, MLP                     |               0.093 |             -0.577 |           0.649 |          -0.902 |                    0.851 |                   0.285 |      0.205 |

## Leave-one-source-out R2 by physical regime

DNB and dryout are physically distinct crises (foundation doc 1.2);
a single pooled number hides which mechanism a model actually handles.

| arm                                          |    DNB |   dryout |     pool |
|:---------------------------------------------|-------:|---------:|---------:|
| Physics only (Katto-Ohno), no learning       |  0.551 |    0.861 |    0.199 |
| Physics only (gated), no learning            |  0.484 |    0.834 |    0.199 |
| A0  no physics, raw features                 |  0.574 |    0.654 |    0.392 |
| A1  latent-heat baseline  [today's pipeline] |  0.69  |    0.76  |    0.71  |
| A2  + Katto-Ohno baseline  [idea 1]          | -2.782 |   -0.202 |    0.542 |
| A3  + dimensionless features  [idea 2]       | -1.961 |   -0.099 |    0.082 |
| A4  + mechanism gating  [idea 4]             |  0.535 |    0.424 |    0.177 |
| A5  + bounded/monotone/trust  [idea 5]       |  0.495 |    0.711 |    0.199 |
| A1-ANN  latent baseline, MLP                 | -1.121 |   -6.698 | -395.473 |
| A5-ANN  constrained, MLP                     |  0.468 |    0.764 |    0.199 |
