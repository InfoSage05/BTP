# Improvement experiments

Evaluated on strategies 3 and 4 only -- the splits where a whole source is
held out. See the module docstring in `run_improvements.py` for what each
experiment is testing and why.

## R2

| arm                         |   strategy3 |   strategy4_pooled_oof |
|:----------------------------|------------:|-----------------------:|
| PHYS_katto                  |      0.5051 |                 0.7851 |
| PHYS_gated                  |      0.5063 |                 0.7511 |
| REF_A1_latent_unbounded     |      0.7112 |                 0.8003 |
| REF_A2_katto_unbounded      |      0.6057 |                -0.797  |
| REF_A5_gated_pi_constrained |      0.5063 |                 0.7103 |
| E1_latent_raw_bounded       |     -0.8165 |                -0.0427 |
| E1_katto_raw_bounded        |      0.5051 |                 0.4894 |
| E1_katto_pi_bounded         |      0.5051 |                 0.6859 |
| E1_katto_raw_bound_notrust  |      0.6083 |                -0.5357 |
| E2_katto_raw_gamma0.25      |      0.5051 |                 0.154  |
| E2_katto_raw_gamma0.5       |      0.5051 |                 0.3505 |
| E2_katto_raw_gamma2.0       |      0.5051 |                 0.5647 |
| E3_stack_raw_unbounded      |      0.4865 |                 0.7155 |
| E3_stack_pi_unbounded       |      0.3285 |                 0.7183 |
| E3_stack_raw_bounded        |     -0.8165 |                 0.3538 |
| E3_stack_pi_bounded         |     -0.8165 |                 0.2779 |

## Leave-one-source-out R2 per fold

|                             |   helical_coil_r123 |   kaeri_nonuniform |   kaeri_uniform |   mentor_master |   nrc_groeneveld_24579pt |   pinfin_chf_water_fc72 |   zhao2020 |
|:----------------------------|--------------------:|-------------------:|----------------:|----------------:|-------------------------:|------------------------:|-----------:|
| PHYS_katto                  |               0.051 |              0.573 |           0.146 |          -0.902 |                    0.926 |                   0.285 |     -0.362 |
| PHYS_gated                  |               0.093 |             -0.433 |           0.462 |          -0.902 |                    0.884 |                   0.285 |      0.22  |
| REF_A1_latent_unbounded     |              -0.192 |              0.511 |           0.768 |           0.333 |                    0.839 |                   0.662 |      0.446 |
| REF_A2_katto_unbounded      |              -2.407 |              0.323 |           0.623 |          -0.083 |                   -1.286 |                   0.571 |      0.207 |
| REF_A5_gated_pi_constrained |               0.093 |             -0.6   |           0.722 |          -0.902 |                    0.829 |                   0.285 |      0.28  |
| E1_latent_raw_bounded       |              -0.916 |              0.531 |           0.42  |          -2.946 |                   -0.226 |                  -1.694 |     -0.117 |
| E1_katto_raw_bounded        |               0.051 |              0.599 |           0.473 |          -0.902 |                    0.487 |                   0.285 |     -0.13  |
| E1_katto_pi_bounded         |               0.051 |              0.716 |           0.435 |          -0.902 |                    0.75  |                   0.285 |     -0.163 |
| E1_katto_raw_bound_notrust  |              -1.826 |              0.484 |           0.479 |          -0.639 |                   -0.93  |                   0.564 |      0.132 |
| E2_katto_raw_gamma0.25      |               0.051 |              0.629 |           0.479 |          -0.902 |                    0.013 |                   0.286 |     -0.002 |
| E2_katto_raw_gamma0.5       |               0.051 |              0.623 |           0.477 |          -0.902 |                    0.288 |                   0.285 |     -0.061 |
| E2_katto_raw_gamma2.0       |               0.051 |              0.587 |           0.464 |          -0.902 |                    0.597 |                   0.285 |     -0.182 |
| E3_stack_raw_unbounded      |              -0.576 |              0.018 |           0.719 |           0.302 |                    0.811 |                  -0.569 |      0.054 |
| E3_stack_pi_unbounded       |              -0.555 |              0.18  |           0.728 |           0.024 |                    0.795 |                   0.064 |      0.119 |
| E3_stack_raw_bounded        |              -0.916 |              0.42  |           0.49  |          -2.946 |                    0.303 |                  -1.694 |      0.029 |
| E3_stack_pi_bounded         |              -0.916 |              0.446 |           0.511 |          -2.946 |                    0.205 |                  -1.694 |     -0.04  |

## Median trust weight on the strategy-3 test set

A value near 0 means the learned correction is switched off and the arm has
reduced to pure physics; near 1 means the correction is fully active.

| arm                         |   trust_weight_median |
|:----------------------------|----------------------:|
| PHYS_katto                  |              nan      |
| PHYS_gated                  |              nan      |
| REF_A1_latent_unbounded     |                1      |
| REF_A2_katto_unbounded      |                1      |
| REF_A5_gated_pi_constrained |                0      |
| E1_latent_raw_bounded       |                0      |
| E1_katto_raw_bounded        |                0      |
| E1_katto_pi_bounded         |                0      |
| E1_katto_raw_bound_notrust  |                1      |
| E2_katto_raw_gamma0.25      |                0.0001 |
| E2_katto_raw_gamma0.5       |                0      |
| E2_katto_raw_gamma2.0       |                0      |
| E3_stack_raw_unbounded      |                1      |
| E3_stack_pi_unbounded       |                1      |
| E3_stack_raw_bounded        |                0      |
| E3_stack_pi_bounded         |                0      |
