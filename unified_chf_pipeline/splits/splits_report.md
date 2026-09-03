# Split Strategy Report

## Strategy 1: Random split (stratified by source)

```
split                   test  train
source_dataset                     
helical_coil_r123         51    206
kaeri_nonuniform         178    710
kaeri_uniform            130    521
mentor_master             11     44
nrc_groeneveld_24579pt  4916  19663
pinfin_chf_water_fc72     35    140
zhao2020                 373   1492
```
## Strategy 2: Condition-wise split (per-source top-pressure holdout)

```
split                   test  train
source_dataset                     
helical_coil_r123         53    204
kaeri_nonuniform         180    708
kaeri_uniform            169    482
mentor_master              0     55
nrc_groeneveld_24579pt  5460  19119
pinfin_chf_water_fc72      0    175
zhao2020                 757   1108
```
## Strategy 3: Surface-wise split (held-out sources: ['pinfin_chf_water_fc72', 'helical_coil_r123'])

```
split                   test  train
source_dataset                     
helical_coil_r123        257      0
kaeri_nonuniform           0    888
kaeri_uniform              0    651
mentor_master              0     55
nrc_groeneveld_24579pt     0  24579
pinfin_chf_water_fc72    175      0
zhao2020                   0   1865
```
## Strategy 4: Leave-one-source-out (fold == held-out source; counts below are rows per fold, i.e. rows per source)

```
fold                    helical_coil_r123  kaeri_nonuniform  kaeri_uniform  mentor_master  nrc_groeneveld_24579pt  pinfin_chf_water_fc72  zhao2020
source_dataset                                                                                                                                    
helical_coil_r123                     257                 0              0              0                       0                      0         0
kaeri_nonuniform                        0               888              0              0                       0                      0         0
kaeri_uniform                           0                 0            651              0                       0                      0         0
mentor_master                           0                 0              0             55                       0                      0         0
nrc_groeneveld_24579pt                  0                 0              0              0                   24579                      0         0
pinfin_chf_water_fc72                   0                 0              0              0                       0                    175         0
zhao2020                                0                 0              0              0                       0                      0      1865
```
