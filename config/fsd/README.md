# Summary of FSD full chain configurations and their characteristics

The configurations below have been trained on ND-LAr (not FSD!) MPV/MPR datasets. This summary is divided by training/validation dataset.

## Configurations for MPV/MPR v00

These configurations have been trained on the first MPV/MPR simulation produced in the ND-LAr (not FSD!) geometry.

Known issues with the simulation:
  - Idealized simulation (almost no induction)
  - Wrong KE range produced for protons (and possibly electrons)
  - Unrealistically low pile-up (interaction multiplicity)

### May 5th 2025

```shell
fsd_full_chain_250505.cfg
fsd_full_chain_data_250505.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Includes flash parsing
  - The `*_data_*` declination is meant to run on real data (no labels)

Known issues:
  - No flash matching
