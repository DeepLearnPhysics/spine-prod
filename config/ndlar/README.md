# Summary of ND-LAr full chain configurations and their characteristics

The configurations below have been trained on ND-LAr MPV/MPR datasets. This summary is divided by training/validation dataset.

## Frankenstein configurations

These configurations have been trained using 2x2 and are not expected to work properly. These configurations are to be exclusively used to benchmark SPINE's resource usage at ND-LAr but are not to be used to benchmark reconstruction performance or produce physics results.

### August 19th 2024

```shell
ndlar_full_chain_flash_nersc_240819.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Includes flash parsing
  - The `*_nersc_*` declination points to a weight path at NERSC (as opposed to S3DF)

Known issue(s):
  - This set of weights is not appropriate for ND-LAr... but it will run (as a test)
  - No flash matching


## Configurations for MPV/MPR v00

These configurations have been trained on the first MPV/MPR simulation produced in the ND-LAr geometry.

Known issues with the simulation:
  - Idealized simulation (almost no induction)
  - Wrong KE range produced for protons (and possibly electrons)
  - Unrealistically low pile-up (interaction multiplicity)

### May 5th 2024

```shell
ndlar_full_chain_250505.cfg
ndlar_full_chain_nersc_250505.cfg
ndlar_single_full_chain_250505.cfg
ndlar_single_full_chain_data_250505.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Includes flash parsing
  - The `*_nersc_*` declination points to a weight path at NERSC (as opposed to S3DF)
  - The `*_single_*` declination is meant to run on FSD (single module of ND-LAr)
  - The `*_data_*` declination is meant to run on real data (no labels)

Known issues:
  - No flash matching
