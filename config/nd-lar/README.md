# Summary of ND-LAr full chain configurations and their characteristics

The configurations below have been trained on ND-LAr MPV/MPR datasets. This summary is divided by training/validation dataset.

## Configuration Structure

All ND-LAr configurations follow a hierarchical include structure to minimize duplication:

- **nd-lar_base.cfg**: Common base configuration with all shared settings (IO loaders, models, post-processing)
- **nd-lar_ovl_mod.cfg**: Overlay modifier that adapts for overlay processing (4x spills)
- **nd-lar_full_chain_240819.cfg**: Frankenstein version using 2x2 weights (for benchmarking only)
- **nd-lar_full_chain_250505.cfg**: May 5th 2025 version with MPV/MPR v00 weights
- **nd-lar_full_chain_250515.cfg**: May 15th 2025 version with updated fiducial and edge lengths
- **nd-lar_full_chain_250806.cfg**: August 6th 2025 version trained on overlays
- **nd-lar_full_chain_ovl_250806.cfg**: Includes 250806 + ovl_mod for overlay processing

**Inheritance chain:**
```
nd-lar_base.cfg
  ↓ includes
nd-lar_full_chain_240819.cfg
nd-lar_full_chain_250505.cfg
nd-lar_full_chain_250515.cfg
nd-lar_full_chain_250806.cfg
  ↓ includes
nd-lar_full_chain_ovl_250806.cfg + nd-lar_ovl_mod.cfg
```

This eliminates duplication across date declinations.

**Runtime composition:**
```shell
python submit.py nd-lar_full_chain_250806.cfg --apply-mods ovl  # Equivalent to nd-lar_full_chain_ovl_250806.cfg
python submit.py nd-lar_full_chain_250806.cfg --list-mods       # Show available modifiers
```

## Frankenstein configurations

These configurations have been trained using 2x2 and are not expected to work properly. These configurations are to be exclusively used to benchmark SPINE's resource usage at ND-LAr but are not to be used to benchmark reconstruction performance or produce physics results.

### August 19th 2024

```shell
nd-lar_full_chain_flash_nersc_240819.cfg
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

### May 5th 2025

```shell
nd-lar_full_chain_250505.cfg
nd-lar_full_chain_nersc_250505.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Includes flash parsing
  - The `*_nersc_*` declination points to a weight path at NERSC (as opposed to S3DF)
  - The `*_data_*` declination is meant to run on real data (no labels)

Known issues:
  - No flash matching

### May 15th 2025

```shell
nd-lar_full_chain_250515.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Includes flash parsing

Known issues:
  - No flash matching

Changes:
  - Changed fiducial definition (10 cm from module boundaries now)
  - Relaxed maximum edge lengths in the GrapPAs

### August 6th 2025

```shell
nd-lar_full_chain_250806.cfg
nd-lar_full_chain_ovl_250806.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Includes flash parsing
  - The `*_ovl_*` declination is meant to run overlays (4x spills)

Known issues:
  - No flash matching

Changes:
  - Trained on overlays (4x images per event)
  - Improves interaction clusering purity as some cost to efficiency
