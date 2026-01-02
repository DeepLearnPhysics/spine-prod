# Summary of FSD full chain configurations and their characteristics

The configurations below have been trained on ND-LAr (not FSD!) MPV/MPR datasets. This summary is divided by training/validation dataset.

## Configuration Structure

All FSD configurations follow a hierarchical include structure to minimize duplication:

- **fsd_base.cfg**: Common base configuration with all shared settings (IO loaders, models, post-processing)
- **fsd_data_mod.cfg**: Data modifier that removes truth labels and adapts for real data reconstruction
- **fsd_full_chain_250505.cfg**: Version-specific weights and overrides, includes base config
- **fsd_full_chain_data_250505.cfg**: Includes full_chain_250505.cfg + data_mod.cfg for data processing

**Inheritance chain:**
```
fsd_full_chain_data_250505.cfg
  ↳ fsd_full_chain_250505.cfg
      ↳ fsd_base.cfg
  ↳ fsd_data_mod.cfg
```

This reduces total lines from ~1000 to ~530 (47% reduction) and eliminates duplication across variants.

**Runtime composition:**
```shell
python submit.py fsd_full_chain_250505.cfg --apply-mods data  # Equivalent to fsd_full_chain_data_250505.cfg
python submit.py fsd_full_chain_250505.cfg --list-mods        # Show available modifiers
```

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
