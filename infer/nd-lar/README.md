# Summary of ND-LAr full chain configurations and their characteristics

**NOTE: These configurations have been refactored to YAML format using modular includes. See the "Configuration Structure" section below for details.**

The configurations below have been trained on ND-LAr MPV/MPR datasets. This summary is divided by training/validation dataset.

## Configuration Structure

All ND-LAr configs now use a **hierarchical YAML include system** with composable components:

### Main Configurations
- **`full_chain_240819.yaml`**: Frankenstein version using 2x2 weights (benchmarking only)
- **`full_chain_250505.yaml`**: May 2025 with MPV/MPR v00 weights
- **`full_chain_250515.yaml`**: May 2025 with updated fiducial and edge lengths
- **`full_chain_250806.yaml`**: August 2025 trained on overlays

### Component Structure
Each main config includes modular YAML files:

**Base Components:**
- **`base/base_240819.yaml`**: Geometry and base settings
- **`base/base_common.yaml`**: Common base configuration

**IO Components:**
- **`io/io_240819.yaml`**: IO configuration
- **`io/io_common.yaml`**: Common IO settings

**Model Components:**
- **`model/model_240819.yaml`**: 2x2 weights (debug/benchmark only)
- **`model/model_250505.yaml`**: May 5 2025 ND-LAr weights (v0)
- **`model/model_250515.yaml`**: May 15 2025 ND-LAr weights (v0)
- **`model/model_250806.yaml`**: August 6 2025 ND-LAr weights (v0, overlay training)
- **`model/model_common.yaml`**: Common model architecture

**Post-processing Components:**
- **`post/post_240819.yaml`**: Post-processing configuration
- **`post/post_common.yaml`**: Common post-processing settings

### Modifiers
Located in `modifier/` subdirectories:
- **`ovl/mod_ovl_240819.yaml`**: Overlay mode (4x spills)
- **`lite/mod_lite_240819.yaml`**: Lite mode (reduced output)

### Legacy Configs
Old `.cfg` files moved to `legacy/` directory for backward compatibility

## Frankenstein configurations

These configurations have been trained using 2x2 weights and are not expected to work properly. These configurations are to be exclusively used to benchmark SPINE's resource usage at ND-LAr but are not to be used to benchmark reconstruction performance or produce physics results.

### August 19th 2024

```shell
full_chain_240819.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Uses 2x2 weights for initial testing/benchmarking
  - Modular YAML structure with base/io/model/post components

Known issue(s):
  - This set of weights is not appropriate for ND-LAr but will run (as a test)
  - No flash matching


## Configurations for MPV/MPR v00

These configurations have been trained on the first MPV/MPR simulation produced in the ND-LAr geometry.

Known issues with the simulation:
  - Idealized simulation (almost no induction)
  - Wrong KE range produced for protons (and possibly electrons)
  - Unrealistically low pile-up (interaction multiplicity)

### May 5th 2025

```shell
full_chain_250505.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - First ND-LAr trained weights (MPV/MPR v00)
  - Modular YAML structure with base/io/model/post components
  - Modifiers available in `modifier/` for overlay and lite modes

Known issues:
  - No flash matching

### May 15th 2025

```shell
full_chain_250515.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Improved configuration with updated fiducial and edge lengths
  - Modular YAML structure with base/io/model/post components
  - Modifiers available in `modifier/` for overlay and lite modes

Known issues:
  - No flash matching

Changes from May 5th:
  - Changed fiducial definition (10 cm from module boundaries now)
  - Relaxed maximum edge lengths in the GrapPAs

### August 6th 2025

```shell
full_chain_250806.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Trained on overlay data (4x spills)
  - Modular YAML structure with base/io/model/post components
  - Modifiers available in `modifier/` for overlay and lite modes

Known issues:
  - No flash matching

Changes from May 15th:
  - Trained on overlay samples (multiple interactions per readout window)

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
