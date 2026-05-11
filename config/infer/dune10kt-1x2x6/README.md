# Summary of DUNE10kt-1x2x6 full chain configurations and their characteristics

**NOTE: These configurations use a modular YAML include system. See the "Configuration Structure" section below for details.**

The configurations below are for DUNE10kt-1x2x6 datasets. This summary is divided by training/validation dataset and configuration version.

## Configuration Structure

All DUNE10kt-1x2x6 configs use a **hierarchical YAML include system** with composable components:

### Main Configurations
- **`full_chain_260510.yaml`**: May 2026, latest DUNE10kt-1x2x6 full chain config
- **`full_chain_260202.yaml`**: February 2026, previous DUNE10kt-1x2x6 full chain config

### Component Structure
Each main config includes modular YAML files:

**Base Components:**
- **`base/base_260202.yaml`**: Geometry and base settings (Jan 2026)
- **`base/base_common.yaml`**: Common base configuration

**IO Components:**
- **`io/io_260202.yaml`**: IO configuration (Jan 2026)
- **`io/io_common.yaml`**: Common IO settings

**Model Components:**
- **`model/model_260510.yaml`**: May 2026 weights and model settings
- **`model/model_260202.yaml`**: Feb 2026 weights and model settings
- **`model/model_common.yaml`**: Common model architecture

**Post-processing Components:**
- **`post/post_260202.yaml`**: Post-processing configuration (Jan 2026)
- **`post/post_common.yaml`**: Common post-processing settings

### Modifiers
Located in `modifier/` subdirectories:
- **`data/mod_data_260202.yaml`**: Data-only mode (no truth labels)
- **`lite/mod_lite_260202.yaml`**: Lite mode (reduced output)

### Legacy Configs
No legacy configs are present in this directory. Add legacy YAML configs to a `legacy/` subdirectory if needed for backward compatibility.

## Configurations for MPV/MPR v01

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/dune/sim/mpvmpr_v1/train_file_list.txt`
- Test set: `/sdf/data/neutrino/dune/sim/mpvmpr_v1/test_file_list.txt`

This training set has a set of better-tuned ghost labels.

## May 10th 2026

```shell
full_chain_260510.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Modular YAML structure with base/io/model/post components

Changes:
  - Fixed the drift velocity issue
  - Trained much longer


## Configurations for MPV/MPR v00

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/dune/sim/mpvmpr_v0/train_file_list.txt`
- Test set: `/sdf/data/neutrino/dune/sim/mpvmpr_v0/test_file_list.txt`

Known issue(s):
  - Poorly tuned ghost labeling

## February 2nd 2026

```shell
full_chain_260202.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Modular YAML structure with base/io/model/post components

Known issue(s):
  - Undertrained (deghosting + transfer) in order to converge for the workshop
  - Mistake in the drift velocity, which is set x10 too large (undercorrecting for lifetime)

---

*For more details on each component, see the corresponding YAML files in the subdirectories.*
