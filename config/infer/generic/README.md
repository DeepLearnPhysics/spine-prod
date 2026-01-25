
# Summary of Generic full chain configurations and their characteristics

**NOTE: These configurations have been refactored to YAML format using modular includes. See the "Configuration Structure" section below for details.**

The configurations below have been trained on generic datasets (no detector simulation). This summary is divided by training/validation dataset.

## Configuration Structure

All generic configs now use a **hierarchical YAML include system** with composable components:

### Main Configurations
- **`full_chain_240718.yaml`**: July 2024, MPV/MPR v04 weights
- **`full_chain_240805.yaml`**: August 2024, updated architecture and training

### Component Structure
Each main config includes modular YAML files:

**Base Components:**
- **`base/base_240718.yaml`**: Geometry and base settings (July 2024)
- **`base/base_common.yaml`**: Common base configuration

**IO Components:**
- **`io/io_240718.yaml`**: IO configuration (July 2024)
- **`io/io_240805.yaml`**: IO configuration (August 2024)
- **`io/io_common.yaml`**: Common IO settings

**Model Components:**
- **`model/model_240718.yaml`**: July 2024 weights and model settings
- **`model/model_240805.yaml`**: August 2024 weights and model settings
- **`model/model_common.yaml`**: Common model architecture

**Post-processing Components:**
- **`post/post_240718.yaml`**: Post-processing configuration (July 2024)
- **`post/post_common.yaml`**: Common post-processing settings

### Modifiers
Located in `modifier/` subdirectories:
- **`data/mod_data_240718.yaml`**: Data-only mode (no truth labels)
- **`ovl/mod_ovl_240718.yaml`**: Overlay mode (if available)
- **`lite/mod_lite_240718.yaml`**: Lite mode (reduced output)

### Legacy Configs
Old `.cfg` files moved to `legacy/` directory for backward compatibility

## Configurations for MPV/MPR v04

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/generic/mpvmpr_2020_01_v04/train.root`
- Test set: `/sdf/data/neutrino/generic/mpvmpr_2020_01_v04/test.root`

### July 18th 2024

```shell
full_chain_240718.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Modular YAML structure with base/io/model/post components

Known issue(s):
  - The shower start point prediction of electron showers is problematic due to the way PPN labeling is trained

### August 5th 2024

```shell
full_chain_240805.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Updates since the previous iteration:
    - Removed end-point prediction from PPN (not used)
    - Restricted the PPN mask to voxels within the particle cluster (improve semantic + PPN)
    - Switched from kNN to radius graph for Graph-SPICE
    - Increased the width of the GrapPA-inter MLP from 64 to 128
  - Modular YAML structure with base/io/model/post components

Known issue(s):
  - Nothing obvious
