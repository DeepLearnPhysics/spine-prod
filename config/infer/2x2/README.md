# Summary of 2x2 full chain configurations and their characteristics

**NOTE: These configurations have been refactored to YAML format using modular includes. See the "Configuration Structure" section below for details.**

The configurations below have been trained on 2x2 MPV/MPR datasets. This summary is divided by training/validation dataset.

## Configuration Structure

All 2x2 configs now use a **hierarchical YAML include system** with composable components:

### Main Configurations
- **`full_chain_240819.yaml`**: Latest August 2024 weights (v2)
- **`full_chain_240719.yaml`**: July 2024 weights (v1)

### Component Structure
Each main config includes modular YAML files:

**Base Components:**
- **`base/base_240719.yaml`**: Geometry and base settings
- **`base/base_common.yaml`**: Common base configuration

**IO Components:**
- **`io/io_240719.yaml`**: July 2024 IO (with PPN point tagging)
- **`io/io_240819.yaml`**: August 2024 IO (no PPN point tagging)
- **`io/io_common.yaml`**: Common IO settings

**Model Components:**
- **`model/model_240719.yaml`**: July 2024 model weights (v1)
- **`model/model_240819.yaml`**: August 2024 model weights (v2)
- **`model/model_common.yaml`**: Common model architecture

**Post-processing Components:**
- **`post/post_240719.yaml`**: Post-processing configuration
- **`post/post_common.yaml`**: Common post-processing settings

### Modifiers
Located in `modifier/` subdirectories:
- **`data/mod_data_240719.yaml`**: Data-only mode (no truth labels)
- **`lite/mod_lite_240719.yaml`**: Lite mode (reduced output)
- **`noflash/mod_noflash_240719.yaml`**: Disable flash parsing
- **`single/mod_single_240719.yaml`**: Switch to single-module mode
- **`single/mod_single_data_240719.yaml`**: Switch to single-module mode for data

### Legacy Configs
Old `.cfg` files moved to `legacy/` directory for backward compatibility

## Configurations for MPV/MPR v1

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/2x2/sim/mpvmpr_v1/train_file_list.txt`
- Test set: `/sdf/data/neutrino/2x2/sim/mpvmpr_v1/test_file_list.txt`

### July 19th 2024

```shell
full_chain_240719.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Uses PPN point tagging for shower start prediction
  - Modular YAML structure with base/io/model/post components
  - Modifiers available in `modifier/` for data-only, lite, and noflash modes

Known issue(s):
  - Module 2 packets are simply wrong (performance in that module is terrible, may affect others)
  - The shower start point prediction of electron showers is problematic due to the way PPN labeling is trained

## Configurations for MPV/MPR v2

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/2x2/sim/mpvmpr_v2/train_file_list.txt`
- Test set: `/sdf/data/neutrino/2x2/sim/mpvmpr_v2/test_file_list.txt`

### August 19th 2024

```shell
full_chain_240819.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Improved PPN labeling and predictions
  - Fixed Module 2 packet issues from previous version
  - PPN mask labeling now only includes points within the cluster providing the label point
  - PPN no longer predicts track end point ordering
  - Modular YAML structure with base/io/model/post components
  - Modifiers available in `modifier/` for data-only, lite, and noflash modes

Known issue(s):
  - No known major issues
