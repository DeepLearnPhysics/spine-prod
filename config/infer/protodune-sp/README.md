# Summary of ProtoDUNE-SP full chain configurations and their characteristics

**NOTE: These configurations use a modular YAML include system. See the "Configuration Structure" section below for details.**

The configurations below are for ProtoDUNE-SP datasets. This summary is divided by training/validation dataset and configuration version.

## Configuration Structure

All ProtoDUNE-SP configs use a **hierarchical YAML include system** with composable components:

### Main Configurations
- **`full_chain_260210.yaml`**: February 2026, latest ProtoDUNE-SP full chain config

### Component Structure
Each main config includes modular YAML files:

**Base Components:**
- **`base/base_260210.yaml`**: Geometry and base settings (Feb 2026)
- **`base/base_common.yaml`**: Common base configuration

**IO Components:**
- **`io/io_260210.yaml`**: IO configuration (Feb 2026)
- **`io/io_common.yaml`**: Common IO settings

**Model Components:**
- **`model/model_260210.yaml`**: Feb 2026 weights and model settings
- **`model/model_260128.yaml`**: Jan 2026 (v2) weights and model settings
- **`model/model_common.yaml`**: Common model architecture

**Post-processing Components:**
- **`post/post_260210.yaml`**: Post-processing configuration (Feb 2026)
- **`post/post_common.yaml`**: Common post-processing settings

### Modifiers
Located in `modifier/` subdirectories:
- **`data/mod_data_260210.yaml`**: Data-only mode (no truth labels)
- **`lite/mod_lite_260210.yaml`**: Lite mode (reduced output)

### Legacy Configs
No legacy configs are present in this directory. Add `.cfg` files to a `legacy/` subdirectory if needed for backward compatibility.

## February 10th 2026

```shell
full_chain_260210.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Modular YAML structure with base/io/model/post components
  - Latest production configuration

Known issue(s):
  - No charged kaon in the training sample

## January 18th 2026

```shell
full_chain_260128.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Modular YAML structure with base/io/model/post components

Known issue(s):
  - No charged kaon in the training sample

---

*For more details on each component, see the corresponding YAML files in the subdirectories.*
