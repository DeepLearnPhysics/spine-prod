# Summary of DUNE10kt-1x2x6 full chain configurations and their characteristics

**NOTE: These configurations use a modular YAML include system. See the "Configuration Structure" section below for details.**

The configurations below are for DUNE10kt-1x2x6 datasets. This summary is divided by training/validation dataset and configuration version.

## Configuration Structure

All DUNE10kt-1x2x6 configs use a **hierarchical YAML include system** with composable components:

### Main Configurations
- **`full_chain_260202.yaml`**: January 2026, main DUNE10kt-1x2x6 full chain config

### Component Structure
Each main config includes modular YAML files:

**Base Components:**
- **`base/base_260202.yaml`**: Geometry and base settings (Jan 2026)
- **`base/base_common.yaml`**: Common base configuration

**IO Components:**
- **`io/io_260202.yaml`**: IO configuration (Jan 2026)
- **`io/io_common.yaml`**: Common IO settings

**Model Components:**
- **`model/model_260202.yaml`**: Jan 2026 weights and model settings
- **`model/model_260202.yaml`**: Jan 2026 (v2) weights and model settings
- **`model/model_common.yaml`**: Common model architecture

**Post-processing Components:**
- **`post/post_260202.yaml`**: Post-processing configuration (Jan 2026)
- **`post/post_common.yaml`**: Common post-processing settings

### Modifiers
Located in `modifier/` subdirectories:
- **`data/mod_data_260202.yaml`**: Data-only mode (no truth labels)
- **`lite/mod_lite_260202.yaml`**: Lite mode (reduced output)

### Legacy Configs
No legacy configs are present in this directory. Add `.cfg` files to a `legacy/` subdirectory if needed for backward compatibility.

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
