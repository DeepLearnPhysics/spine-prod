# Summary of 2x2-single full chain configurations and their characteristics

**NOTE: These configurations have been refactored to use the `include:` mechanism to avoid duplication. See the "Configuration Structure" section below for details.**

The configurations below have been trained on 2x2 MPV/MPR datasets. This summary is divided by training/validation dataset.

## Configuration Structure

All 2x2-single configs now use a **hierarchical include system** with composable modifiers:

### Base Configuration
- **`2x2-single_base.cfg`**: Main base config with all common settings (IO, model architecture, post-processing)

### Modifier Configs
- **`2x2-single_data_mod.cfg`**: Transforms any config to data-only mode (removes truth labels, sets reco-only)

### Version-Specific Configs
- **`2x2-single_full_chain_240819.cfg`**: Latest v2 weights → `2x2-single_base.cfg` + overrides

### Composed Configs
- **`2x2-single_full_chain_data_240819.cfg`**: `240819` + `data_mod`

**Inheritance Example:**
```
2x2-single_full_chain_data_240819.cfg
  ↳ 2x2-single_full_chain_240819.cfg
      ↳ 2x2-single_base.cfg
  ↳ 2x2-single_data_mod.cfg
```

**Runtime Composition:**
You can also compose configs at runtime using the orchestrator:
```bash
# Apply modifiers at runtime instead of using pre-composed configs
./submit.py --config infer/2x2-single/2x2-single_full_chain_240819.cfg \
            --files data.root --apply-mods data

# List available modifiers
./submit.py --list-mods infer/2x2-single/2x2-single_full_chain_240819.cfg
```

## Configurations for MPV/MPR v02

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/2x2/sim/mpvmpr_v2/train_file_list.txt`
- Test set: `/sdf/data/neutrino/2x2/sim/mpvmpr_v2/test_file_list.txt`

### August 19th 2024

```shell
2x2-single_full_chain_240819.cfg
2x2-single_full_chain_data_240819.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - The `*_data_*` variant is tailored for data (no labels)
  - Uses v2 weights with updated model settings

Known issue(s):
  - Module 2 packets have been fixed w.r.t. to the previous set of weights
  - PPN labeling and predictions have been fixed
    - PPN mask labeling now only includes points within the cluster providing the label point
    - PPN no longer predicts track end point ordering
  - No known major issue
