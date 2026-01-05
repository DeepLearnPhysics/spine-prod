# Summary of 2x2 full chain configurations and their characteristics

**NOTE: These configurations have been refactored to use the `include:` mechanism to avoid duplication. See the "Configuration Structure" section below for details.**

The configurations below have been trained on 2x2 MPV/MPR datasets. This summary is divided by training/validation dataset.

## Configuration Structure

All 2x2 configs now use a **hierarchical include system** with composable modifiers:

### Base Configuration
- **`2x2_base.cfg`**: Main base config with all common settings (IO, model architecture, post-processing)

### Modifier Configs
- **`2x2_data_mod.cfg`**: Transforms any config to data-only mode (removes truth labels, sets reco-only)
- **`2x2_flash_mod.cfg`**: Adds flash parsing to any config

### Version-Specific Configs
- **`2x2_full_chain_240819.cfg`**: Latest v2 weights → `2x2_base.cfg`
- **`2x2_full_chain_240719.cfg`**: Older v1 weights + legacy settings → `2x2_base.cfg` + overrides

### Composed Configs
Variants built by combining base versions with modifiers:
- **`2x2_full_chain_flash_240819.cfg`**: `240819` + `flash_mod`
- **`2x2_full_chain_data_240819.cfg`**: `240819` + `data_mod`
- **`2x2_full_chain_data_flash_240819.cfg`**: `data_240819` + `flash_mod`
- (Same pattern for 240719 versions)

**Inheritance Example:**
```
2x2_full_chain_data_flash_240819.cfg
  ↳ 2x2_full_chain_data_240819.cfg
      ↳ 2x2_full_chain_240819.cfg
          ↳ 2x2_base.cfg
      ↳ 2x2_data_mod.cfg
  ↳ 2x2_flash_mod.cfg
```

### Latest Configs

Symlinks pointing to the most recent versions:
- `latest.cfg` → `2x2_full_chain_LATEST.cfg`
- `latest_data.cfg` → `2x2_full_chain_data_LATEST.cfg`
- `latest_flash.cfg` → `2x2_full_chain_flash_LATEST.cfg`
- `latest_data_flash.cfg` → `2x2_full_chain_data_flash_LATEST.cfg`

## Configurations for MPV/MPR v01

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/2x2/sim/mpvmpr_v1/train_file_list.txt`
- Test set: `/sdf/data/neutrino/2x2/sim/mpvmpr_v1/test_file_list.txt`

### July 19th 2024

```shell
2x2_full_chain_240719.cfg
2x2_full_chain_flash_240719.cfg
2x2_full_chain_data_240719.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - The `*_flash_*` declination includes flash parsing
  - The `*_data_*` declination is tailored for data (no labels)

Known issue(s):
  - Module 2 packets are simply wrong (performance in that module is terrible, may affect others)
  - The shower start point prediction of electron showers is problematic due to the way PPN labeling is trained

### August 19th 2024

```shell
2x2_full_chain_240819.cfg
2x2_full_chain_flash_240819.cfg
2x2_full_chain_data_240819.cfg
2x2_full_chain_data_flash_240819.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - The `*_flash_*` declination includes flash parsing
  - The `*_data_*` declination is tailored for data (no labels)
  - The `*_data_flash_*` declination is tailored for data (no labels) and includes flash parsing

Known issue(s):
  - Module 2 packets have been fixed w.r.t. to the previous set of weights
  - PPN labeling and predictions have been fixed
    - PPN mask labeling now only includes points within the cluster providing the label point
    - PPN no longer predicts track end point ordering
  - No known major issue
