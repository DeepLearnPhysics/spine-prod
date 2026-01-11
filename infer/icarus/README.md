# Summary of ICARUS full chain configurations and their characteristics

The configurations below have been trained on ICARUS MPV/MPR datasets. This summary is divided by training/validation dataset.

## Directory Structure

The ICARUS configurations now follow a modular structure where each full chain configuration is built from reusable components:

- **`base/`**: Common base configurations including detector geometry and builder settings
- **`io/`**: Input/output configurations defining data schema and dataset parameters
- **`model/`**: Model architecture and weights paths for different training iterations
- **`modifier/`**: Configuration modifiers that transform base configs (e.g., data-only mode)
- **`post/`**: Post-processing configurations including flash matching and analysis modules
- **`legacy/`**: Archived configurations for backward compatibility

Each top-level configuration file (e.g., `icarus_full_chain_co_250625.yaml`) includes the appropriate modular components to build the complete chain. This structure makes it easier to:
- Mix and match components from different versions
- Update individual parts without duplicating settings
- Maintain consistency across similar configurations

### Configuration Composition

Features that previously required separate config files (e.g., `*_data_*`, `*_lite_*`, `*_numi_*`) are now handled through:

1. **Modifiers** in the `modifier/` directory that can be applied to any base configuration
2. **CLI options** when running inference (e.g., `--data`, `--lite`, `--beam-window`)
3. **Include statements** that compose functionality from different modules

For example:
- Data-only mode (no truth labels): Apply `modifier/data/mod_data_*.yaml` or use CLI flag
- Lite output: Use CLI flag for direct lite file output
- NuMI beam window: Specify beam window timing via CLI or post-processing config
- Single cryostat: Use appropriate geometry configuration
- Calibration variations (4ms, 8ms, transparency): Handled upstream or via modifiers

Legacy `.yaml` files have been moved to the `legacy/` directory.

## Configurations for MPV/MPR v02

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v2/train_file_list.txt`
- Test set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v2/test_file_list.txt`

### July 19th 2024

```shell
icarus_full_chain_240719.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions

**Note:** Legacy `.yaml` files with all naming variations are in `legacy/`. Features like data-only mode, NuMI beam windows, and single cryostat processing are now handled through modular composition and CLI options.

Known issue(s):
  - The shower start point prediction of electron showers is problematic due to the way PPN labeling is trained

### August 12th 2024

```shell
icarus_full_chain_240812.yaml
icarus_full_chain_co_240812.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions
  - Collection charge configurations available

**Note:** Legacy `.yaml` files with all naming variations are in `legacy/`. Features like data-only mode, NuMI beam windows, and single cryostat processing are now handled through modular composition and CLI options.

Known issue(s):
  - Resolves the issue with the PPN target in the previous set of weights
  - Removed PPN-based end point predictions
  - The signal gain on the first induction plane is wrong (`_fitFR` fcl file)
  - No other known issue


## Configurations for MPV/MPR v03

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v3/train_file_list.txt`
- Test set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v3/test_file_list.txt`

### January 15th 2025 

```shell
icarus_full_chain_250115.yaml
icarus_full_chain_co_250115.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions
  - Collection charge configurations available

**Note:** Legacy `.yaml` files with all naming variations are in `legacy/`. Features like data-only mode, lite output, NuMI beam windows, single cryostat processing, and unblind mode are now handled through modular composition and CLI options.

Known issue(s):
  - Resolves the issue with the first induction plane gain
  - Uses correct calibration constant (courtesy of Lane Kashur)
  - No other known issue

### March 3rd 2025 

```shell
icarus_full_chain_co_250303.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions
  - Collection charge configurations available
  - Calibration variations supported (4ms/8ms lifetimes, YZ transparency)

**Note:** Legacy `.yaml` files with all naming variations are in `legacy/`. Features like data-only mode, lite output, NuMI beam windows, single cryostat processing, calibration variations, and unblind mode are now handled through modular composition and CLI options.

Known issue(s):
  - Moved all calibrations upstream of the full chain (for the better!)
  - No other known issue


## Configurations for MPV/MPR v04

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v4/train_file_list.txt`
- Test set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v4/test_file_list.txt`

This dataset is a superset of v03, with the additional of samples with different
electron lifetimes (4 ms and 8 ms) to match what was recorded during ICARUS run 2.

### June 25th 2025 

```shell
icarus_full_chain_co_250625.yaml
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions
  - Collection charge configurations available
  - Calibration variations supported (4ms/8ms lifetimes, YZ transparency)

**Note:** Legacy `.yaml` files with all naming variations are in `legacy/`. Features like data-only mode, lite output, NuMI beam windows, single cryostat processing, calibration variations, low batch size, and unblind mode are now handled through modular composition and CLI options.

Changes:
  - Weights trained on a mixture of lifetimes (3, 4 and 8 ms)
  - Multiple shower quality cuts added (for nue analysis)
    - Cut on Michel KE of 100 MeV (anything above that is not a Michel)
    - Shower start merge (merge start with track stubs, if present)
    - Shower start point correction (based on vertex distance)
    - Exclude EM activity from containment checks
    - 50 MeV visibility threshold on electron showers (for topology)
    - Compute start dE/dx
    - Compute start straightness
    - Compute particle spread
    - Compute EM shower conversion distance
  - Use true energy depoisitions (SED) for the contaiment check
  - Use the vertex to update the track orientations (fix track flipping)
