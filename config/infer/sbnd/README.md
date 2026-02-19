# Summary of SBND full chain configurations and their characteristics

The configurations below have been trained on SBND MPV/MPR datasets. This summary is divided by training/validation dataset.

## Directory Structure

The SBND configurations now follow a modular structure where each full chain configuration is built from reusable components:

- **`base/`**: Common base configurations including detector geometry and builder settings
- **`io/`**: Input/output configurations defining data schema and dataset parameters
- **`model/`**: Model architecture and weights paths for different training iterations
- **`modifier/`**: Configuration modifiers that transform base configs (e.g., data-only mode)
- **`post/`**: Post-processing configurations including flash matching and analysis modules
- **`legacy/`**: Archived configurations for backward compatibility

Each top-level configuration file (e.g., `sbnd_full_chain_co_250901.yaml`) includes the appropriate modular components to build the complete chain. This structure makes it easier to:
- Mix and match components from different versions
- Update individual parts without duplicating settings
- Maintain consistency across similar configurations

### Configuration Composition

Features that previously required separate config files (e.g., `*_data_*`) are now handled through:

1. **Modifiers** in the `modifier/` directory that can be applied to any base configuration
2. **CLI options** when running inference (e.g., `--data`, `--lite`)
3. **Include statements** that compose functionality from different modules

For example:
- Data-only mode (no truth labels): Apply `modifier/data/mod_data_*.yaml` or use CLI flag
- Lite output: Use CLI flag for direct lite file output

Legacy `.yaml files have been moved to the `legacy/` directory.

## Configurations for MPVMPR v02

Training samples MPVMPR using `sbndcode v10_04_01` which can be found [here](https://github.com/SBNSoftware/sbndcode/tree/v10_04_01) . The training samples are generated using the following fcls:
```
run_mpvmpr_sbnd.fcl
g4_sce_lite.fcl
detsim_sce_lite.fcl
reco1_mpvmpr.fcl
```

These weights have been trained using the following files at Polaris:
- Training set: `/lus/eagle/projects/neutrinoGPU/bearc/simulation/mpvmpr_v02/train/files.txt` (255k)
- Test set: `/lus/eagle/projects/neutrinoGPU/bearc/simulation/mpvmpr_v02/test/larcv/files.txt` (68k)

...using the configs in the [sbnd_spine_train](https://github.com/bear-is-asleep/sbnd_spine_train/tree/master) repo.

## February 18th 2026

```shell
sbnd_full_chain_co_260218.yaml
```

The following modifications were made w.r.t. the September 2025 configuration
- Removed X-Arapuca flash information parsing

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions

## September 1st 2025

```shell
sbnd_full_chain_co_250901.yaml
```

The following modifications were made w.r.t. the August configurations
- Changed interaction clustering mode to anchor to tracks
- Fix issues with the back-tracking (do not move cathode crossers)
- Change time containment and containment modules to exclude showers
- Update MCS resolution modeling using a Rayleigh mixture
- Minor update to the calibration constants

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions

**Note:** Legacy `.yaml` files with `*_data_*` naming are in `legacy/`. Data-only mode is now handled through modular composition and CLI options.

## August 18th 2025

```shell
sbnd_full_chain_co_250818.cfg
```

The following modifications were made w.r.t. the March configurations
- Add a logical containment post-processor
- Do not move cathode crosser depositions to fix the truth matching
- Restrict FM interaction candidates to those not clearly out-of-time

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions

**Note:** Legacy `.cfg` files with `*_data_*` naming are in `legacy/`. Data-only mode is now handled through modular composition and CLI options.

## March 28th 2025

```shell
sbnd_full_chain_co_250328.cfg
```

The following modifications were made to the `sbndcode` configuration:
- Ghost labeling parameters - [Supera PR #54](https://github.com/DeepLearnPhysics/Supera/pull/54)
- Doublets are used - [sbndcode PR #661](https://github.com/SBNSoftware/sbndcode/pull/661)
- Updated clock - [sbndcode PR #645](https://github.com/SBNSoftware/sbndcode/pull/645)
- `larwirecell` patch - [larwirecell PR #55](https://github.com/LArSoft/larwirecell/pull/55)

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions

**Note:** Legacy `.cfg` files with `*_data_*` naming are in `legacy/`. Data-only mode is now handled through modular composition and CLI options.

## Configurations for MPV/MPR v01

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/sbnd/simulation/mpvmpr_v01/train.list`
- Test set: `/sdf/data/neutrino/sbnd/simulation/mpvmpr_v01/test.list`

### July 20th 2024

```shell
sbnd_full_chain_240720.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions

**Note:** Legacy `.cfg` files with `*_data_*` naming are in `legacy/`. Data-only mode is now handled through modular composition and CLI options.

Known issue(s):
  - The shower start point prediction of electron showers is problematic due to the way PPN labeling is trained
  - Flashes but no flash matching

### August 14th 2024

```shell
sbnd_full_chain_240814.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions

**Note:** Legacy `.cfg` files with `*_data_*` naming are in `legacy/`. Data-only mode is now handled through modular composition and CLI options.

Known issue(s):
  - Resolves the issue with the PPN target in the previous set of weights
  - Removed PPN-based end point predictions
  - No other known issue

### September 18th 2024

```shell
sbnd_full_chain_240918.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions
  - This is the first configuration which **includes flash matching**

**Note:** Legacy `.yaml` files with `*_data_*` naming are in `legacy/`. Data-only mode is now handled through modular composition and CLI options.

Known issue(s):
  - Includes flash matching
