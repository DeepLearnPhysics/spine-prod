# Changelog

All notable changes to the SPINE Production System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.2] - 2026-07-21

### Changed
- Made deprecated `submit.py --no-writer` a warning-only no-op. Automatic output options are always passed to SPINE, which safely ignores them when no writer is configured.

### Fixed
- Prevented `--no-writer` from suppressing the managed output directory when a configuration does contain a writer, which could place generated HDF5 files in the submission directory.

Full Changelog: [v0.7.1...v0.7.2](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.7.1...v0.7.2)

## [0.7.1] - 2026-07-20

### Changed
- Updated the default SPINE container release to `docker:ghcr.io/deeplearnphysics/spine:v0.15.3`.
- Updated the derived default S3DF Singularity image path to `/sdf/data/neutrino/images/spine_v0-15-3.sif`.
- Replaced generic `io.writer.*` overrides with SPINE's dedicated `--output`, `--output-dir`, and `--output-suffix` options.
- Deprecated `submit.py --no-writer`, which is no longer needed with SPINE v0.15.3 or newer.

### Fixed
- Prevented explicit-source jobs using configurations without an `io.writer` block from creating an incomplete writer configuration and crashing.

Full Changelog: [v0.7.0...v0.7.1](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.7.0...v0.7.1)

## [0.7.0] - 2026-07-17

### Added
- Expanded and reorganized the automated test suite to mirror the source layout and provide complete line and branch coverage for the production Python modules.

### Changed
- Updated the default SPINE container release to `docker:ghcr.io/deeplearnphysics/spine:v0.15.1`.
- Updated the derived default S3DF Singularity image path to `/sdf/data/neutrino/images/spine_v0-15-1.sif`.
- Declared reusable `*_common.yaml` inference configurations as metadata fragments, preventing missing-metadata warnings during composition.
- Declared versioned modifier configurations as metadata modifiers and the detector-independent litify configuration as a standalone bundle.
- Corrected stale configuration versions, dates, detector labels, and comments using dated filenames as the source of truth.
- Refreshed inference READMEs to match the available configurations, modifiers, and current submission CLI.
- Made configuration validation recognize expected fragment-metadata diagnostics from older SPINE releases without suppressing genuine warnings from fragment-aware releases.

Full Changelog: [v0.6.2...v0.7.0](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.6.2...v0.7.0)

## [0.6.2] - 2026-06-30

### Changed
- Updated the default SPINE container release to `docker:ghcr.io/deeplearnphysics/spine:v0.14.1`
- Updated the derived default S3DF Singularity image path to `/sdf/data/neutrino/images/spine_v0-14-1.sif`

### Fixed
- Restored compatibility with Python 3.6-era submit environments by replacing Python 3.9-only string prefix/suffix helpers in submit-time code.
- Normalized leading `v` container versions when deriving the default S3DF Singularity image path.

Full Changelog: [v0.6.1...v0.6.2](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.6.1...v0.6.2)

## [0.6.1] - 2026-06-26

### Added
- Added `submit.py --no-writer` to suppress automatic `io.writer.*` overrides for explicit `--source` and `--source-list` submissions.
- Added the June 26, 2026 DUNE10kt-1x2x6 full-chain inference configuration, using the May 2026 model weights with updated June post-processing.

### Changed
- Updated the default SPINE container release to `docker:ghcr.io/deeplearnphysics/spine:0.14.0`
- Updated the derived default S3DF Singularity image path to `/sdf/data/neutrino/images/spine_v0-14-0.sif`

### Fixed
- Fixed NERSC 80 GB GPU profiles to request the `gpu&hbm80g` constraint.
- Updated the DUNE10kt-1x2x6 post-processing configuration to remove the Michel maximum kinetic-energy cut and patch the calibration gain.

Full Changelog: [v0.6.0...v0.6.1](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.6.0...v0.6.1)

## [0.6.0] - 2026-06-02

### Added
- Added config-owned submissions: when no `--source` or `--source-list` is provided, jobs rely on the input configuration already present in the SPINE config and run as a single task.
- Added a centralized `DEFAULT_SPINE_VERSION` file used by both `configure.sh` and Python fallback container execution.

### Changed
- Updated the default SPINE container release to `docker:ghcr.io/deeplearnphysics/spine:0.13.2`
- Updated the derived default S3DF Singularity image path to `/sdf/data/neutrino/images/spine_v0-13-2.sif`
- Changed explicit file submissions to default to all files in one job instead of one task per file.
- Changed `--ntasks N` without `--files-per-task` to split explicit input files roughly evenly across `N` tasks.
- Stopped injecting `io.writer.*` overrides for config-owned submissions.
- Refreshed the README and quick reference to describe centralized container version handling.

### Fixed
- Rejected submit-time splitting and output overrides when no explicit file list is provided, avoiding ambiguous training/config-owned behavior.

Full Changelog: [v0.5.3...v0.6.0](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.5.3...v0.6.0)

## [0.5.3] - 2026-05-12

### Changed
- Updated the default SPINE container release to `docker:ghcr.io/deeplearnphysics/spine:0.12.2`
- Updated the derived default S3DF Singularity image path to `/sdf/data/neutrino/images/spine_v0-12-2.sif`
- Refreshed the README and quick reference to match the current default container release

### Fixed
- Pulls in SPINE `v0.12.2`, which fixes a critical truth position attribute unit issue in the upstream runtime container

Full Changelog: [v0.5.2...v0.5.3](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.5.2...v0.5.3)

## [0.5.2] - 2026-05-11

### Changed
- Updated the default SPINE container release to `docker:ghcr.io/deeplearnphysics/spine:0.12.1`
- Updated the derived default S3DF Singularity image path to `/sdf/data/neutrino/images/spine_v0-12-1.sif`
- Refreshed the README and quick reference to match the current default container release

### Fixed
- Preserved the standard S3DF `/sdf/` bind root when batch jobs add a custom `--spine-path` checkout to the container bind list

Full Changelog: [v0.5.1...v0.5.2](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.5.1...v0.5.2)

## [0.5.1] - 2026-05-11

### Added
- Added the `fsd` detector profile with `infer/fsd` as its configuration root and `s3df_ampere` as its default execution profile

### Changed
- Added `--output-suffix` and updated default output naming so SPINE writes HDF5 files under the job `output/` directory using input-derived names and the final config stem as the default suffix
- Updated batch job templates to pass writer output overrides through `--set io.writer.directory` and `--set io.writer.suffix`, while preserving explicit output file paths passed with `--output`

### Fixed
- Prevented nested generated composite configs from accumulating repeated `_composite` suffixes in their filenames

Full Changelog: [v0.5.0...v0.5.1](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.5.0...v0.5.1)

## [0.5.0] - 2026-05-11

### Breaking Changes
- SPINE is no longer packaged as a git submodule. Production jobs now rely on the tagged SPINE container image instead of a repository-local SPINE checkout.
- Batch and interactive execution now call the container-provided `spine` executable instead of `python3 $SPINE_BASEDIR/bin/run.py`.
- OpT0Finder setup is no longer sourced from `FMATCH_BASEDIR`; OpT0Finder is packaged in the SPINE container.

### Changed
- Documented the container-first runtime model in the README and quick reference
- Set the default SPINE container tag to `docker:ghcr.io/deeplearnphysics/spine:0.12.0`
- Derive the default S3DF Singularity image path from `SPINE_CONTAINER_VERSION` as `/sdf/data/neutrino/images/spine_v0-12-0.sif`
- Added `submit.py --set KEY=VALUE` support for SPINE runtime config overrides
- Added `--flashmatch-path /path/to/flashmatch` to source custom flash-matching setups for interactive or batch execution
- Added `--interactive-runtime` to let interactive mode use local `spine`, force container execution, or fall back to the container automatically
- Added `--spine-path /path/to/spine` support to point interactive and batch execution at a checkout without requiring a released `spine` executable on `PATH`, including container bind propagation where supported
- Renamed `--larcv` to `--larcv-path`
- Docker/Podman interactive fallback now requests `linux/amd64` by default
- Kept `--flashmatch` as a deprecated compatibility option; no external flash-matching setup is needed

### Removed
- Removed the `spine` submodule from version control
- Removed the SPINE submodule update helper script
- Removed `SPINE_BASEDIR` and `FMATCH_BASEDIR` defaults from `configure.sh`

Full Changelog: [v0.4.6...v0.5.0](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.4.6...v0.5.0)

## [0.4.6] - 2026-05-10

### Added
- New ICARUS v5 production configurations with corrected calibration handling
- New SBND Gen II production configuration with corrected weights
- New ND-LAr 260409 inference configuration
- Submit-time download preloading support for files referenced by `!download` tags
- PBS client support and ANL Polaris submission profile

### Changed
- Updated SPINE dependency to v0.10.10
- Updated GitHub Actions to current Node.js-supported action versions
- Made CVMFS binding opt-in and improved site submission templates
- Moved ICARUS calibration database lookup to `ICARUS_DATA_DIR`

### Fixed
- Corrected SBND Gen II model weights
- Corrected ICARUS trigger and post-processing configuration details
- Fixed CI preload import coverage and Codecov action parameters

Full Changelog: [v0.4.5...v0.4.6](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.4.5...v0.4.6)

## [0.4.5] - 2026-03-27

### Fixed
- Critical fix for litification script: now properly builds representation and preserves derived attributes

Full Changelog: [v0.4.4...v0.4.5](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.4.4...v0.4.5)

## [0.4.4] - 2026-03-25

### Added
- New ICARUS v4 inference configuration with GrapPA shower/track fix
- New SBND configuration with fully transfer trained weights and feature engineering fix
- ND-LAr v1 training weights (preliminary)

### Changed
- Reordered configuration descriptions
- Updated ICARUS and SBND READMEs to include newer configurations

Full Changelog: [v0.4.3...v0.4.4](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.4.3...v0.4.4)

## [0.4.3] - 2026-03-06

### Added
- New ICARUS MPV/MPR v5 trained configuration
- New SBND v2 configuration with fixed GrapPA feature engineering

Full Changelog: [v0.4.2...v0.4.3](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.4.2...v0.4.3)

## [0.4.2] - 2026-03-01

### Changed
- Updated SPINE dependency to v0.10.4

Full Changelog: [v0.4.1...v0.4.2](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.4.1...v0.4.2)

## [0.4.1] - 2026-03-01

### Added
- New SBND configuration without X-Arapuca parsing
- New ICARUS configuration with weights trained on v4, leveraging correct GrapPA feature extraction scheme
- ProtoDUNE-SP weights

### Changed
- Updated SPINE dependency to v0.10.3

### Fixed
- Interactive submitter functionality
- Configuration `!include` parsing issues (SPINE v0.10.2)

Full Changelog: [v0.4.0...v0.4.1](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.4.0...v0.4.1)

## [0.4.0] - 2026-02-10

**Post-reformatting baseline release** - Major architectural overhaul of the configuration system and codebase structure. This release marks a stable reference point before planned architectural evolution.

### ⚠️ Breaking Changes
- **Requires SPINE v0.9.0 or later**
- Configuration directory structure completely reorganized
- Include paths in existing configurations need updating

### What's Changed

- **Complete Configuration System Restructure:**
  - Introduced modular organization with separation of concerns
  - Mode-agnostic model configurations now in `config/common/model/`
  - Mode-specific configs (base/io/post) organized under `config/{infer,train}/{detector}/`
  - Implemented intelligent fallback resolution: detector-specific → common defaults
  - New standardized naming conventions with date-stamped configuration files

- **Submitter Codebase Modularization:**
  - Refactored monolithic submission script into maintainable modules
  - Separated concerns: configuration management, file handling, SLURM client
  - Improved code structure for extensibility and testing

- **Enhanced Job Submission Features:**
  - Local directory mode: jobs write to `--local-dir` by default (use `--central-dir` for production)
  - Better dependency handling for multi-array job requirements
  - Improved chunking logic with sequential execution guarantees
  - Smart job array generation (skips arrays for single-task jobs)

- **Extended Detector Support:**
  - Added DUNE 10kt-1x2x6 inference configurations
  - Improved NERSC cluster integration

### Migration Guide

For users with custom v0.3.x configurations, see detailed migration instructions in `config/common/README.md` and `config/infer/README.md`. Key steps:

1. Reorganize model configs → `config/common/model/`
2. Move detector-specific inference configs → `config/infer/{detector}/`
3. Update include paths in composite configurations
4. Update SPINE dependency to v0.9.0+

Full Changelog: [v0.3.1...v0.4.0](https://github.com/DeepLearnPhysics/spine-prod/compare/v0.3.1...v0.4.0)

## [0.3.1] - 2025-12-19
Fixed issue with `shower_start_merge` blocks in ICARUS configurations.

See [GitHub release](https://github.com/DeepLearnPhysics/spine-prod/releases/tag/v0.3.1) for details.

## [0.3.0] - 2025-12-17
Updated all configurations to reflect SPINE v0.8.0 geometry handling changes. Created new `geo` blocks, renamed `ndlar` to `nd-lar`, restructured detector naming conventions.

See [GitHub release](https://github.com/DeepLearnPhysics/spine-prod/releases/tag/v0.3.0) for details.

## [0.2.5] - 2025-12-12
Last release before geometry handling overhaul. Added lithification configurations, latest ICARUS production configs (250625), new ND-LAr configurations, and single 2x2 module support.

See [GitHub release](https://github.com/DeepLearnPhysics/spine-prod/releases/tag/v0.2.5) for details.

## [0.2.4] - 2025-09-04
Updated SBND configurations: interaction clustering anchored to tracks, backtracking fixes, containment module updates, MCS resolution modeling improvements.

See [GitHub release](https://github.com/DeepLearnPhysics/spine-prod/releases/tag/v0.2.4) for details.

## [0.2.3] - 2025-08-18
Updated SBND configurations: containment checks, cathode crosser handling, flash matching improvements.

See [GitHub release](https://github.com/DeepLearnPhysics/spine-prod/releases/tag/v0.2.3) for details.

## [0.2.2] - 2025-07-30
Bug fixes, shower quality attributes, SBND calibration updates, CRT info integration, cathode crosser post-processor.

See [GitHub release](https://github.com/DeepLearnPhysics/spine-prod/releases/tag/v0.2.2) for details.

## [0.2.1] - 2025-03-31
Additional configurations for ICARUS and SBND. Compatible with SPINE v0.2.0 through v0.3.X.

See [GitHub release](https://github.com/DeepLearnPhysics/spine-prod/releases/tag/v0.2.1) for details.

## [0.2.0] - 2025-01-17
All configurations work with SPINE v0.2.X. Flash matching configurations incompatible with SPINE v0.1.3 and below.

See [GitHub release](https://github.com/DeepLearnPhysics/spine-prod/releases/tag/v0.2.0) for details.

## [0.1.0] - 2024-11-25
First reference release. All configurations work with SPINE v0.1.X. Flash matching configurations incompatible with SPINE v0.2.0 and above.

See [GitHub release](https://github.com/DeepLearnPhysics/spine-prod/releases/tag/v0.1.0) for details.
