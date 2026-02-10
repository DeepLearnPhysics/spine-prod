# Changelog

All notable changes to the SPINE Production System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

---

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
