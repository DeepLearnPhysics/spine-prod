# SPINE Production Inference Configurations

This directory contains production inference configurations for processing real data and simulation from various LArTPC (Liquid Argon Time Projection Chamber) detectors using SPINE.

## Purpose

The configurations in this directory are used to:
- Process detector data with trained SPINE models for physics analysis
- Apply full reconstruction chains (UResNet, PPN, gSPICE, GrapPAs) to LArTPC events
- Generate standardized outputs for physics analyses across multiple experiments
- Provide reproducible, version-controlled production settings

## Supported Detectors

Each subdirectory contains detector-specific configurations:

- **`2x2/`**: 2x2 demonstrator detector configurations
- **`2x2-single/`**: Single-module 2x2 configurations
- **`fsd/`**: Full Scale Demonstrator (FSD) configurations
- **`generic/`**: Generic detector configurations for testing and development
- **`icarus/`**: ICARUS detector configurations
- **`nd-lar/`**: DUNE Near Detector Liquid Argon (ND-LAr) configurations
- **`sbnd/`**: Short-Baseline Near Detector (SBND) configurations

## Configuration Structure

All detector directories follow a modular structure:

- **`base/`**: Detector geometry and base builder configurations
- **`io/`**: Input/output schemas and dataset definitions
- **`model/`**: Model architectures and trained weights paths
- **`modifier/`**: Configuration modifiers (e.g., data-only mode)
- **`post/`**: Post-processing including flash matching and analysis tools
- **`legacy/`**: Archived configurations for backward compatibility

Top-level configuration files (e.g., `detector_full_chain_YYMMDD.yaml`) compose these modules into complete reconstruction chains.

## Versioning and Reproducibility

Configurations are versioned by date (YYMMDD format) to ensure reproducibility:
- Each version corresponds to specific model weights and processing parameters
- Legacy configurations are preserved in `legacy/` directories
- Version history and changes are documented in detector-specific README files

## Usage

Configurations are used with the SPINE inference framework:

```bash
# Basic usage
./submit.py -c <detector>/<config_file>.yaml -s <input_file>

# With modifiers (data-only mode, lite output, etc.)
./submit.py -c <detector>/<config_file>.yaml -s <input_file> --apply_mods data,lite
```

Refer to individual detector README files for detector-specific details, training datasets, and version-specific changes.
