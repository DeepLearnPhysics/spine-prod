# SPINE Training Configurations

This directory contains training configurations for SPINE models used across various LArTPC (Liquid Argon Time Projection Chamber) detectors.

## Purpose

The configurations in this directory are used to:
- Train SPINE models (UResNet, PPN, gSPICE, GrapPAs) on detector-specific datasets
- Document training hyperparameters and model architectures
- Maintain reproducible training recipes for each production model
- Track model evolution and improvements across training iterations

## Supported Detectors

Each subdirectory will contain detector-specific training configurations:

- **`2x2/`**: 2x2 demonstrator detector training configs
- **`2x2-single/`**: Single-module 2x2 training configs
- **`fsd/`**: Far Site Detector (FSD) training configs
- **`generic/`**: Generic detector configs for development
- **`icarus/`**: ICARUS detector training configs
- **`nd-lar/`**: Near Detector Liquid Argon (ND-LAr) training configs
- **`sbnd/`**: Short-Baseline Near Detector (SBND) training configs

## Configuration Structure

Training configurations typically include:

- **Training datasets**: File lists and data locations
- **Model architecture**: Network structure and parameters
- **Loss functions**: Class weights and loss configurations
- **Training parameters**: Learning rates, batch sizes, epochs
- **Augmentation**: Data augmentation strategies
- **Validation**: Validation datasets and metrics

## Versioning and Reproducibility

Training configurations are versioned to match their corresponding inference configurations:
- Each training version (YYMMDD format) produces model weights used in production
- Training configs document the exact setup used to produce each set of weights
- Links training parameters to specific model versions in `infer/` directory

## Usage

Training configurations are used with the SPINE training framework:

```bash
# Basic training
./submit.py -c <detector>/<config_file>.yaml

# Multi-GPU training
./submit.py -c <detector>/<config_file>.yaml --gpus 4

# Resume training
./submit.py -c <detector>/<config_file>.yaml --weight_path <checkpoint>
```

## Model Weights

Trained model weights are stored separately and referenced in the corresponding `infer/<detector>/model/` configurations. Training configurations document:
- Training dataset versions (e.g., MPV/MPR v02, v03, v04)
- Training infrastructure (e.g., Polaris, SDF)
- Training duration and convergence metrics
- Links to weight storage locations

Refer to individual detector directories (when populated) for detector-specific training details and dataset information.
