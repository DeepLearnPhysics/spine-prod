# SPINE Training Configurations

This directory contains training configurations for SPINE models used across various LArTPC (Liquid Argon Time Projection Chamber) detectors.

## Purpose

The configurations in this directory are used to:
- Train SPINE models (UResNet, PPN, gSPICE, GrapPAs) on detector-specific datasets
- Document training hyperparameters and model architectures
- Maintain reproducible training recipes for each production model
- Track model evolution and improvements across training iterations

## Available Configurations

The currently populated configuration trees are:

- **`config/`**: Legacy component and full-chain `.cfg` training recipes
- **`icarus/`**: YAML training configurations for ICARUS

The `dune10kt-1x2x6/` directory is reserved for future checked-in DUNE 10 kt training configurations and is not currently populated.

## Configuration Structure

Training configurations typically include:

- **Training datasets**: File lists and data locations
- **Model architecture**: Network structure and parameters
- **Loss functions**: Class weights and loss configurations
- **Training parameters**: Learning rates, batch sizes, epochs
- **Augmentation**: Data augmentation strategies
- **Validation**: Validation datasets and metrics

## Versioning and Reproducibility

Training configurations preserve the names used for each training campaign. Where a dated version is present, it uses the same YYMMDD convention as inference configurations. Each recipe should document the setup used to produce its corresponding weights and link those weights to the appropriate configuration under `config/infer/`.

## Usage

Training configurations are used with the SPINE training framework:

```bash
# Basic training
./submit.py -c config/train/icarus/deghost/deghost.yaml

# Multi-GPU training
./submit.py -c config/train/icarus/deghost/deghost.yaml --gpus 4

# Resume training
./submit.py -c config/train/icarus/deghost/deghost.yaml --set model.weight_path=/path/to/checkpoint.ckpt
```

## Model Weights

Trained model weights are stored separately and referenced in the corresponding `infer/<detector>/model/` configurations. Training configurations document:
- Training dataset versions (e.g., MPV/MPR v02, v03, v04)
- Training infrastructure (e.g., Polaris, SDF)
- Training duration and convergence metrics
- Links to weight storage locations

Refer to the checked-in configuration files for detector-specific training details and dataset information.
