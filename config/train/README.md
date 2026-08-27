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
- **`generic/`**: YAML benchmark recipes for the generic dataset
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

SPINE v1.0.0 uses a top-level `train` block and an optional sibling
`validation` block. Integrated validation at checkpoint boundaries is
recommended for new training productions because it records the validation
metrics with the training process and supports early stopping and stable best
checkpoints.

## Versioning and Reproducibility

Training configurations preserve the names used for each training campaign. Where a dated version is present, it uses the same YYMMDD convention as inference configurations. Each recipe should document the setup used to produce its corresponding weights and link those weights to the appropriate configuration under `config/infer/`.

## Usage

Training configurations are submitted as persistent named runs:

```bash
# Basic training
./submit.py -c train/generic/uresnet.yaml \
  --stage train --run-dir /path/to/experiments/uresnet/default \
  --source-list /path/to/train_file_list.txt \
  --val-source-list /path/to/validation_file_list.txt

# --gpus controls both scheduler allocation and SPINE world size. An explicit
# --world-size is accepted only when it matches the allocated GPU count.

# Multi-GPU training
./submit.py -c train/generic/uresnet.yaml \
  --stage train --run-dir /path/to/experiments/uresnet/default \
  --gpus 4 \
  --source-list /path/to/train_file_list.txt \
  --val-source-list /path/to/validation_file_list.txt

# Strictly resume complete training state from the latest checkpoint
./submit.py -c train/generic/uresnet.yaml \
  --stage train --run-dir /path/to/experiments/uresnet/default --resume
```

spine-prod selects the latest numeric checkpoint, verifies its SPINE
checksum sidecar when present, and forwards SPINE's explicit `--resume` flag.
Legacy checkpoints without checksum sidecars remain supported. Use
`--resume-from RUN_DIR/weights/snapshot-N.ckpt` for an intentional rollback.

For an alternate dataset or a legacy training without integrated validation,
run standalone incremental validation against the same run directory:

```bash
./submit.py -c /path/to/validation.yaml \
  --stage validation --run-dir /path/to/experiments/deghost/default
```

## Model Weights

Trained model weights are stored separately and referenced in the corresponding `infer/<detector>/model/` configurations. Training configurations document:
- Training dataset versions (e.g., MPV/MPR v02, v03, v04)
- Training infrastructure (e.g., Polaris, SDF)
- Training duration and convergence metrics
- Links to weight storage locations

Refer to the checked-in configuration files for detector-specific training details and dataset information.
