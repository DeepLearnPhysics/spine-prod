# SPINE Production System

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![CI](https://github.com/DeepLearnPhysics/spine-prod/actions/workflows/ci.yml/badge.svg)](https://github.com/DeepLearnPhysics/spine-prod/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DeepLearnPhysics/spine-prod/branch/main/graph/badge.svg)](https://codecov.io/gh/DeepLearnPhysics/spine-prod)

A production system for running [SPINE](https://github.com/DeepLearnPhysics/SPINE) (Scalable Particle Imaging with Neural Embeddings) reconstruction on SLURM-based HPC clusters.

## Overview

SPINE is a deep learning-based reconstruction framework for liquid argon time projection chamber (LArTPC) detectors. This production system provides tools for running SPINE at scale on large datasets using SLURM job arrays.

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/DeepLearnPhysics/spine-prod.git
cd spine-prod

# Configure environment to point to your SPINE installation
# Edit configure.sh to set SPINE_BASEDIR to your SPINE install location
source configure.sh
```

**Note:** This repository contains SPINE configuration files for production runs. You need an existing SPINE installation - either a system-wide installation or a local build. The `configure.sh` script should be edited to point `SPINE_BASEDIR` to your SPINE installation directory.

### 2. Basic Job Submission

```bash
# Run on a single file
./submit.py --config infer/icarus/latest.cfg --source /path/to/data.root

# Run on multiple files (glob)
./submit.py --config infer/icarus/latest_data.cfg --source /path/to/data/*.root

# Run from a file list (recommended)
./submit.py --config infer/2x2/latest.cfg --source-list file_list.txt
```

### 3. Advanced Usage

```bash
# Use a specific resource profile
./submit.py --config infer/icarus/latest.cfg --source data/*.root --profile s3df_turing

# Process multiple files per job
./submit.py --config infer/icarus/latest.cfg --source data/*.root --files-per-task 5

# Limit parallel tasks
./submit.py --config infer/icarus/latest.cfg --source data/*.root --ntasks 50

# Run a multi-stage pipeline
./submit.py --pipeline pipelines/icarus_production_example.yaml

# Dry run (see what would be submitted)
./submit.py --config infer/icarus/latest.cfg --source test.root --dry-run
```

## Directory Structure

```
spine-prod/
├── configure.sh              # Environment setup script
├── submit.py                 # Main submission orchestrator (NEW!)
├── run.sh                    # Legacy submission script
├── README.md                 # This file
│
├── infer/                   # SPINE configurations
│   ├── icarus/              # ICARUS detector configs
│   ├── sbnd/                # SBND detector configs
│   ├── 2x2/                 # 2x2 detector configs
│   ├── nd-lar/              # ND-LAr detector configs
│   └── generic/             # Generic configs
│
├── templates/               # Job templates
│   ├── profiles.yaml       # Resource profiles
│   └── job_template.sbatch # SLURM job template
│
├── pipelines/              # Multi-stage pipeline definitions
│   └── icarus_production_example.yaml
│
├── scripts/                # Utility scripts
├── slurm/                  # Legacy SLURM scripts
├── tests/                  # Test suite
└── jobs/                   # Job artifacts (auto-created)
```

## Configuration System

SPINE uses a hierarchical configuration system where configs can include and override base configurations.

### Config Organization

```
infer/<detector>/
├── <detector>_base.cfg           # Base MC configuration
├── <detector>_base_data.cfg      # Base data-only configuration
└── <detector>_full_chain_*.cfg   # Version-specific configs
```

### Example: ICARUS Configurations

```bash
# Latest MC configuration
infer/icarus/latest.cfg

# Latest data-only configuration
infer/icarus/latest_data.cfg

# Specific version with cosmic overlay
infer/icarus/icarus_full_chain_co_250625.cfg

# Data with lite outputs
infer/icarus/icarus_full_chain_data_co_lite_250625.cfg
```

See individual config directories for detector-specific documentation.

## Resource Profiles

Resource profiles define SLURM resource requirements for different use cases. Profiles are defined in `templates/profiles.yaml`.

### S3DF Node Resources

Understanding the available resources on each partition helps justify the profile configurations:

| Partition | GPUs/Node | GPU Type | CPUs/Node | RAM/Node | Resources per GPU |
|-----------|-----------|----------|-----------|----------|------------------|
| `ampere` | 4 | A100 (40GB) | 112 | 952 GB | 28 CPUs, 238 GB |
| `turing` | 10 | RTX 2080 Ti (11GB) | 40 | 160 GB | 4 CPUs, 16 GB |
| `milano` | 0 | - | 120 | 480 GB | - |
| `roma` | 0 | - | 120 | 480 GB | - |

Profile allocations are designed to:
- **Ampere**: Request full resources per GPU (28 CPUs × 8 GB/CPU = 224 GB per GPU)
- **Turing**: Request full resources per GPU (4 CPUs × 4 GB/CPU = 16 GB per GPU)
- **CPU nodes**: Request minimal resources (1 CPU × 4 GB = 4 GB) for flexible scheduling

### Available Profiles

| Profile | Partition | GPU Type | GPU Memory | GPUs | CPUs | Memory | Time | Use Case |
|---------|-----------|----------|------------|------|------|--------|------|----------|
| `s3df_ampere` | ampere | A100 | 40GB | 1 | 28 | 8GB/CPU | 2h | High-performance GPU processing (default) |
| `s3df_turing` | turing | RTX 2080 Ti | 11GB | 1 | 4 | 4GB/CPU | 2h | Cheaper GPU inference |
| `s3df_milano` | milano | - | - | 0 | 1 | 4GB/CPU | 2h | CPU-only analysis |
| `s3df_roma` | roma | - | - | 0 | 1 | 4GB/CPU | 2h | CPU-only analysis |

### Profile Selection

Profiles are auto-detected based on detector and config, or can be specified explicitly:

```bash
# Auto-detect (default)
./submit.py --config infer/icarus/latest.cfg --source data.root

# Explicit profile
./submit.py --config infer/icarus/latest.cfg --source data.root --profile s3df_turing

# Override specific resources
./submit.py --config infer/icarus/latest.cfg --source data.root --time 2:00:00 --cpus-per-task 8
```

## Pipeline Mode

Pipelines allow you to chain multiple processing stages with automatic dependency management.

### Pipeline Definition

Create a YAML file in `pipelines/`:

```yaml
stages:
  - name: reconstruction
    config: infer/icarus/latest_data.cfg
    files: /path/to/raw/*.root
    profile: s3df_ampere
    ntasks: 100
  
  - name: analysis
    depends_on: [reconstruction]  # Wait for reconstruction to complete
    config: infer/icarus/analysis.cfg
    files: output_reco/*.h5
    profile: s3df_milano
    ntasks: 20
```

### Submit Pipeline

```bash
./submit.py --pipeline pipelines/my_pipeline.yaml
```

## Job Management

### Job Artifacts

Each submission creates a timestamped directory in `jobs/`:

```
jobs/20260101_143022_spine_icarus_latest/
├── job_metadata.json           # Complete job metadata
├── files_chunk_0.txt          # Input file lists
├── submit_chunk_0.sbatch      # Generated submission scripts
├── logs/                      # SLURM stdout/stderr
│   ├── spine_icarus_latest_12345_1.out
│   └── spine_icarus_latest_12345_1.err
└── output/                    # Output files
    └── spine_icarus_latest.h5
```

### Monitoring Jobs

```bash
# View job status
squeue -u $USER

# View job details
scontrol show job <job_id>

# View logs
tail -f jobs/<job_dir>/logs/spine_*.out

# Cancel job
scancel <job_id>
```

### Job Metadata

Each job saves complete metadata for reproducibility:

```json
{
  "job_name": "spine_icarus_latest",
  "detector": "icarus",
  "config": "infer/icarus/latest.cfg",
  "profile": "s3df_ampere",
  "num_files": 100,
  "job_ids": ["12345", "12346"],
  "submitted": "2026-01-01T14:30:22",
  "command": "./submit.py --config ..."
}
```

### Automatic Cleanup of Intermediate Files

Pipelines can automatically clean up intermediate outputs once downstream stages complete. Add a `cleanup` field to any stage that produces temporary files:

```yaml
stages:
  - name: reconstruction
    config: infer/icarus/latest.cfg
    files: /path/to/input/*.root
    output: output_reco
    # Clean up output_reco/ after all dependent stages finish
    cleanup:
      - output_reco
      - temp_files
  
  - name: analysis
    depends_on: [reconstruction]
    config: infer/icarus/analysis.cfg
    files: output_reco/*.h5
    output: output_analysis
```

The cleanup job:
- Only runs if downstream stages complete successfully (`afterok` dependency)
- Runs as a minimal resource job (1 CPU, 1GB RAM, 10min timeout)
- Safely checks for path existence before removal
- Logs all cleanup actions for auditing

This is especially useful for large-scale production to save disk space by removing intermediate reconstruction outputs after final analysis completes.

## Advanced Features

### Custom Software Paths

```bash
# Use custom LArCV installation
./submit.py --config infer/icarus/latest.cfg --source data.root --larcv /path/to/larcv

# Enable flash matching
./submit.py --config infer/icarus/latest.cfg --source data.root --flashmatch
```

### Job Dependencies

```bash
# Submit with dependency on another job
./submit.py --config infer/icarus/stage2.cfg --source output/*.h5 --dependency afterok:12345
```

### Array Job Optimization

```bash
# Process 5 files per job (reduces overhead)
./submit.py --config infer/icarus/latest.cfg --source data/*.root --files-per-task 5

# Limit concurrent tasks to 50
./submit.py --config infer/icarus/latest.cfg --source data/*.root --ntasks 50
```

## Detector-Specific Guides

### ICARUS

ICARUS uses split cryostat processing with cosmic overlay:

```bash
# Standard cosmic overlay processing
./submit.py --config infer/icarus/latest.cfg --source data.root

# Data-only mode (no truth labels)
./submit.py --config infer/icarus/latest_data.cfg --source data.root

# NuMI beam configuration
./submit.py --config infer/icarus/latest_numi.cfg --source data.root

# Lite output (reduced file size)
./submit.py --config infer/icarus/icarus_full_chain_data_co_lite_250625.cfg --source data.root
```

### SBND

```bash
./submit.py --config infer/sbnd/latest.cfg --source data.root
```

### 2x2

2x2 uses higher resource requirements:

```bash
./submit.py --config infer/2x2/latest.cfg --source data.root --profile s3df_ampere
```

### ND-LAr

```bash
./submit.py --config infer/nd-lar/nd-lar_base.cfg --source data.root
```

## Troubleshooting

### Environment Not Set

```
WARNING: MLPROD_BASEDIR not set. Did you source configure.sh?
```

**Solution:** Source the environment:
```bash
source configure.sh
```

### Missing Dependencies

```
ERROR: jinja2 is required. Install with: pip install jinja2
```

**Solution:** Install Python dependencies:
```bash
pip install jinja2 pyyaml
```

### Job Failures

1. Check SLURM logs in `jobs/<job_dir>/logs/`
2. Review job metadata in `jobs/<job_dir>/job_metadata.json`
3. Test configuration on a single file with `--dry-run`
4. Verify input files exist and are accessible

### Out of Memory

**Solution:** Use a profile with more memory or override memory:
```bash
./submit.py --config infer/icarus/latest.cfg --source data.root --profile s3df_ampere
```

Or override memory:
```bash
./submit.py --config infer/icarus/latest.cfg --source data.root --mem-per-cpu 16g
```

### Job Time Limit

**Solution:** Request more time:
```bash
./submit.py --config infer/icarus/latest.cfg --source data.root --time 4:00:00
```

## Legacy Scripts

The original submission system (`run.sh`, `slurm/run_slurm.sh`) is still available for backwards compatibility but is deprecated. New users should use `submit.py`.

### Migrating from Legacy Scripts

Old:
```bash
./run.sh -c infer/icarus/latest.cfg -n 50 -t 2:00:00 file_list.txt
```

New:
```bash
./submit.py --config infer/icarus/latest.cfg --source-list file_list.txt --ntasks 50 --time 2:00:00
```

## Best Practices

### 1. Test Before Production

Always test configurations on a small sample:

```bash
# Test with dry run
./submit.py --config infer/icarus/latest.cfg --source test.root --dry-run

# Test with single file
./submit.py --config infer/icarus/latest.cfg --source test.root
```

### 2. Use Appropriate Profiles

- Use `s3df_ampere` for high-performance GPU processing (default)
- Use `s3df_turing` for cheaper GPU inference
- Use `s3df_milano` or `s3df_roma` for CPU-only analysis

### 3. Optimize File Batching

```bash
# For many small files, batch them
./submit.py --config infer/icarus/latest.cfg --source small_files/*.root --files-per-task 10

# For large files, process individually
./submit.py --config infer/icarus/latest.cfg --source large_files/*.root --files-per-task 1
```

### 4. Monitor Resource Usage

Check actual resource usage to optimize future jobs:

```bash
seff <job_id>
```

### 5. Track Your Work

Job metadata is automatically saved. Keep important job directories:

```bash
# Jobs are in timestamped directories
ls -lt jobs/
```

## Environment Variables

Set by `configure.sh`:

- `MLPROD_BASEDIR` - Base directory of this repository
- `MLPROD_CFGDIR` - Configuration directory
- `SPINE_BASEDIR` - SPINE installation path
- `FMATCH_BASEDIR` - OpT0Finder installation path
- `SINGULARITY_PATH` - Container image path

## Contributing

### Adding New Profiles

Edit `templates/profiles.yaml`:

```yaml
profiles:
  my_custom_profile:
    partition: my_partition
    gpus: 2
    cpus_per_task: 16
    mem_per_cpu: 8g
    time: "6:00:00"
    description: "My custom profile description"
```

### Adding Detector Defaults

Edit `templates/profiles.yaml`:

```yaml
detectors:
  my_detector:
    default_profile: s3df_ampere
    configs_dir: infer/my_detector
    account: "my:account"
```

## Development

### Installation for Development

For contributors who need to run tests and development tools:

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac

# Install SPINE (required for running parsing tests)
pip install "spine-ml @ git+https://github.com/DeepLearnPhysics/SPINE.git@main"

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

**Note:** SPINE installation is only required for development if you want to run the configuration parsing tests. Production users only need to point to an existing SPINE installation via `configure.sh`.

### Running Tests

```bash
# Run all tests
pytest

# Run config validation tests only
pytest tests/test_config_validation.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Pre-commit Hooks

This repository uses pre-commit hooks for code quality:
- **check-yaml**: Validates YAML syntax
- **yamllint**: Lints YAML files for style
- **prettier**: Formats YAML files
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with newline

```bash
# Run hooks manually
pre-commit run --all-files
```

### Config Validation

All configuration files are automatically validated in CI to ensure they parse correctly:

```python
from config import load_config
config = load_config('infer/icarus/latest.yaml')
```

## Related Tools

### Production Database (spine-db)

A companion tool for indexing and browsing SPINE production metadata. See [spine-db](https://github.com/DeepLearnPhysics/spine-db) for documentation.

## Support

- **Issues:** https://github.com/DeepLearnPhysics/spine-prod/issues
- **SPINE Documentation:** https://github.com/DeepLearnPhysics/spine
- **Contact:** SPINE development team

## License

This software is provided under the same license as SPINE.

## Citation

If you use SPINE in your research, please cite the relevant SPINE publications.

## Acknowledgments

Development supported by the DOE Office of High Energy Physics and the National Science Foundation.
