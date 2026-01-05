# SPINE Production Quick Reference

## Common Commands

### Basic Submission
```bash
# Single file
./submit.py --config infer/icarus/latest --source data.root

# Multiple files (glob)
./submit.py --config infer/icarus/latest --source data/*.root

# From file list (recommended)
./submit.py --config infer/icarus/latest --source-list files.txt
```

### Modifiers
```bash
# Apply data modifier
./submit.py --config infer/icarus/latest --source data.root --apply-mods data

# Multiple modifiers
./submit.py --config infer/icarus/latest --source data.root --apply-mods data lite

# Pin specific modifier version
./submit.py --config infer/icarus/latest --source data.root --apply-mods data:240719 lite

# List available modifiers for a config
./submit.py --list-mods infer/icarus/icarus_full_chain_co_250625.yaml
```

### Profiles
```bash
# Auto-detect (default)
./submit.py --config infer/icarus/latest --source data.root

# High-performance A100
./submit.py --config infer/icarus/latest --source data.root --profile s3df_ampere

# Cheaper RTX 2080 Ti
./submit.py --config infer/icarus/latest --source data.root --profile s3df_turing

# CPU only
./submit.py --config infer/icarus/latest --source data.root --profile s3df_milano
```

### Job Control
```bash
# Limit parallel tasks
./submit.py --config infer/icarus/latest --source data/*.root --ntasks 50

# Multiple files per task
./submit.py --config infer/icarus/latest --source data/*.root --files-per-task 5

# Custom time limit
./submit.py --config infer/icarus/latest --source data.root --time 2:00:00

# Dry run (test without submitting)
./submit.py --config infer/icarus/latest --source data.root --dry-run
```

### Pipeline Mode
```bash
./submit.py --pipeline pipelines/my_pipeline.yaml
```

## Monitoring

```bash
# Job status
squeue -u $USER

# Job details
scontrol show job JOB_ID

# Cancel job
scancel JOB_ID

# View logs
tail -f jobs/*/logs/*.out

# Resource usage
seff JOB_ID
```

## Profiles Quick Reference

| Profile | GPU Type | Use When |
|---------|----------|----------|
| `s3df_ampere` | A100 (40GB) | High-performance GPU processing (default) |
| `s3df_turing` | RTX 2080 Ti (11GB) | Cheaper GPU inference |
| `s3df_milano` | None | CPU-only analysis |
| `s3df_roma` | None | CPU-only analysis |

## Common Configs

### ICARUS
```bash
# Latest MC
infer/icarus/latest.cfg

# Latest composite (dynamic)
./submit.py --config infer/icarus/latest --source data.root

# Specific version
./submit.py --config infer/icarus/icarus_full_chain_co_250625.yaml --source data.root

# With modifiers
./submit.py --config infer/icarus/latest --source data.root --apply-mods data lite
```

### Other Detectors
```bash
# SBND
./submit.py --config infer/sbnd/sbnd_full_chain_240918.cfg --source data.root

# 2x2
./submit.py --config infer/2x2/2x2_full_chain_240819.cfg --source data.root

# ND-LAR
./submit.py --config infer/nd-lar/nd-lar_full_chain_ovl_250806.cfg --source data.root
```

| Problem | Solution |
|---------dependencies | `pip install -r requirements.txt` |
| Out of memory | Use `--profile s3df_ampere` or `--mem-per-cpu 16g` |
| Job timeout | Use `--time 4:00:00` |
| Unknown modifier | Use `--list-mods CONFIG` to see available modifier |
| Out of memory | Use `--profile s3df_ampere` or `--mem-per-cpu 16g` |
| Job timeout | Use `--time 4:00:00` |
| Need more info | Check `jobs/*/job_metadata.json` |

## Job Directory Structure

```
jobs/TIMESTAMP_JOBNAME/
├── job_metadata.json      # All job info
├── files_chunk_*.txt      # Input files
├── submit_chunk_*.sbatch  # Submission scripts
├── logs/                  # SLURM logs
└── output/                # Results
```

## Full Help

```bash
./submit.py --help
```
