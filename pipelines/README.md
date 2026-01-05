# Pipeline Examples

This directory contains example pipeline definitions for multi-stage SPINE processing.

## Pipeline Format

Pipelines are defined in YAML format with the following structure:

```yaml
stages:
  - name: stage_name
    config: path/to/config.cfg
    files: [input files or pattern]
    profile: s3df_ampere    # optional (default: auto-detect)
    ntasks: 50              # optional
    files_per_task: 1       # optional
    depends_on: []          # optional list of stage names
```

## Usage

```bash
./submit.py --pipeline pipelines/my_pipeline.yaml
```

See `icarus_production_example.yaml` for a complete example.
