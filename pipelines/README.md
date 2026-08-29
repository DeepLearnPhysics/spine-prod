# Pipeline Examples

This directory contains example pipeline definitions for multi-stage SPINE processing.

## Pipeline Format

Pipelines are defined in YAML format with the following structure:

```yaml
stages:
  - name: stage_name
    config: path/to/config.yaml
    source: [input files or pattern]
    # source_list: /path/to/files.txt  # mutually exclusive with source
    # val_source / val_source_list are available for training stages
    # Composite datasets use named sources:
    # sources:
    #   larcv: {source: /path/to/raw.root}
    #   hdf5: {source: /path/to/cache.h5}
    # validation_sources: ...         # same shape for validation
    # module_weight: {module: /path/to/checkpoint.ckpt}
    # set: [nested.config.key=value]
    profile: s3df_ampere    # optional (default: auto-detect)
    ntasks: 50              # optional, target number of tasks if files_per_task is omitted
    files_per_task: 5       # optional, overrides even splitting and uses ntasks as concurrency cap
    depends_on: []          # optional list of stage names
```

## Usage

```bash
./submit.py --pipeline pipelines/my_pipeline.yaml
```

See `icarus_production_example.yaml` for a complete example.

## Generic staged-training prototype

`generic/uresnet_ppn_to_graph_spice_240805.yaml` defines the first cached model
transition:

1. Train standalone UResNet-PPN and select `snapshot-best.ckpt`.
2. Materialize its canonical `seg_pred` product into separate training and
   validation staged caches.
3. Train standalone Graph-SPICE from raw LArCV truth plus the aligned cache.

Replace the `/path/to/workflow` destination before submission. The prototype
uses SPINE's target-qualified source overrides for the mixed Graph-SPICE
dataset and `--module-weight` for the cached segmentation jobs; it requires no
generic `--set` overrides. SPINE validates stored source provenance and fails
rather than silently pairing the wrong events.

Until these interfaces appear in a tagged SPINE release, run the prototype
against a checkout containing commits `bc7c76fb`, `8e5e918b`, and `b432ae80`
by setting `spine_path` on its stages.
