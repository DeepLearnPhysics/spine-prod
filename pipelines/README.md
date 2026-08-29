# Pipeline Examples

This directory contains example pipeline definitions for multi-stage SPINE processing.

## Pipeline Format

Pipelines are defined in YAML format with the following structure:

```yaml
defaults:
  profile: s3df_ampere
  time: "08:00:00"

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
    profile: s3df_hopper    # optional stage override of the default
    ntasks: 50              # optional, target number of tasks if files_per_task is omitted
    files_per_task: 5       # optional, overrides even splitting and uses ntasks as concurrency cap
    depends_on: []          # optional list of stage names
```

## Usage

```bash
./submit.py --pipeline pipelines/my_pipeline.yaml
```

Pipeline settings resolve in this order: profile defaults, pipeline `defaults`,
stage fields, then explicit CLI overrides. For example, this runs every stage
with the same checkout and scheduler account while overriding any profiles in
the YAML:

```bash
./submit.py --pipeline pipelines/my_pipeline.yaml \
  --spine-path /path/to/spine --profile s3df_hopper --account my_account
```

Pipeline-wide CLI overrides are supported for software paths, profiles,
scheduler resources, and first-class SPINE runtime options. Data sources,
outputs, dependencies, run lifecycle settings, and model weights must remain on
their individual stages. Unknown fields and unsupported pipeline CLI options
are rejected rather than ignored.

See `icarus_production_example.yaml` for a complete example.

## Generic staged-training prototype

`generic/uresnet_ppn_to_graph_spice_240805.yaml` defines the first cached model
transition:

1. Train standalone UResNet-PPN and select `snapshot-best.ckpt`.
2. Materialize its canonical `seg_pred` and adapted `clust_label_adapt`
   products into separate training and validation staged caches.
3. Train standalone Graph-SPICE from raw LArCV truth plus the aligned cache.

Replace the `/path/to/workflow` destination before submission. The prototype
uses SPINE's target-qualified source overrides for the mixed Graph-SPICE
dataset and `--module-weight` for the cached segmentation jobs; it requires no
generic `--set` overrides. SPINE validates stored source provenance and fails
rather than silently pairing the wrong events.

To run against an unreleased checkout, pass `--spine-path /path/to/spine` when
submitting the pipeline. A stage-level `spine_path` remains available when only
one stage needs a different checkout.
