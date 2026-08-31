# Pipeline Examples

This directory contains example pipeline definitions for multi-stage SPINE processing.

## Pipeline Format

Pipelines are defined in YAML format with the following structure:

```yaml
workspace: null

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
    # export_weights: /path/to/composed.ckpt  # terminal model-only stage
    # set: [nested.config.key=value]
    profile: s3df_hopper    # optional stage override of the default
    ntasks: 50              # optional, target number of tasks if files_per_task is omitted
    files_per_task: 5       # optional, overrides even splitting and uses ntasks as concurrency cap
    depends_on: []          # optional list of stage names
```

## Usage

```bash
./submit.py \
  --pipeline pipelines/my_pipeline.yaml \
  --workspace /path/to/workflow
```

Pipeline settings resolve in this order: profile defaults, pipeline `defaults`,
stage fields, then explicit CLI overrides. For example, this runs every stage
with the same checkout and scheduler account while overriding any profiles in
the YAML:

```bash
./submit.py --pipeline pipelines/my_pipeline.yaml \
  --workspace /path/to/workflow \
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

Each original source file has one staged cache. Later materialization jobs in
the full-chain prototype append named groups to that same HDF5 file rather
than producing a new physical file for every transition.

The generic prototypes define their train and validation inputs once under
`collections.splits`. A stage-level `for_each` expands cache templates into
independent, concretely named jobs before dependency validation and submission.
The full-chain workflow finishes with a CPU-only `export_weights` stage that
loads every standalone `snapshot-best.ckpt` and writes one directly runnable
full-chain checkpoint plus its SHA-256 sidecar.

The generic prototypes declare `workspace: null`; choose the shared output root
at launch with `--workspace /path/to/workflow`. The prototype uses SPINE's
target-qualified source overrides for the mixed Graph-SPICE dataset and
`--module-weight` for the cached segmentation jobs; it requires no generic
`--set` overrides. SPINE validates stored source provenance and fails rather
than silently pairing the wrong events.

To run against an unreleased checkout, pass `--spine-path /path/to/spine` when
submitting the pipeline. A stage-level `spine_path` remains available when only
one stage needs a different checkout.
