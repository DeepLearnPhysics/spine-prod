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
    #   primary: {source: /path/to/raw.root}
    #   cache: {source: /path/to/train/cache.spine-cache}
    # validation_sources: ...         # same shape for validation
    # val_entry_fraction_range: [0.0, 0.5]  # validation-only partition
    # entry_filter: /path/to/train-filter.yaml
    # val_entry_filter: /path/to/validation-filter.yaml
    # module_weight: {module: /path/to/checkpoint.ckpt}
    # weight_path: /path/to/composed.ckpt  # complete-model checkpoint
    # export_weights: /path/to/composed.ckpt  # terminal model-only stage
    # cache_repository: /path/to/train/cache.spine-cache
    # cache_stage: segmentation
    # set: [nested.config.key=value]
    profile: s3df_hopper    # optional stage override of the default
    ntasks: 4               # target tasks, or array concurrency with files_per_task
    files_per_task: 5       # optional, overrides even splitting and uses ntasks as concurrency cap
    depends_on: []          # optional list of stage names
```

`entry_fraction_range` and `val_entry_fraction_range` forward SPINE's
half-open fractional entry selectors. The generic full-chain pipelines retain
the entire validation source in their caches, validate training against
`[0.0, 0.5)`, and reserve `[0.5, 1.0)` for the final evaluation. Keeping the
derived caches complete preserves positional alignment for mixed datasets.

Standalone `kind: filter` stages run the reusable `spine-filter` workflow
before raw data reaches the parser. A scan records per-source measurements;
its dependent build publishes a file-aware manifest and accepted-source list:

```yaml
- name: scan_train_filter
  kind: filter
  operation: scan
  config: filter/protodune-sp/space_points_260210.yaml
  source_list: /path/to/train.txt
  cache_dir: ${workspace}/filter/counts
  run_dir: ${workspace}/filter/train/scan
  workers: 16

- name: build_train_filter
  kind: filter
  operation: build
  depends_on: [scan_train_filter]
  config: filter/protodune-sp/space_points_260210.yaml
  source_list: /path/to/train.txt
  cache_dir: ${workspace}/filter/counts
  output: ${workspace}/filter/train/accepted.yaml
  output_source_list: ${workspace}/filter/train/accepted_files.txt
  run_dir: ${workspace}/filter/train/build
```

SPINE stages consume these artifacts through `entry_filter` and
`val_entry_filter`. Apply them whenever a stage reads the corresponding raw
LArCV domain; do not reapply them to compact cache repositories.

`ntasks` controls scheduler-array splitting and, when paired with
`files_per_task`, caps concurrent array tasks. `workers` belongs specifically
to `spine-filter scan` and controls source-file inspection processes within its
single CPU job. SPINE's `num_workers` independently controls DataLoader worker
processes inside one training or inference task.

## Usage

```bash
./submit.py \
  --pipeline pipelines/my_pipeline.yaml \
  --workspace /path/to/workflow
```

Pipeline settings resolve in this order: profile defaults, pipeline `defaults`,
stage fields, global CLI overrides, then stage-specific module-weight
overrides. For example, this runs every stage with the same checkout and
scheduler account while overriding any profiles in the YAML:

```bash
./submit.py --pipeline pipelines/my_pipeline.yaml \
  --workspace /path/to/workflow \
  --spine-path /path/to/spine --profile s3df_hopper --account my_account
```

Pipeline-wide CLI overrides are supported for software paths, profiles,
scheduler resources, and first-class SPINE runtime options. Data sources,
outputs, dependencies, and run lifecycle settings remain on their individual
stages. Unknown fields and unsupported pipeline CLI options are rejected rather
than ignored.

An existing checkpoint can initialize one destination module without editing
the stable pipeline document. Qualify each override by both stage and module:

```bash
./submit.py --pipeline pipelines/my_pipeline.yaml \
  --workspace /path/to/workflow \
  --stage-module-weight train_uresnet_ppn \
    uresnet_ppn=/path/to/uresnet_ppn.ckpt \
  --stage-module-weight train_graph_spice \
    graph_spice=/path/to/graph_spice.ckpt
```

The option may be repeated. It overrides a matching `module_weight` entry in
YAML and initializes parameters without requesting training-state resume.

To continue an interrupted workflow in the same workspace, cancel or confirm
termination of its old jobs and restart at the first failed stage:

```bash
./submit.py --pipeline pipelines/my_pipeline.yaml \
  --workspace /path/to/workflow \
  --from-stage cache_train_fragment_graphs
```

Earlier stages are treated as completed. Retry attempts preserve previous
scheduler artifacts, and training resumes the latest checkpoint when one is
available. An optional `--to-stage NAME` bounds the submitted range
inclusively.

See `icarus_production_example.yaml` for a complete example.

## Generic staged training

`generic/uresnet_ppn_to_graph_spice_240805.yaml` defines the first cached model
transition:

1. Train standalone UResNet-PPN and select `snapshot-best.ckpt`.
2. Materialize its canonical `seg_pred` and adapted `clust_label_adapt`
   products, together with `ppn_points`, into separate training and validation
   sharded cache repositories.
3. Train standalone Graph-SPICE from raw LArCV truth plus the aligned cache.

Each split has one logical `.spine-cache` repository. Every source and stage
owns an immutable HDF5 V2 shard internally, while downstream jobs consume the
repository as one input. This avoids copying prior stages or maintaining
physical cache-file lists.

Cache stages declare `cache_repository` and `cache_stage`. spine-prod generates
one publication identity per submission, registers its fence inside the
scheduled job after dependencies clear, and shares it across every array task
and scheduler chunk. It derives SPINE's completion barrier from the number of
source files assigned to the stage.

The generic pipelines define their train and validation inputs once under
`collections.splits`. A stage-level `for_each` expands cache templates into
independent, concretely named jobs before dependency validation and submission.
The full-chain workflow finishes with a CPU-only `export_weights` stage that
loads every standalone `snapshot-best.ckpt` and writes one directly runnable
full-chain checkpoint plus its SHA-256 sidecar. It then evaluates that exact
checkpoint on the held-out dataset and runs a dependent CPU-only report stage:

```yaml
- name: report_full_chain
  kind: report
  depends_on: [evaluate_full_chain]
  config: test/common/full_chain/report_v1.yaml
  input_dir: ${workspace}/metrics/full_chain/raw/latest
  output_dir: ${workspace}/metrics/full_chain/report/artifacts
  run_dir: ${workspace}/metrics/full_chain/report
  checkpoint: ${workspace}/weights/full_chain_240805.ckpt
  dataset: ${validation_source}
  dataset_selection:
    entry_fraction_range: [0.5, 1.0]
  profile: s3df_milano
```

`spine-report` recursively consumes every completed analyzer CSV shard. The
resolved report recipe records the dataset and hashes the composed checkpoint;
`summary.json` and plots are written beneath the stable artifact directory.

The generic pipelines declare `workspace: null`; choose the shared output root
at launch with `--workspace /path/to/workflow`. They use SPINE's
target-qualified source overrides for the mixed Graph-SPICE dataset and
`--module-weight` for the cached segmentation jobs; it requires no generic
`--set` overrides. SPINE validates stored source provenance and fails rather
than silently pairing the wrong events.

To run against an unreleased checkout, pass `--spine-path /path/to/spine` when
submitting the pipeline. A stage-level `spine_path` remains available when only
one stage needs a different checkout.

## ProtoDUNE-SP staged training

`protodune-sp/full_chain_260210.yaml` extends the same cache-and-train model to
a chain with learned deghosting. It first scans the train and validation file
collections and rejects entries containing 500,000 or more reconstructed space
points. It then trains binary UResNet deghosting,
then materializes calibrated charge, the original-row mapping and raw
supervision exactly once. UResNet-PPN trains on that cached point domain, so
the expensive deghosting, calibration and LArCV parsing paths are not repeated
each epoch. The remaining Graph-SPICE and GrapPA transitions publish only their
new products to the split repositories.

The `260210` pipeline intentionally preserves the deployed model choices. It
is the reviewable baseline from which a new dated ProtoDUNE-SP revision can
adopt selected decisions from the generic `260828` study.

ProtoDUNE-SP cache arrays partition only the authoritative `primary` LArCV
source. Every task receives the same scalar `cache` repository path, and the
cache reader projects its immutable source shards onto that task's raw-source
subset.

### Recovering the PPN cache transition

Caches produced before `ppn_points` was retained need only their segmentation
stage rebuilt. Preserve the trained UResNet-PPN and Graph-SPICE weights by
submitting the bounded cache pair first:

```bash
./submit.py --pipeline pipelines/generic/full_chain_240805.yaml \
  --workspace /path/to/workflow \
  --from-stage cache_train_segmentation \
  --to-stage cache_validation_segmentation \
  --spine-path /path/to/fixed/spine
```

After both jobs complete, continue from the failed transition without
resubmitting Graph-SPICE training:

```bash
./submit.py --pipeline pipelines/generic/full_chain_240805.yaml \
  --workspace /path/to/workflow \
  --from-stage cache_train_fragment_graphs \
  --spine-path /path/to/fixed/spine
```
