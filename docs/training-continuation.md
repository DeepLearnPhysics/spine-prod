# Training across scheduler allocations

Use `--site` to run pipeline training across allocations automatically, or add
`continuation` explicitly to a training stage.
A submission-host controller accounts for completed jobs, verifies checkpoint
outcomes, and resumes recoverable pauses. Downstream stages wait for logical
training completion and successful scheduler accounting; a paused job's zero
exit code does not release them. Pipelines without continuation keep their
existing scheduler dependencies.

## Runtime requirements

The submission host requires Python 3.9 or newer and `requirements.txt`. On
NERSC, load the Python module before submitting. The controller retains that
Python executable for unattended invocations.

The training runtime must provide **SPINE continuation capability 2**, including
pending-validation replay. **SPINE 1.5.0** provides this support and is the
default runtime selected by `DEFAULT_SPINE_VERSION`. The implementation is
maintained in SPINE. Check the executable inside the selected training environment:

```bash
spine --continuation-contract
# Expected output: 2
```

The launcher performs this check before continuation training. The repository's
baseline SPINE version requirement alone does not establish continuation support.
Keep both repositories and runtime environments fixed throughout a running
campaign; updating a checkout does not update already queued scripts.

## Configure a training stage

Keep the scientific stopping criteria, validation cadence, and best-checkpoint
policy in the SPINE model configuration. Add allocation policy to the pipeline
stage, for example:

```yaml
- name: train
  stage: train
  profile: nersc_gpu_full
  config: train/generic/uresnet_ppn/train_240805.yaml
  run_dir: ${workspace}/train
  source_list: ${train_source_list}
  val_source_list: ${validation_source_list}
  time: "48:00:00"
  continuation:
    runtime_budget: "120:00:00"
    preferred_time: "24:00:00"
    checkpoint_margin: "00:02:00"
    startup_allowance: "00:05:00"
    max_infrastructure_retries: 2
    max_attempts: 100
```

`time` bounds one allocation, `preferred_time` selects the requested chunk, and
`runtime_budget` limits cumulative allocated runtime. Queue waits are excluded;
startup, validation, checkpointing, and shutdown consume the budget. Checkpoint
margins can therefore cause a nominal 24-hour budget with six-hour requests to
need more than four allocations. A final fragment is submitted only if the site
minimum and startup/checkpoint allowances fit.

Optional `min_time` permits shorter Slurm grants. The supervisor uses the actual
assigned deadline, never a timer based on the requested maximum. Choose a margin
that covers the slowest load/forward operation, checkpoint publication, and
shutdown: a running batch cannot be interrupted safely. Site limits still apply.
When launching a portable pipeline with `--site`, training stages automatically
use continuation. Their original YAML `time` (falling back to the source profile)
becomes the total budget unless `continuation.runtime_budget` explicitly overrides
it. Allocation ceilings are bounded by the destination queue's hard limit.
CLI `--time` changes only the allocation ceiling; it does not replace the original
budget. Without `--site`, continuation still requires an explicit mapping. Site-level `continuation_defaults` provide chunk preferences;
profile defaults and explicit stage settings take precedence. Inherited chunk
preferences shrink to fit shorter stage ceilings. Explicit CLI `--time` and
stage chunk preferences are validated as supplied, so invalid requests fail.
NERSC defaults to a 1–24-hour request range, with a two-minute checkpoint margin
and five-minute startup allowance. This is a starting policy, not a measured
optimum. Cache stages have a separate site allocation policy described below;
other non-continuation stages retain their requested time.
PBS shrink-to-fit additionally requires `allow_pbs_shrink_to_fit: true` and local
queue validation; fixed-duration PBS allocations do not require this flag.

Dry-run reports total budget, allocation bounds, preferred duration, checkpoint
margin, startup allowance, retry limits, and exhaustion policy for every training
stage, including stages deferred behind earlier work.

## Storage and submission

The workspace and `run_dir` retain their existing meaning: scientific data,
logs, job scripts, outputs, and checkpoints. Continuation state defaults to the
workspace; `--continuation-dir` moves controller manifests, locks, per-attempt
control/outcome files, and the controller tick script to a separate shared root.

**NERSC requires an explicit Scratch continuation directory**, even when the
workspace is already on Scratch. CFS can still hold scientific outputs. Missing
state-directory options and paths outside recognized Scratch mounts fail before
submission, including dry-run. Symlinks are resolved before validation.

For an existing pipeline containing continuation stages:

```bash
module load python
source configure.sh
python3 submit.py --pipeline /path/to/pipeline.yaml \
  --workspace /global/cfs/cdirs/YOUR_PROJECT/training-run \
  --continuation-dir "$SCRATCH/training-run-state" \
  --cache-dir "$SCRATCH/training-run-cache" \
  --account YOUR_ACCOUNT --dry-run
```

Inspect the plan, then remove `--dry-run` to submit. For portable pipeline
profiles, add `--site nersc`. Polaris and S3DF allow colocated state or a separate
shared continuation directory.

Both storage roots must remain available to submission and compute nodes.
Continuation storage needs POSIX locks, atomic rename, and fsync; checkpoint
storage needs committed publication and integrity checks. The supervisor tests
state storage before launching training. Keep manifests for the lifetime of the
campaign, including while jobs are queued. Deleting them loses accounting and
submission history.

### Working caches and optional durable copies

`--cache-dir ROOT` relocates pipeline-declared `cache_repository` and `cache_dir`
paths, including downstream references and configuration overrides. Cache paths
must originally be under the workspace. For example, `WORKSPACE/cache/train.spine-cache`
becomes `ROOT/train.spine-cache`; `WORKSPACE/filter/counts` becomes `ROOT/filter/counts`.
NERSC requires these working caches on Scratch because publication uses file locks
that CFS/DVS does not support. S3DF can keep everything under `/sdf`; Polaris keeps
its existing shared-storage layout. `--continuation-dir` alone does not move caches.

Optionally add `--cache-archive-dir /durable/path/cache-archive`. After each cache
producer succeeds, a CPU batch job copies its committed cache repository and
verifies checksums before publishing an immutable snapshot. Consumers wait for
this archive job to succeed. Unchanged shards are shared across snapshots. The
working cache stays in place; the archive needs no filesystem locks and can live
on NERSC CFS. Filter count caches are relocated but are not archived by this option.
Archive jobs use the site's CPU profile and the producer's account.

After Scratch data is lost, explicitly restore a selected snapshot into a **new**
working directory:

```bash
python3 scripts/cache_archive.py restore \
  /durable/path/cache-archive/train.spine-cache/snapshots/SNAPSHOT_HASH.json \
  "$SCRATCH/restored-cache/train.spine-cache"
```

Restoration verifies every shard and clears old publication fences. It restores
cache data only: it does not resume a controller, skip pipeline stages, or check
whether the data matches a new training configuration. Select the appropriate
snapshot and stage when reusing it.

### Historical monitoring records

New continuation pipelines write a verified audit under `WORKSPACE/monitoring`
at terminal outcomes and before removing their managed timer. `--audit-dir PATH`
selects another destination; choose durable storage if the workspace is on Scratch.
Each pipeline gets immutable JSON snapshots and a `latest.json` index. These contain
the pipeline and logical-stage manifests, available attempt outcomes and failure
reports, controller output, and stage summaries with job IDs and runtime totals.
They exclude cache payloads and checkpoints, which retain their own storage policy.

These are historical records, with no restore or job-reactivation operation.
If archival fails, the timer remains registered and retries on a later tick;
completed training is not resubmitted. Existing campaigns created without an audit
destination retain their previous behavior.

## Automatic advancement

NERSC submission installs and verifies a managed `scrontab` entry before launching
training. One recurring task per user/account checks registered pipelines roughly
every minute; scheduling can add latency. You can log out after submission.

The account registry and timer log live under
`$SCRATCH/.spine-prod/automation/ACCOUNT/`. Each pipeline keeps its manifest,
`advance.sh`, and controller log under `CONTINUATION_DIR/.spine-prod/`.
Registration preserves unrelated schedule entries and fails before training if
installation cannot be verified. Repeating the same launch verifies or repairs
registration without duplicating the managed entry.

When no registered pipeline needs monitoring, the controller removes its own
scrontab block. Logs and registry state remain. A new launch or explicit recovery
operation restores the timer. Missing or unreadable state delays retirement.
Inspect registration and progress with:

```bash
scrontab -l
python3 scripts/continue_training.py status \
  "$SCRATCH/training-run-state/.spine-prod/continuation_pipeline.json"
```

**Polaris and S3DF require an externally hosted periodic invocation**; automatic
hosting is currently implemented only for NERSC. Their launch output explains
this requirement. The generated tick script is also available for recovery:

```bash
bash /path/to/continuation-dir/.spine-prod/advance.sh
```

## Update an existing campaign

Changing a pipeline YAML or site default does not rewrite a controller's saved
submission options. For an active campaign, use:

```bash
python3 scripts/update_continuation.py STATE/.spine-prod/continuation_pipeline.json \
  --workspace /durable/original/workspace \
  --min-time 01:00:00 --cache-dir "$SCRATCH/campaign-cache"
```

This previews the affected stages. Add `--apply` to save the update under the
controller lock, after writing and verifying a backup beside its manifest.
Completed stages, checkpoint identities, consumed budgets, existing allocation
records, and scheduler jobs remain unchanged. Future submissions use the new
minimum. `--checkpoint-margin 00:02:00` similarly changes the margin for future
allocations; it does not change a running supervisor's deadline or saved policy.
For a margin-only update, omit `--cache-dir` to preserve already relocated caches.
Current pending Slurm jobs can separately be updated with
`scontrol update JobId=JOB_ID TimeMin=01:00:00`; no cancellation is needed.

Use `--cache-ntasks 100` to change the file-grouping target of existing waiting
cache stages. This requires all cache producers to be waiting and no explicit
`files_per_task` policy. It preserves walltimes, cache paths, and training options.
Use `--cache-time 02:00:00` to change the fixed walltime for waiting cache jobs;
this also requires all producers to be waiting and validates site allocation limits.
It can be combined with `--cache-ntasks`, or used alone without changing grouping.
New pipelines mapped with `--site nersc` default to this target; other sites keep
the pipeline's cache task counts. The target is configured by the site's
`cache_ntasks` setting. NERSC also defaults cache jobs to a fixed two-hour request
(`cache_time`), a provisional budget pending measured runtimes. Explicit launch
walltime overrides take precedence. Explicit `files_per_task` policies retain their
existing grouping and walltime semantics. These defaults apply to cache producers,
not training, filtering, evaluation, or reporting.
With no `files_per_task`, files are grouped in batches of `ceil(num_files / ntasks)`
(minimum one), so fewer than 100 files produce one nonempty task per file.

Cache relocation is limited to campaigns where **all cache producers are still
waiting**, and their old repositories are absent or empty. It rewrites future
cache outputs and consumer references, while preserving completed filter stages.
This is not a general cache-data migration. Existing controllers do not acquire
optional archive jobs or historical-audit destinations through this command.
Continue using the existing controller and state directory; do not create a fresh
campaign to change these operational settings. The initial launch digest is kept
as provenance, and updates are recorded in `configuration_updates`.

## Completion and recovery

An allocation pause saves complete restart state without forcing validation.
Scheduled checkpoints retain the global iteration cadence. Interrupted validation
discards partial metrics and replays before further training; completed validation
alone updates best-checkpoint selection, early-stopping patience, and validation
schedulers. SPINE owns these semantics and checkpoint implementation.

The controller distinguishes scientific completion, allocation expiry, budget
exhaustion, preemption, and application failure. Budget exhaustion blocks
consumers, even when a valid checkpoint exists. The best checkpoint and latest
restart checkpoint remain available in the run directory.

```bash
python3 scripts/continue_training.py status MANIFEST
python3 scripts/continue_training.py advance MANIFEST
python3 scripts/continue_training.py metrics MANIFEST > allocations.csv
python3 scripts/continue_training.py extend MANIFEST --stage train --value 168:00:00
python3 scripts/continue_training.py retry MANIFEST --stage train
python3 scripts/continue_training.py retry-cache MANIFEST --stage cache_train_deghosting --value 00:30:00
python3 scripts/continue_training.py accept MANIFEST --stage train
python3 scripts/continue_training.py accept-best MANIFEST --stage train
python3 scripts/continue_training.py adopt MANIFEST --stage train --value JOB_ID
python3 scripts/continue_training.py cancel MANIFEST
python3 scripts/continue_training.py preflight STATE_DIRECTORY --site nersc
```

- `extend` sets a larger **total** budget without changing scientific configuration.
- `retry` requires a confirmed failed allocation and a checksum-valid checkpoint.
- `retry-cache` rearms a failed cache stage only after all its scheduler jobs
  have terminal accounting and every downstream stage is still waiting. It
  preserves failed job IDs/accounting in `retry_history`; the next tick creates
  a new attempt and publication ID. Optional `--value` changes this stage's
  walltime only. It reruns inference; it does not adopt unpublished shards.
- `accept` explicitly releases consumers using an inactive, verified result; it
  does not invent a best checkpoint if validation has never produced one.
- `accept-best` explicitly ends an accounted failed, paused, or budget-exhausted
  stage using a checksum-verified `weights/snapshot-best.ckpt`. It preserves the
  attempt history and records the accepted artifact separately. Consumers must
  already reference that best checkpoint (as the staged detector pipelines do).
  No training allocation or sampler-state restore is needed.
- `adopt` resolves a lost submission response after verifying the job-name token.
- `cancel` durably inhibits future submissions before cancelling known jobs.
- `metrics` exports allocation timing, progress, and checkpoint latency.

Automatic infrastructure retries are opt-in (default zero). Slurm preemption,
node/boot failure, and timeout/deadline can recover only from a verified periodic
checkpoint newer than the attempt's starting progress. Application errors, OOM,
and cancellation never trigger automatic retries. PBS nonzero exits require
explicit recovery until site-specific failure classification is validated.
All attempts, including clean pauses, obey `max_attempts`.

Submission intent and scheduler accounting are persisted durably. A lost scheduler
response never triggers blind resubmission: training jobs are discovered by their
unique token or left for operator reconciliation. Unknown non-training submissions
also block progression. Never delete a manifest to force retry while a job may
still exist. Cancellation with an unknown job ID requires the operator to find
and cancel that scheduler job separately.

## Limits

Continuation pipelines currently serialize all stages. Explicit profiles are
required; stage slicing, automatic cleanup, and changes in GPU world size are
unsupported. Missing scheduler history blocks advancement. Polaris shrink-to-fit
and multi-node restart behavior require site validation.

Regression tests live in `tests/continuation/`; native training and validation
replay tests live in SPINE. Before a new deployment, exercise pause/resume and
consumer release through the actual container and distributed launcher, verify
checkpoint integrity, and measure checkpoint latency to choose a suitable margin.
