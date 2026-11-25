# Training across scheduler allocations

Use `--site` to run pipeline training across allocations automatically, or add
`continuation` explicitly to a training stage.
A submission-host controller accounts for completed jobs, verifies checkpoint
outcomes, and resumes recoverable pauses. Downstream stages wait for logical
training completion and successful scheduler accounting; a paused job's zero
exit code does not release them. Pipelines without continuation keep their
existing scheduler dependencies.

## Runtime requirements

The submission host requires Python 3.10 or newer and `requirements.txt`. On
NERSC, load the Python module before submitting. The controller retains that
Python executable for unattended invocations.

The compute-side supervisor uses the worker host’s `python3` and requires only
Python 3.9+ and the standard library. It does not reuse the login node’s Python
path. Submission and the periodic controller still require Python 3.10+.

The training runtime must provide **SPINE continuation capability 2**, including
pending-validation replay. SPINE 1.5.0 introduced this support; **SPINE 1.5.2**
is the default runtime selected by `DEFAULT_SPINE_VERSION`. The implementation is
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
    checkpoint_margin: "00:15:00"
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
NERSC defaults to a 4–24-hour request range, with a 15-minute checkpoint margin
and five-minute startup allowance. This is a starting policy, not a measured
optimum. Non-continuation stages are never automatically shortened.
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

**S3DF submission automatically registers a regular cron entry on `sdfcron001`.**
Submit from the usual interactive node; noninteractive SSH is used only for
preflight and schedule management. Cron runs the controller locally every minute,
without an SSH session or GPU allocation. Registration uses `BatchMode=yes` and
strict host-key checking: establish trusted access to `sdfcron001` beforehand.
Authentication or preflight failures stop submission before training starts.
Dry-run describes registration but does not connect or modify crontab.

Preflight verifies the submission interpreter and dependencies on the cron host,
Slurm/cron utilities, and shared state storage. The cron entry explicitly sets
Slurm's PATH and uses a nonblocking `flock` so scheduled invocations cannot pile up.
Each pipeline retains its submission interpreter and selected container/source.
The selected Python executable (for example `/usr/bin/python3.12`) must exist
on both interactive and cron hosts; compute workers independently use Python 3.9+.

Account registries and controller logs live under
`~/.spine-prod/automation/s3df/ACCOUNT/`. Existing cron entries are preserved and
backed up before changes; installation is verified. Finished or blocked campaigns
retire their managed timer when no registered pipeline needs monitoring. Missing
state delays retirement, and explicit recovery operations re-register the timer.
Inspect the schedule from your interactive node with:

```bash
ssh -o BatchMode=yes sdfcron001 crontab -l
```

No manual polling loop is needed on S3DF. Use a fresh workspace for the automated
smoke test and check both pipeline completion and retirement of its cron entry.

**Polaris still requires an externally hosted periodic invocation.**
The generated tick script remains available for recovery on every site:

```bash
bash /path/to/continuation-dir/.spine-prod/advance.sh
```

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
python3 scripts/continue_training.py accept MANIFEST --stage train
python3 scripts/continue_training.py adopt MANIFEST --stage train --value JOB_ID
python3 scripts/continue_training.py cancel MANIFEST
python3 scripts/continue_training.py preflight STATE_DIRECTORY --site nersc
```

- `extend` sets a larger **total** budget without changing scientific configuration.
- `retry` requires a confirmed failed allocation and a checksum-valid checkpoint.
- `accept` explicitly releases consumers using an inactive, verified result; it
  does not invent a best checkpoint if validation has never produced one.
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
unsupported. Missing scheduler history blocks advancement. Polaris shrink-to-fit requires site validation. Multi-node training uses a fixed
node/GPU count across attempts; validate pause/resume on each site before production use.
See [multi-node training](multi-node-training.md).

Regression tests live in `tests/continuation/`; native training and validation
replay tests live in SPINE. Before a new deployment, exercise pause/resume and
consumer release through the actual container and distributed launcher, verify
checkpoint integrity, and measure checkpoint latency to choose a suitable margin.
