# Testing training across allocations

This branch implements an opt-in, submission-host controller for Slurm and PBS.
It submits new jobs and serializes the pipeline. Consumers are submitted only
when the logical training outcome is `completed` **and** the last scheduler job
has finished successfully. Paused jobs exit zero but do not release consumers.
Existing pipelines without `continuation` keep their scheduler dependencies.

## Submission-host Python

Use Python 3.9 or newer for `submit.py` and the continuation controller. On
Perlmutter, the default system `python3` may be too old. Load the Python module
before running the submission commands:

```bash
module load python
python3 --version
python3 -c 'import yaml, jinja2'
```

If the dependencies are missing, install `requirements.txt` into your own Python
environment. Use this environment for periodic controller invocations as well.
This host-side Python requirement is separate from SPINE inside the container.
See [NERSC Python setup](https://docs.nersc.gov/development/languages/python/using-python-perlmutter/).

## Install the SPINE runtime contract

Unmodified SPINE 1.4.4 does not distinguish a pause from successful graceful
completion. Apply the included patch to a **separate SPINE checkout** based on
the 1.4.4 driver APIs inspected for this implementation:

```bash
cd /path/to/spine-continuation-checkout
git apply --check /path/to/spine-prod/patches/spine-1.4.4-continuation-v4.patch
git apply /path/to/spine-prod/patches/spine-1.4.4-continuation-v4.patch
python bin/run.py --continuation-contract
# Must print: 2
```

If the checkout already has the v1 patch (as in the first two site tests),
apply the v1-to-v2 upgrade below, then the v2-to-v3 and v3-to-v4 upgrades:

```bash
cd /path/to/spine-continuation-checkout
git apply --check /path/to/spine-prod/patches/spine-continuation-v1-to-v2.patch
git apply /path/to/spine-prod/patches/spine-continuation-v1-to-v2.patch
python bin/run.py --continuation-contract
# Must print: 2
```

For a checkout already patched to v2 (including the third site test), apply the v2-to-v3 upgrade, followed by v3-to-v4:

```bash
cd /path/to/spine-continuation-checkout
git apply --check /path/to/spine-prod/patches/spine-continuation-v2-to-v3.patch
git apply /path/to/spine-prod/patches/spine-continuation-v2-to-v3.patch
```

For a checkout already patched to v3, apply only the latest upgrade:

```bash
cd /path/to/spine-continuation-checkout
git apply --check /path/to/spine-prod/patches/spine-continuation-v3-to-v4.patch
git apply /path/to/spine-prod/patches/spine-continuation-v3-to-v4.patch
```

Revision v4 changes the default for the existing `validation.initial` option:

```yaml
validation:
  # Omit initial to validate loaded weights, but skip random initialization.
  initial: false  # Explicitly skip baseline validation, including warm starts.
```

Set `initial: true` to request a baseline even for random initialization. The
automatic default includes module-only weight imports. Continuation resumes do
not repeat baseline validation; unfinished validation is always replayed before
training, regardless of `initial`. Scheduled validation remains unchanged.

Patch revision v3 fixes checkpoint/validation scheduling across resumes; the
runtime capability contract remains **2**, since its interface is unchanged.
Scheduled checkpoints count from global iteration zero. For example, with a
980-step save period, a resume at iteration 526 still validates after iteration
979, rather than waiting for 980 additional steps. Allocation-only snapshots do
not shift this schedule or trigger extra validation.

The NERSC and Polaris 20-minute smoke tests reserve two minutes for checkpointing.
The scheduler warning and runtime deadline use the same `checkpoint_margin`.
Longer production runs can choose a larger margin for their checkpoint sizes and
filesystem performance. Existing manifests and queued scripts retain the policy
with which they were created; pulling changes does not rewrite a running test.

Use that checkout with `--spine-path`, or rebuild your container with the patch.
The actual executable is capability-checked inside the container before training;
a version number alone is insufficient. Submission hosts need not import SPINE.
The patch is kept in this repository because the upstream SPINE repository is
separate; it does not alter your installed SPINE or another working checkout.

The patch uses the existing synchronous, collective checkpoint boundary. Pause-only
snapshots skip validation, best-metric selection, patience updates and validation
LR updates. Scheduled validation remains scheduled, and early stopping or reaching
the total training limit wins over a simultaneous pause. Model/optimizer/LR state,
validation history, per-rank RNG and loader state are saved by the existing
checkpoint machinery. SPINE 1.4.4's inspected model manager has no AMP scaler;
a future AMP implementation must serialize its scaler before declaring this
contract supported. World-size changes and missing rank runtime state are rejected.
If a pause arrives during scheduled validation, all ranks stop at a safe batch
boundary and discard partial metrics. The full checkpoint records
`validation.pending_validation` (iteration, epoch and scheduling context).
Resume reruns that validation before any training step, even if the training
limit was reached. Only a completed pass updates best selection, patience and
the validation scheduler. Training RNG state is restored around validation;
the training loader is separate and remains checkpointed. Partial validation
CSV logs are archived with an `.interrupted-...` suffix before rerunning.

A blocking load/forward cannot be interrupted safely mid-batch: the margin must
cover the slowest such operation plus checkpointing and shutdown. The NERSC
smoke example now uses five minutes; the second site test took approximately
2m38s to reach its first scheduled-validation result, exceeding the previous
two-minute margin. This is a test setting, not a universal production margin.

Periodic checkpoints remain enabled. Resume uses the latest verified continuation
snapshot; existing best-checkpoint promotion remains tied to scientific validation.

Runtime capability 2 retains outcome schema 1 and uses `SPINE_CONTINUATION_TOKEN`, `SPINE_PAUSE_FILE`, and
`SPINE_OUTCOME_FILE`, explicitly passed through the container launcher. Rank zero
atomically publishes JSON with `contract`, `token`, `state` (`paused`/`completed`),
`reason`, `checkpoint`, `sha256`, and zero-based `iteration`, after driver cleanup.
An outcome is usable only after the launcher and scheduler finish successfully.
No decision is inferred from logs.

## First site test

The workspace and training `run_dir` keep their existing meaning: data, training
logs, job scripts, outputs and checkpoints can stay on **CFS**. By default,
continuation state shares the workspace. Use `--continuation-dir` to move only
controller manifests/locks, per-attempt control/outcome files and the controller
tick script to a separate shared directory.

**NERSC requires an explicit `--continuation-dir "$SCRATCH/..."`.** Missing
options, CFS/Home paths and paths outside recognized Scratch mounts fail before
submission, including dry-run. Symlinks are resolved before validation; even a
workspace already on Scratch requires the explicit option. Other sites retain
the colocated default and can also opt into a separate state directory.

The failed `cont_debug` run hit errno 524 on a compute-side manifest lock on CFS,
before training started. Moving these locks fixes that storage assumption without
moving scientific outputs. See
[NERSC DVS file-locking restrictions](https://docs.nersc.gov/performance/io/dvs/#do-not-use-file-locking).
The supervisor also tests locking, atomic publication and fsync in the state
directory before launching. Keep this directory available for the entire run:
Scratch retention policies apply to controller state too. Do not change or delete
it to restart an existing pipeline; that would discard its job/accounting history.

Start a **new** workspace with a small, known-good dataset and the example:

```bash
python submit.py --pipeline pipelines/continuation/nersc.yaml \
  --workspace /global/cfs/cdirs/YOUR_PROJECT/continuation-test \
  --continuation-dir "$SCRATCH/continuation-test-state" \
  --spine-path /shared/path/spine-continuation-checkout \
  --account YOUR_ACCOUNT \
  --source-list /shared/path/train.txt \
  --val-source-list /shared/path/validation.txt --dry-run
```

For Polaris, use a workspace on its shared filesystem and omit
`--continuation-dir` to keep state there, or supply a separate shared path. Substitute
`pipelines/continuation/polaris.yaml`. Inspect the generated
scripts, then repeat without `--dry-run`. Profiles and queue limits must be suitable
for your allocation; sample timings are experimental, not production tuning.
Set the scientific epoch/iteration limit and validation policy in your training
configuration so the test lasts through at least two pauses and then completes.
Use a separate reference workspace for an uninterrupted run with the same policy,
seed, data, batch size and rank topology. Do not shorten the scientific policy for
each chunk. Normal best-checkpoint consumers/export stages can follow the training
stage; the sample uses a validation stage as its completion-gate check.

On **NERSC**, real submission automatically installs and verifies a managed
`scrontab` entry before submitting training. You can log out after submission.
There is one recurring task per user/account, checking registered pipelines
approximately once per minute. Scheduling can add latency; this is not a
guaranteed one-minute turnaround. No GPU allocation is used for monitoring.

The account registry and timer log live under
`$SCRATCH/.spine-prod/automation/ACCOUNT/`. Each pipeline retains its own
`CONTINUATION_DIR/.spine-prod/advance.sh`, pinned submission Python and container
settings, and `controller.log`. Keep the repository and Python environment
available for the whole campaign. The controller clears inherited Slurm resource
variables before submitting GPU jobs.

Registration preserves unrelated scrontab entries, saves the previous schedule
before edits, and verifies the installed schedule. Registration failure stops
the launch before training is submitted. Rerunning the same launch repairs or
verifies registration without duplicating a managed entry. Do not edit scrontab
concurrently with registration. Dry-run prints the automation plan without
registering a timer or submitting a job.

Inspect automation with:

```bash
scrontab -l
python scripts/continue_training.py status \
  "$SCRATCH/continuation-test-state/.spine-prod/continuation_pipeline.json"
```

Status reports registration, controller errors and stale heartbeats. The timer
stops advancing completed, cancelled, failed or budget-exhausted pipelines.
When none of the account's registered pipelines needs monitoring, the controller
automatically removes only its marked scrontab block. Other accounts and unrelated
scheduled tasks are preserved. Registry state and logs remain available for review.
Missing/unreadable pipeline state defers retirement rather than assuming completion.

A new launch, budget extension, explicit retry, acceptance or job adoption restores
and verifies the timer before changing the pipeline's operational state.
Registration, reactivation and retirement share a lock, so cleanup cannot remove
the timer during a concurrent reactivation. Status reports `stopped` after verified
removal; it may report `retiring` if Slurm terminates the current cron invocation
when its own entry is removed. No training restart is needed to deploy this change.

To disable the service manually, remove only its marked block using `scrontab -e`.
NERSC cancellation/maintenance can disable entries: inspect the schedule and
rerun the launch to restore the managed block when needed.

**Polaris and S3DF do not yet have automatic hosting implemented.** Submission
prints this limitation; arrange an approved periodic invocation there. On any
site, this command remains available for manual recovery and debugging:

```bash
bash "$SCRATCH/continuation-test-state/.spine-prod/advance.sh"
```

You can test continuation storage independently with:

```bash
python scripts/continue_training.py preflight "$SCRATCH/continuation-test-state" --site nersc
```

A submission-host preflight alone cannot prove compute-host filesystem support.
Both roots must be shared across compute and submission nodes. Continuation
storage must support POSIX file locks, atomic rename and fsync; training storage
must support committed checkpoint publication and integrity checks. Compute-node
submission is not required. Slurm requeue is deliberately not used.

See [NERSC scrontab](https://docs.nersc.gov/jobs/workflow/scrontab/), including
its guidance on inherited Slurm variables and disabled entries.

For variable NERSC allocations, enable `min_time` after fixed-chunk testing.
Slurm receives `--time-min` and `--signal=B:USR2@<margin>`; USR1 retains explicit
successful graceful completion. The shell stays in an interruptible wait and
forwards signals only to the supervisor. The supervisor queries assigned start/end
from `scontrol show job --json`; a watchdog also covers warnings missed during
startup. It never starts a timer based on requested maximum walltime.

Polaris fixed chunks use `walltime`; the supervisor queries `qstat -f -F json`
for `stime` and assigned `Resource_List.walltime`. Optional PBS shrink-to-fit emits
`min_walltime,max_walltime` and requires `allow_pbs_shrink_to_fit: true` after local
queue/routing validation. This flag is an operator assertion, not evidence of
Polaris support. Startup queries retry three times, then fail closed. Missing
accounting/history blocks advancement; retain job history on the submission host.

## Multi-day VD training

Keep scientific stopping criteria in the model configuration. The continuation
budget is cumulative **allocated runtime**, excluding queue wait; it can exceed
the site's single-allocation maximum. For example, add this to each training
stage in the VD pipeline (inference/export stages do not need continuation):

```yaml
time: "48:00:00"
continuation:
  runtime_budget: "120:00:00"
  preferred_time: "24:00:00"
  checkpoint_margin: "00:15:00"
  startup_allowance: "00:05:00"
  max_infrastructure_retries: 2
  max_attempts: 100
```

Use `--site nersc --account YOUR_ACCOUNT` with portable pipeline roles.
Here `time` is the allocation ceiling, `preferred_time` is the requested chunk
length, and `runtime_budget` is the stage's total allowance. Set preferred time
to 48 hours if you want full-length allocations. Optional `min_time` allows
shorter grants; actual assigned deadlines always drive checkpoint timing.
Site profile minimums cannot be lowered by stage overrides. A final fragment
is submitted only if the site minimum, startup allowance and checkpoint margin
fit; otherwise the stage becomes `budget_exhausted` and blocks consumers.
Use `extend` to raise the total budget, or explicitly `accept` a verified result.

Dry-run prints each training stage's total budget, allocation bounds, preferred
chunk, checkpoint margin, startup allowance, site minimum, retry/attempt limits
and exhaustion policy, including stages deferred behind earlier work.

The controller records distinct outcome/failure reasons for early stopping,
training limits, allocation deadlines, budget exhaustion, Slurm preemption and
application failure. Scheduler accounting is charged exactly once. The same run
directory and weights directory persist across allocations, retaining the best
checkpoint as well as the latest restart checkpoint. Downstream stages remain
blocked until logical completion and successful terminal accounting.

## Operations and recovery

```bash
python scripts/continue_training.py status MANIFEST
python scripts/continue_training.py metrics MANIFEST > allocations.csv
python scripts/continue_training.py extend MANIFEST --stage train_uresnet_ppn --value 02:00:00
python scripts/continue_training.py advance MANIFEST
python scripts/continue_training.py cancel MANIFEST
python scripts/continue_training.py accept MANIFEST --stage train_uresnet_ppn
python scripts/continue_training.py retry MANIFEST --stage train_uresnet_ppn
python scripts/continue_training.py adopt MANIFEST --stage train_uresnet_ppn --value JOB_ID
```

`extend` sets a new **total** budget; it does not change scientific configuration.
`advance` resumes an accounted paused stage. `cancel` first durably inhibits
submissions, then cancels known jobs. `accept` intentionally releases consumers
using an inactive, verified result. It does not invent a best checkpoint if no
scheduled validation has produced one. `retry` is explicit recovery after a
confirmed failed allocation; it selects the latest checksum-valid periodic
checkpoint. Automatic infrastructure recovery is opt-in through
`max_infrastructure_retries` (default 0). Slurm preemption, node/boot failure,
and allocation timeout/deadline can resume only from a checksum-valid periodic
checkpoint newer than the attempt's starting progress. Application errors, OOM
and cancellation never trigger automatic retries. PBS nonzero exit statuses
remain failures requiring explicit recovery until site-specific termination
classification is validated. Attempts are bounded by
`max_attempts` (default 100); paused attempts making no progress fail.

Every scheduler call has a durable intent and unique `spc-...` job-name token.
If a response is lost, subsequent invocations do **not** resubmit. For training,
the controller queries scheduler history for the exact token and adopts a unique
matching job. No match blocks advancement; multiple matches require intervention.
The explicit `adopt` operation also validates the job name before recording the ID. Runtime startup can also record its own ID after validating the
token. If no job was accepted, or a non-training consumer's response was lost,
leave the controller stopped for operator reconciliation; automatic retry of
uncertain/non-training submissions is intentionally unavailable in this rollout.
Do not delete a manifest to force a retry while any allocation may still exist.

Cancellation of a submission with an unknown ID still inhibits future jobs; find
and cancel its token-matching scheduler job as well. Keep the controller inactive.

With colocated state, the run manifest is `RUN/.spine-prod/continuation.json`.
With a separate state root, it is
`CONTINUATION_DIR/runs/<run-path-hash>/.spine-prod/continuation.json`; the pipeline
manifest and `advance.sh` are in `CONTINUATION_DIR/.spine-prod/`. It contains a scientific
configuration/input/topology digest, generation, state, verified checkpoint,
budget, attempts, submission intent, requested bounds, rounding, assigned deadline,
provisional wrapper elapsed time, and confirmed scheduler accounting. Queue waits
are excluded. `sacct ElapsedRaw` or PBS `resources_used.walltime` charges actual
allocation runtime, including hard kills. A final request is rounded down to
scheduler granularity (Slurm minutes, PBS seconds); it may be below `min_time`
only when it meets `site_min_time` and leaves time beyond startup plus the margin.
Budget exhaustion never releases a consumer.

Profiles may supply `continuation_defaults`; explicit stage values win. Presence
of a stage `continuation` mapping is always required to opt in. Specify explicit
profiles for every stage. This first controller serializes independent stages too;
stage slicing and automatic cleanup are rejected. GPU elasticity is unsupported.

## Validation and measurement

Local regression suite:

```bash
.venv/bin/python -m pytest tests/continuation/test_controller.py tests/continuation/test_service.py -o addopts='' -q
PYTHONPATH=/path/to/patched-spine/src .venv/bin/python -m pytest \
  tests/continuation/test_spine_contract.py -o addopts='' -q
```

On each site, verify all of the following before production use:

1. Run through the actual shell, container and multi-rank launcher; obtain two
   paused outcomes followed by terminal early stopping or the configured limit.
2. Confirm that no consumer job exists while paused and that the consumer runs
   exactly once after completion. Test explicit acceptance separately.
3. Compare optimizer/LR state, global progress, RNG/loader state, validation
   history, patience and best-checkpoint selection to the uninterrupted reference.
   Nondeterministic GPU kernels may prevent bitwise equivalence.
4. Test USR2 pause, USR1 successful completion, cancellation during checkpointing,
   a hard timeout, and corrupt/incomplete checkpoint rejection. Confirm runtime
   charging from scheduler accounting; verify only the configured recoverable failures retry, within their limit.
5. For variable allocations, capture a grant shorter than the maximum and a full
   grant. Verify the assigned deadline in the manifest against scheduler output.
6. Record queue wait, startup overhead, checkpoint latency, allocated runtime,
   steps advanced, and time to the same validation target. Compare fixed long,
   fixed short and variable requests. Set the margin above the worst observed
   multi-rank checkpoint/shutdown latency with an operational buffer.

`metrics` exports one CSV row per allocation with requested/assigned duration,
queue time, confirmed/provisional runtime, progress, and pause-to-checkpoint-commit
latency. Queue time is measured from submission intent to scheduler start. Compare
these exports between experiments alongside your validation curves.

Local tests do not establish NERSC scheduling policy, Polaris shrink-to-fit support,
real multi-rank termination, or training effectiveness. Those are explicit rollout
checks, and no cluster execution has been performed by this implementation.

Scheduler references: [Slurm sbatch](https://slurm.schedmd.com/sbatch.html),
[NERSC scheduling guidance](https://docs.nersc.gov/jobs/best-practices/), and
[Polaris running jobs](https://docs.alcf.anl.gov/polaris/running-jobs/).
