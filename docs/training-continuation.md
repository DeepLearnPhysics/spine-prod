# Testing training across allocations

This branch implements an opt-in, submission-host controller for Slurm and PBS.
It submits new jobs and serializes the pipeline. Consumers are submitted only
when the logical training outcome is `completed` **and** the last scheduler job
has finished successfully. Paused jobs exit zero but do not release consumers.
Existing pipelines without `continuation` keep their scheduler dependencies.

## Install the SPINE runtime contract

Unmodified SPINE 1.4.4 does not distinguish a pause from successful graceful
completion. Apply the included patch to a **separate SPINE checkout** based on
the 1.4.4 driver APIs inspected for this implementation:

```bash
cd /path/to/spine-continuation-checkout
git apply --check /path/to/spine-prod/patches/spine-1.4.4-continuation-v1.patch
git apply /path/to/spine-prod/patches/spine-1.4.4-continuation-v1.patch
python bin/run.py --continuation-contract
# Must print: 1
```

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
Periodic checkpoints remain enabled. Resume uses the latest verified continuation
snapshot; existing best-checkpoint promotion remains tied to scientific validation.

Contract 1 uses `SPINE_CONTINUATION_TOKEN`, `SPINE_PAUSE_FILE`, and
`SPINE_OUTCOME_FILE`, explicitly passed through the container launcher. Rank zero
atomically publishes JSON with `contract`, `token`, `state` (`paused`/`completed`),
`reason`, `checkpoint`, `sha256`, and zero-based `iteration`, after driver cleanup.
An outcome is usable only after the launcher and scheduler finish successfully.
No decision is inferred from logs.

## First site test

Start with a small, known-good training dataset and the appropriate example:

```bash
python submit.py --pipeline pipelines/continuation/nersc.yaml \
  --workspace /shared/path/continuation-test \
  --spine-path /shared/path/spine-continuation-checkout \
  --account YOUR_ACCOUNT \
  --source-list /shared/path/train.txt \
  --val-source-list /shared/path/validation.txt --dry-run
```

For Polaris, substitute `pipelines/continuation/polaris.yaml`. Inspect the generated
scripts, then repeat without `--dry-run`. Profiles and queue limits must be suitable
for your allocation; sample timings are experimental, not production tuning.
Set the scientific epoch/iteration limit and validation policy in your training
configuration so the test lasts through at least two pauses and then completes.
Use a separate reference workspace for an uninterrupted run with the same policy,
seed, data, batch size and rank topology. Do not shorten the scientific policy for
each chunk. Normal best-checkpoint consumers/export stages can follow the training
stage; the sample uses a validation stage as its completion-gate check.

After initial submission, invoke this bounded command periodically from a
site-approved submission service (for example an approved user timer/cron job):

```bash
python scripts/continue_training.py advance \
  /shared/path/continuation-test/.spine-prod/continuation_pipeline.json
```

It queries accounting, commits state, submits at most one ready allocation/stage,
and exits. It does not need a terminal or spend GPU time waiting for later chunks.
Use the same configured submission environment and repository for every invocation.
A one-minute timer is generally sufficient; arrange the timer through the site's
approved mechanism. No timer or cluster jobs are installed by this repository.
All controller/run paths must be shared across compute and submission nodes and
support POSIX file locks, atomic rename and fsync. Compute-node submission is not
required. Slurm requeue is deliberately not used.

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
checkpoint. No automatic infrastructure retries occur. Attempts are bounded by
`max_attempts` (default 100); paused attempts making no progress fail.

Every scheduler call has a durable intent and unique `spc-...` job-name token.
If a response is lost, subsequent invocations do **not** resubmit. Find the token
in scheduler history and use `adopt` for training; it validates the job name before
recording the ID. Runtime startup can also record its own ID after validating the
token. If no job was accepted, or a non-training consumer's response was lost,
leave the controller stopped for operator reconciliation; automatic retry of
uncertain/non-training submissions is intentionally unavailable in this rollout.
Do not delete a manifest to force a retry while any allocation may still exist.

Cancellation of a submission with an unknown ID still inhibits future jobs; find
and cancel its token-matching scheduler job as well. Keep the controller inactive.

The run manifest is `RUN/.spine-prod/continuation.json`. It contains a scientific
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
.venv/bin/python -m pytest tests/test_continuation.py -o addopts='' -q
PYTHONPATH=/path/to/patched-spine/src .venv/bin/python -m pytest \
  tests/test_spine_continuation_contract.py -o addopts='' -q
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
   charging from scheduler accounting and that no automatic retry occurs on failure.
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
