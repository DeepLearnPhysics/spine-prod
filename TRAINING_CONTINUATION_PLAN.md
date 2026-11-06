# Training continuation across scheduler allocations

Status: initial implementation on `feature/training-allocation-continuation`.
See [the test and rollout guide](docs/training-continuation.md) for implemented
semantics, required SPINE patch, operational limits, and outstanding site checks.
The proposal below remains the design reference; multi-rank/site effectiveness
has not yet been validated.
Date: 2026-11-06.
Suggested branch: `feature/training-allocation-continuation`.

## Problem and intended behavior

Long training runs are difficult to schedule at HPC sites with short walltime
limits or queues that favor short requests. A training stage should be able to
span several Slurm or PBS allocations without changing its scientific training
policy or exposing intermediate allocations as completed pipeline stages.

Separate three concepts:

- **Training completion:** early stopping or reaching the configured epoch/step
  limit, or an explicit user decision to accept the current result.
- **Allocation duration:** the walltime requested from the scheduler for one
  execution of the training process.
- **Training runtime budget:** a cap on cumulative allocation runtime across
  executions, excluding time spent waiting in the queue.

For example, a stage could have a 48-hour runtime budget and request allocations
between 4 and 24 hours. A scheduler may grant a long allocation when available or
start the job sooner in a shorter backfill window. This avoids forcing every
allocation to be four hours while preserving access to short opportunities.

The range does not bypass queue/QOS limits, guarantee immediate scheduling, or
require the scheduler to maximize the duration. Allocation duration flexibility
is distinct from preemption and from changing the number of GPUs.

## Proposed configuration

Illustrative syntax, subject to review before implementation:

```yaml
stages:
  - name: train_grappa_shower
    # Existing training configuration and resource declarations omitted.
    time: "24:00:00"
    continuation:
      runtime_budget: "48:00:00"
      min_time: "04:00:00"
      checkpoint_margin: "00:15:00"
```

Proposed semantics:

- `time` retains its existing meaning: maximum scheduler walltime per allocation.
- `continuation` is opt-in and initially supported only for training stages.
- `runtime_budget` includes allocation startup, training, validation, checkpoint
  writes, and shutdown. It excludes queue waits and is elapsed walltime, not
  GPU-hours or an elapsed calendar deadline.
- `min_time` is optional. Omit it for fixed-duration chunks; specify it to allow
  the scheduler to choose a duration between `min_time` and `time`.
- `checkpoint_margin` is the lead time before the effective allocation deadline
  at which training is asked to pause. It must cover the current training step,
  distributed coordination, checkpoint writing, and shutdown.
- Existing stages without `continuation` retain their behavior.

Allow site profiles to supply operational defaults, with explicit stage values
winning. Keep model/loss configuration and scientific stopping criteria separate
from these scheduling options.

Validate positive durations, `min_time <= time`, a margin shorter than the minimum
allocation, and compatibility with the selected backend. Reject unsupported
variable-duration requests with a clear message rather than silently ignoring
`min_time`.

Before each continuation, cap the next maximum walltime by the remaining budget.
If less than the normal minimum remains, permit a final shorter request only if
it satisfies the site's minimum and leaves useful time beyond startup and the
checkpoint margin. Otherwise mark the budget exhausted. Record rounding to the
scheduler's walltime granularity. The budget is an operational limit, not a
promise of precisely that many hours of gradient updates.

## One logical training stage, multiple allocations

Introduce durable lifecycle states independent of scheduler job status:

| State | Meaning | Release downstream stages? |
| --- | --- | --- |
| Pending/running | Initial or resumed allocation is queued/running | No |
| Pausing | Training is writing a restart checkpoint | No |
| Paused | Checkpoint committed; continuation required | No |
| Completed | Early stopping, epoch/step limit, or explicit accepted completion | Yes |
| Budget exhausted | Training unfinished and no usable runtime remains | No |
| Failed | Application or infrastructure failure requiring retry policy/action | No |
| Cancelled | User stopped the logical stage | No |

A successful checkpoint-and-pause is not successful training completion. An
allocation may exit cleanly after pausing, but its exit code alone must not release
pipeline consumers. Similarly, exhausting the budget must not silently export an
unfinished model as if early stopping had succeeded.

At a boundary where training actually finishes, terminal completion wins over a
concurrent walltime pause request. Budget extension resumes the same logical run;
it must not reset validation history or the scientific stopping policy.

## Existing foundation and required SPINE changes

The current repository already has:

- Scheduler-aware graceful-stop commands in `src/control.py` and scheduler clients.
- Signal-to-marker translation in the Slurm and PBS job templates.
- Run directories, resume checkpoint selection, and checksum validation in
  `src/run_manager.py`.
- Pipeline dependency construction using scheduler `afterok` in `src/pipeline.py`.

The inspected SPINE 1.4.4 installation already supports safe training-loop control
boundaries and checkpoints containing optimizer/scheduler state, validation state,
and per-rank RNG and loader state. Existing graceful stop represents intentional
successful completion. Reusing that outcome unchanged would be incorrect here.

Define a versioned SPINE/spine-prod contract for a distinct pause request and a
machine-readable outcome. Do not infer completion from log messages. Check the
required SPINE capability at submission/runtime and reject unsupported versions.

On a pause request, SPINE should:

1. Observe the request at a safe training-step boundary and coordinate all ranks.
2. Complete the in-flight optimizer step without asynchronous checkpoint writes
   racing with model updates.
3. Save full restart state, including all applicable optimizer, LR scheduler,
   mixed-precision scaler, progress, validation/early-stopping, RNG, and loader
   state. Audit which optional states are already covered by the current version.
4. Commit the checkpoint and integrity metadata before reporting `paused`.
5. Exit normally through the launcher so all ranks and child processes terminate.

Resume from the **latest valid continuation checkpoint**, not the best validation
checkpoint. Export the **best checkpoint** only when the logical stage completes,
according to the existing pipeline policy.

Operational snapshots must preserve scientific validation cadence. They must not
consume patience, introduce extra monitored LR-scheduler updates, reset the best
metric, or restart the epoch schedule. Preserve the current GPU/rank topology
initially; elastic world-size changes are outside this proposal.

Keep periodic checkpoints enabled. They provide recovery if a node fails or a
hard kill prevents the final pause checkpoint from finishing.

## Scheduler integration

### Slurm: NERSC and S3DF

For a variable-duration request, generate directives such as:

```bash
#SBATCH --time=24:00:00
#SBATCH --time-min=04:00:00
#SBATCH --signal=B:USR1@900
```

Slurm may lower the requested limit when allocating resources to fit a backfill
window. Under this mechanism, the chosen duration is fixed when resources are
allocated. Independent preemption or administrative actions remain possible.

The advance signal is relative to the actual job end time. A six-hour allocation
therefore receives its warning around 5h45m, not 23h45m. Slurm documents that the
signal can arrive up to 60 seconds early. `B:` targets the batch shell, which should
translate it into the pause marker; do not signal arbitrary training ranks.

The generated shell must be able to service the trap while the launcher runs.
Test signal handling through the actual shell, container, and distributed launcher
rather than assuming that adding a directive makes forwarding work.

Record the assigned start/end time and walltime for diagnostics and accounting.
Do not base the pause timer on the requested maximum. Do not reuse the existing
USR1 completion behavior unchanged: continuation-enabled jobs must dispatch this
warning to the new pause action while preserving explicit user completion control.

NERSC explicitly documents variable walltime as useful for backfill. S3DF uses
Slurm, so the same scheduler interface is applicable in principle; validate local
policy and behavior before enabling it in a profile. S3DF can retain long fixed
allocations by default.

### PBS: Polaris

PBS supports a comparable shrink-to-fit request:

```bash
#PBS -l min_walltime=04:00:00,max_walltime=24:00:00
```

This is a generic PBS example, not a validated Polaris recipe. Confirm acceptance
by Polaris's routing queues, applicable node/walltime limits, and the installed
scheduler configuration before enabling it. PBS also imposes restrictions on
shrink-to-fit jobs, including job arrays; start with ordinary training jobs.

For fixed-duration PBS allocations, retain the normal `walltime` request. For
shrink-to-fit jobs, obtain the assigned `Resource_List.walltime` after the job
starts, together with the scheduler start time, and calculate the remaining time.
Use a lightweight watchdog to request a pause before the assigned deadline.
Do not assume the requested maximum or an unverified environment variable is the
actual limit. Validate the query and deadline interpretation on Polaris.

A watchdog must exit when training exits, avoid signalling later allocations,
and handle deadline-query failures explicitly. Avoid frequent scheduler polling;
one startup query with bounded retries should normally suffice.

## Continuation submission and pipeline dependencies

The current pipeline submits dependencies against scheduler job IDs. Simply
resubmitting a paused job under a new ID is insufficient: downstream jobs may
still depend on the previous, successfully exited allocation.

Implement and test an explicit strategy before enabling continuation:

- **Slurm requeue:** attractive where permitted because it can retain the job ID.
  Validate dependency behavior across explicit requeues on the target site.
  `--requeue` by itself does not restart a job that reaches its walltime limit;
  the wrapper must checkpoint and arrange continuation explicitly.
- **New-job continuation:** required as a portable alternative. Downstream
  submission must be deferred until logical-stage completion, or attached to a
  tested completion-gate mechanism that cannot succeed after an intermediate
  allocation. Do not pre-submit consumers against the first chunk's `afterok`.

Prefer a durable, short-lived orchestration mechanism over a user terminal that
must stay connected. Determine whether each site permits submissions/requeues from
compute nodes and provide an appropriate submission-side alternative if necessary.
Do not spend a GPU allocation waiting for future chunks to finish.

Select one mechanism for the first implementation and document its site limits.
A scheduler-neutral lifecycle is required even if only Slurm requeue is initially
supported. Do not claim PBS portability before submission and dependency handling
have been validated there.

## Durable state, accounting, and recovery

Maintain an atomic manifest under the existing training run directory containing:

- Logical run/stage identity and training configuration digest.
- Lifecycle state, terminal/pause reason, and a state generation number.
- Attempt number, scheduler/backend, and job ID(s).
- Requested duration bounds, assigned deadline, start/end timestamps, and runtime.
- Cumulative runtime used and remaining budget.
- Latest committed checkpoint path, integrity metadata, and training progress.
- Continuation submission intent and resulting scheduler job ID.

Use locking or an equivalent single-writer protocol so repeated signals, retries,
or simultaneous controllers cannot submit duplicate continuations. Submission and
manifest writes cannot be one atomic transaction with the scheduler: persist intent,
use an identifiable submission token, and reconcile uncertain outcomes before retry.

Reset attempt-specific pause markers only after validating the intended attempt.
An old marker must not immediately stop a new allocation. A user cancellation must
inhibit subsequent automatic submissions, including races during checkpointing.

Reconcile scheduler accounting after hard kills so lost allocations do not vanish
from the total budget. Distinguish provisional elapsed-time estimates from confirmed
accounting. Charge runtime actually consumed, not the requested maximum.

Handle these conditions separately:

- Planned deadline pause: continue from the committed checkpoint.
- Early stopping or configured training limit: finalize and release consumers.
- Budget exhausted: preserve state and require extension or explicit acceptance.
- OOM/configuration error/corrupt checkpoint: fail with an actionable explanation.
- Node failure/preemption: apply a bounded, explicit infrastructure retry policy.
- Submission failure: retain paused state and recover without retraining or duplicates.

Do not implement an unbounded retry loop. Record and bound repeated infrastructure
failures and attempts that make no training progress.

## User experience and observability

Dry-run output should show the total runtime budget, per-allocation range, checkpoint
margin, selected continuation strategy, and how downstream stages are gated.

Status should report the logical stage plus its allocation history, for example:

```text
train_grappa_shower: queued for continuation
  allocation: 3
  runtime consumed: 17h 42m / 48h
  next request: 4h–24h
  latest checkpoint: iteration 123400
  last outcome: paused before allocation deadline
```

Provide clear operations to extend a budget, resume a paused stage, cancel the entire
logical stage, and intentionally accept a current result. Preserve the distinction
between pausing and the existing user-requested successful graceful completion.
Operational changes such as a budget extension must not invalidate the immutable
scientific configuration digest.

## Validation and rollout

1. **Lifecycle contract and SPINE support.** Define pause/terminal outcomes, audit
   full restart state, and test distributed pause/resume at safe boundaries.
2. **Fixed chunks on one Slurm site.** Implement durable accounting, bounded
   continuation, and correct dependency gating before variable walltime.
3. **Variable Slurm allocations.** Add `min_time`, actual-deadline handling, and
   site profile validation. Test on NERSC; keep S3DF behavior opt-in.
4. **PBS/Polaris.** Validate fixed-chunk continuation first, then shrink-to-fit if
   supported by the site's queues and submission policies.
5. **Operational tuning.** Measure startup/checkpoint overhead and queue waits;
   choose minimum durations and margins based on observed workloads.

Required automated coverage:

- Configuration validation and unchanged behavior when continuation is absent.
- Scheduler directive generation and actual-versus-requested deadline handling.
- Early stopping, epoch limits, and best-checkpoint selection across restarts.
- Continuation snapshots leaving validation cadence and patience unchanged.
- Repeated signals, stale markers, budget boundaries, and cancellation races.
- Atomic checkpoint publication, corrupt/incomplete checkpoint rejection, and
  recovery from interruption during checkpoint writing.
- Submission failure/uncertain submission recovery without duplicate jobs.
- Downstream stages remaining blocked through every intermediate allocation.
- Runtime accounting after successful pauses, failures, and hard timeouts.

Required integration checks include a small multi-rank training job through the
real container launcher, at least two continuations, a terminal early-stop outcome,
and successful downstream execution exactly once. Compare resumed progress/state
against an uninterrupted reference where deterministic execution is supported;
do not promise bitwise equivalence for nondeterministic GPU operations.

Use short test limits in an appropriate test queue. Verify both a granted duration
shorter than the maximum and a full-length allocation where possible. Measure the
worst observed checkpoint latency before settling on a production margin.

## Decisions to resolve in the separate discussion

- Final public configuration names and location of site defaults.
- First continuation backend and portable downstream gating architecture.
- Machine-readable SPINE pause/terminal outcome format and minimum supported version.
- Site permissions for requeue/submission from running jobs.
- Polaris shrink-to-fit support and queue-routing behavior.
- Budget accounting reconciliation and final short-allocation policy.
- Default infrastructure retry and no-progress limits.
- Whether to add an independent calendar deadline in a later iteration.

GPU elasticity, automatic batch-size changes, and automatic QOS selection are outside
this proposal. They alter resource or scientific behavior beyond walltime continuation.

## References

These references informed the design discussion; site-specific support still needs
validation against the actual queues used by this project.

- [Slurm sbatch: time-min and advance signals](https://slurm.schedmd.com/sbatch.html)
- [NERSC scheduling best practices](https://docs.nersc.gov/jobs/best-practices/)
- [NERSC job examples and requeue caveats](https://docs.nersc.gov/jobs/examples/)
- [S3DF Slurm documentation](https://github.com/slaclab/sdf-docs/blob/prod/slurm.md)
- [PBS resource definitions, including shrink-to-fit](https://github.com/openpbs/openpbs/blob/master/doc/man1/pbs_resources.7B)
- [Polaris job submission and queue guidance](https://docs.alcf.anl.gov/polaris/running-jobs/)
