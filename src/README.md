# Source organization

`submit.py` parses command-line options and calls `Submitter`. Source modules are
internal to spine-prod; imports use their current locations without compatibility
shims for former module names.

```text
src/
├── submitter.py       # Construct shared services and dispatch CLI operations
├── config.py          # Profiles, modifiers, and composite SPINE configurations
├── files.py           # Source resolution, file validation, and chunking
├── runs.py            # Run directories, attempts, checkpoints, and validation
├── sites.py           # Site resource limits and allocation defaults
├── control.py         # Graceful job-stop requests
├── preload.py         # Materialize configuration download directives
├── client/            # Scheduler clients: shared base, Slurm, and PBS
├── submission/
│   ├── base.py        # Shared access to Submitter services
│   ├── batch.py       # Plan, submit, and record batch jobs
│   ├── validation.py  # Reject incompatible submission options
│   ├── render.py      # Render and write executable scheduler scripts
│   ├── interactive.py # Execute SPINE locally or in a container
│   ├── filter.py      # Submit file-aware entry filtering
│   ├── report.py      # Submit completed-metric report reduction
│   ├── runtime.py     # Resolve executables, containers, and bind paths
│   ├── arguments.py   # Format SPINE command-line arguments
│   └── sharding.py    # Partition joint input datasets
├── pipeline/
│   ├── fields.py     # Allowed fields and reference syntax
│   ├── validation.py # Validate pipeline and stage declarations
│   ├── definition.py # Load definitions and resolve configuration precedence
│   └── runner.py     # Submit stages in dependency order
└── continuation/
    ├── policy.py     # Budget, allocation bounds, and final-chunk decisions
    ├── state.py      # Durable manifests, locking, and atomic writes
    ├── scheduler.py  # Scheduler discovery, deadlines, and accounting
    ├── checkpoints.py # Verify outcomes and recovery checkpoint integrity
    ├── controller.py # Reconcile attempts and gate downstream stages
    ├── service.py    # Register, run, and retire unattended controller timers
    └── runtime.py    # Supervise training within the actual allocation deadline
```

## Boundaries

Submission validates options before preparing files or contacting the scheduler.
Pipeline definitions validate the complete document before the runner submits any
stage. Continuation separates scheduler observations, durable state, checkpoint
verification, and lifecycle decisions so each can be exercised independently.

The compute-side continuation supervisor uses only the Python standard library.
It must remain importable without the submission host's YAML or template packages.
The entry points in `scripts/` stay thin and retain their paths because generated
jobs and registered timers invoke them. Manifest formats and checkpoint contracts
are independent of the internal Python module layout.

## Tests

`tests/client/`, `tests/submission/`, `tests/pipeline/`, and `tests/continuation/`
follow the source packages. Root tests cover configuration, files, runs, sites,
CLI entry points, and detector configuration content. Submission integration tests
are separated by configuration, input files, batch execution, interactive
execution, and pipelines. Native checkpoint and validation-replay tests belong
to SPINE; this repository tests the controller, scheduler adapters, and runtime
contract boundary. See [training continuation](../docs/training-continuation.md).
