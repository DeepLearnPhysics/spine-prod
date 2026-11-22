# Multi-node training

SPINE supports externally launched distributed workers. spine-prod launches one
PyTorch agent per node, and each agent launches one worker per allocated GPU.
The scheduler allocation owns the complete distributed training job.

Use `--nodes 2` with an existing pipeline `--site s3df`, `--site nersc`, or
`--site alcf` command. In site-mapped pipelines this override applies only to
training stages; filtering, caches, evaluation, checkpoint export, and reports
retain their existing allocations. A training stage can also specify `nodes: 2`
in YAML. Direct training submissions accept `--nodes` without `--site`.

The GPU count is per node: `gpus`/`--gpus` on S3DF and
`gpus_per_node`/`--gpus-per-node` on NERSC and Polaris. Thus two nodes with four
GPUs each produce `--world-size 8`. Explicit conflicting world sizes are rejected.
CPU requests remain per node-local launcher, shared by its GPU workers.

| Site | Allocation | Node-local launcher | Container |
| --- | --- | --- | --- |
| S3DF | Slurm, one task per node | `srun` → PyTorch agent | Singularity |
| NERSC | Slurm, one task per node | `srun` → PyTorch agent | Shifter |
| Polaris | PBS select, one MPI process per node | `mpiexec` → PyTorch agent | Apptainer |

Each agent uses the first allocated hostname as its rendezvous address and a
job-derived port (or explicit exported `MASTER_PORT`). Slurm process IDs or PMI
ranks identify nodes; torchrun supplies global and local GPU worker ranks.
The selected runtime must include PyTorch's distributed launcher and SPINE's
external-rank support (the default SPINE 1.5 runtime provides these).
Runtime, config, input, checkpoint and control paths must be accessible from all
nodes. Container mounts and network access must cover those paths and rendezvous.

Queue node/GPU limits still apply. Site mapping promotes NERSC shared GPU
profiles to their corresponding full-node profiles when multiple nodes are
requested, preserving the 40/80 GB GPU class. Explicit CLI profiles and QOS
choices remain authoritative and are rejected if incompatible with node counts.
Direct submissions must select a suitable full-node profile themselves. Single-node launch behavior
is unchanged. Distributed inference and multi-node filter/report processes are
not enabled by this training feature; inference arrays remain independent jobs.

## Training and continuation

`batch_size` is global and must divide evenly across the total GPU count.
`minibatch_size` is per GPU: keeping it fixed while adding nodes increases the
global batch size. Set an appropriate global batch size for a controlled scaling
comparison; the launcher does not silently change scientific training settings.

Continuation still budgets allocation wall-clock time, not GPU-hours. Doubling
nodes does not halve the runtime budget and can increase total allocation charges.
Worker counts must remain fixed across restarts. One supervisor controls the
whole scheduler launch, and the existing shared pause marker reaches SPINE rank
zero, which coordinates the distributed checkpoint. PyTorch automatic elastic
restarts are disabled; the continuation controller remains responsible for retries.
Slurm kills the remaining node tasks when a task fails; PBS uses mpiexec's job
failure handling. A failed worker must never be treated as successful completion.

## Deployment validation

Local tests exercise rank/command construction, scheduler resource mapping,
rendered shell execution through simulated Slurm/PBS/container launchers, and
failure exit propagation. They do not validate actual NCCL networking or GPU
scaling on the clusters.

Before migrating a production run on each site:

1. Run a short two-node training job; confirm distinct global ranks and the
   intended local GPU assignments, and compare results with one node.
2. Request a pause, verify one complete checkpoint with all rank state, and resume
   on a new allocation with the same topology.
3. Exercise a worker failure and confirm the allocation fails and downstream
   stages remain blocked.
4. Compare throughput, GPU-hours, and time to validation target. Confirm that
   site container/network settings use the intended interconnect before scaling
   beyond two nodes.

References: [NERSC distributed training](https://docs.nersc.gov/machinelearning/distributed-training/),
[Polaris PyTorch](https://docs.alcf.anl.gov/polaris/data-science/frameworks/pytorch/),
[PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html).
