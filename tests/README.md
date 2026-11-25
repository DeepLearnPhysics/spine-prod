# Cluster smoke tests

These helpers validate deployments; they are not production runtime entry points.

## Short UResNet smoke test

From the repository root, run the helper on each site's submission node:

```bash
bash tests/smoke_multinode.sh SITE ACCOUNT TRAIN_ROOT VALIDATION_ROOT RUN_DIR
```

For example, on S3DF:

```bash
bash tests/smoke_multinode.sh s3df neutrino:ml-dev \
  /sdf/data/neutrino/generic/mpvmpr_2020_01_v04/train.root \
  /sdf/data/neutrino/generic/mpvmpr_2020_01_v04/test.root \
  /sdf/data/neutrino/YOUR_SHARED_RUN_DIRECTORY/multinode-smoke
```

For NERSC use `nersc m5252`; for Polaris use `polaris neutrinoGPU::SpineDev`, followed by
site-local copies of those generic files and a fresh shared run directory.
The inputs must contain `sparse3d_pcluster` and `sparse3d_pcluster_semantics`.
Each file needs at least 16 entries; training is limited to the first 64 and
validation to the first 16. The helper sources its own checkout's `configure.sh`.
The submission host needs Python 3.10+ with spine-prod's requirements, even
though training runs inside a container. Sourcing `configure.sh` does not
activate Python. Activate your submission environment before invoking the helper,
or select its interpreter explicitly:

```bash
export SPINE_PROD_PYTHON=/path/to/python-environment/bin/python
"$SPINE_PROD_PYTHON" --version
"$SPINE_PROD_PYTHON" -m pip install -r requirements.txt
```

Without this override the helper uses `python3` from `PATH`. The override
selects only the submission interpreter; training uses the container's Python.

To test a custom SPINE branch without rebuilding the image, export
`SPINE_LOCAL_PATH=/shared/path/to/spine` before running the helper. The checkout
must contain `bin/run.py` (or `bin/spine`) and be readable on both compute
nodes. The helper runs this code inside the configured container, which supplies
the native dependencies. Use a fresh run directory for each comparison.

The helper defaults `NCCL_DEBUG` to `INFO` for transport diagnostics. Set
`NCCL_DEBUG=WARN` to compare SPINE progress output without NCCL debug traffic.

Before a Polaris submission, export `SPINE_CONTAINER_PATH` pointing to the
SPINE 1.5 SIF image on shared Polaris storage. PBS forwards this setting and the
NCCL diagnostic setting; the job requests `singularity_fakeroot=true` for Apptainer.
NERSC uses the configured Shifter image; S3DF uses its configured SIF image.
All input/run/repository paths must be accessible inside the site's container.

The test requests two nodes, four GPUs per node, and 30 minutes. NERSC and
Polaris use debug queues; S3DF uses the Ampere full-node profile. It trains the
existing generic UResNet for 20 steps with one event per GPU, two data-loader
workers per GPU, and validation/checkpointing every 10 steps. Early stopping is
disabled. This is an ordinary training job, not a continuation pipeline, so no
cron registration or continuation state directory is involved.

You may append `--dry-run` to inspect the submission. Use a separate fresh
`RUN_DIR` for the real submission because a dry-run writes run metadata too.

After scheduler completion, check `RUN_DIR/latest/stdout.log`, `stderr.log`,
training/validation CSV logs, and `RUN_DIR/weights/`. Success means scheduler
exit status zero, eight participating ranks across two hosts (NCCL INFO logs),
20 training steps with finite losses, validation, and saved checkpoints at steps
10 and 20 (zero-based iteration indices 9 and 19). This tests real training and
checkpoint writes; automatic cross-allocation resume is a separate follow-up test.
