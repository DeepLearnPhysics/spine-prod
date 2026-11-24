#!/usr/bin/env bash
# Short, ordinary training submission: test multi-node launch before continuation.
set -eo pipefail
if [[ $# -lt 5 ]]; then
    cat >&2 <<'USAGE'
Usage: bash scripts/smoke_multinode.sh SITE ACCOUNT TRAIN_ROOT VALIDATION_ROOT RUN_DIR [--dry-run]
SITE: s3df, nersc, or polaris (alcf also accepted).
SPINE_PROD_PYTHON: submission interpreter (default: python3); requires Python 3.10+.
Use generic MPV/MPR v04 LArCV files on storage visible to both compute nodes.
RUN_DIR must be a fresh shared directory. This submits a job; it does not wait.
USAGE
    exit 2
fi
site=$1
account=$2
train_source=$3
validation_source=$4
run_dir=$5
shift 5
for option in "$@"; do
    if [[ "$option" != --dry-run ]]; then
        echo "Unsupported option: $option (only --dry-run is accepted)" >&2
        exit 2
    fi
done
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo_dir/configure.sh"
set -u
if [[ "$SPINE_PROD_BASEDIR" != "$repo_dir" ]]; then
    echo "FORCE_SPINE_PROD_BASEDIR points at another checkout; unset it and retry." >&2
    exit 2
fi
for input in "$train_source" "$validation_source"; do
    if [[ ! -r "$input" ]]; then
        echo "Cannot read generic input file: $input" >&2
        exit 2
    fi
done
case "$site" in
    s3df) resources=(--profile s3df_ampere_full --gpus 4) ;;
    nersc) resources=(--profile nersc_gpu_full --gpus-per-node 4 --qos debug) ;;
    polaris|alcf) resources=(--profile anl_polaris_debug_full --gpus-per-node 4) ;;
    *) echo "Unknown site: $site" >&2; exit 2 ;;
esac
# Retain NCCL initialization details in the attempt logs for rank/network diagnosis.
export NCCL_DEBUG=INFO
"${SPINE_PROD_PYTHON:-python3}" "$repo_dir/submit.py" \
    --config train/generic/uresnet/train_240718.yaml \
    --stage train --run-dir "$run_dir" \
    --account "$account" "${resources[@]}" \
    --nodes 2 --world-size 8 --time 00:30:00 \
    --source "$train_source" --val-source "$validation_source" \
    --minibatch-size 1 --num-workers 2 \
    --num-entries 64 --val-num-entries 16 --iterations 20 \
    --set base.epochs=null \
    --set train.save_epoch=null --set train.save_step=10 \
    --set validation.early_stopping=null \
    "$@"
printf '\nInspect %s/latest/{stdout,stderr}.log and %s/weights/ after completion.\n' "$run_dir" "$run_dir"
printf 'Expect world size 8, 20 training steps, validation, and saved checkpoints.\n'
