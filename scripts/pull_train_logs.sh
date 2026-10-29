#!/usr/bin/env bash
# Pull lightweight training artifacts; rerun to refresh growing logs.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/pull_train_logs.sh [--dry-run] SOURCE [DESTINATION]

Copy the contents of a training folder into ./runs/train/ by default.
SOURCE may be an SSH host:path or a local directory. Both ends need rsync.

Examples (run locally):
  scripts/pull_train_logs.sh --dry-run user@cluster:/path/to/runs/train/
  scripts/pull_train_logs.sh user@cluster:/path/to/runs/train/
  scripts/pull_train_logs.sh user@cluster:/path/to/train/ ./runs/my-training/

Includes logs, CSV/TSV, JSON/JSONL, YAML, text, scheduler scripts, plots,
and TensorBoard events. Excludes weights, checkpoints, caches, and all
other file types (including ROOT/HDF5/NumPy data). Nothing is deleted.
Relative symlinks within the source are preserved; external links are skipped.
Use your SSH config for host aliases, identity files, or jump hosts.
EOF
}

options=(--human-readable)
dry_run=false
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if [[ ${1:-} == --dry-run ]]; then
    options+=(--dry-run)
    dry_run=true
    shift
fi
if [[ $# -lt 1 || $# -gt 2 || $1 == -* ]]; then
    usage >&2
    exit 2
fi

source_dir=${1%/}/
destination=${2:-./runs/train/}
if [[ $destination == *:* ]]; then
    echo "DESTINATION must be a local directory" >&2
    exit 2
fi
if [[ $dry_run == false ]]; then
    mkdir -p -- "$destination"
elif [[ ! -d $(dirname -- "$destination") ]]; then
    echo "For a dry run, create the destination's parent directory first." >&2
    exit 2
fi

# Directory exclusions precede traversal and file includes. Do not follow
# symlinks: doing so could download a cache or checkpoint outside this tree.
exec rsync -rltv --safe-links --prune-empty-dirs "${options[@]}" \
    --exclude='weights/' \
    --exclude='checkpoints/' \
    --exclude='checkpoint/' \
    --exclude='cache/' \
    --exclude='caches/' \
    --exclude='.cache/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='*.ckpt*' \
    --exclude='*.pt' \
    --exclude='*.pth*' \
    --exclude='*.safetensors*' \
    --include='*/' \
    --include='latest' \
    --include='*.log' --include='*.log.*' \
    --include='*.csv' --include='*.tsv' \
    --include='*.err' --include='*.out' \
    --include='*.json' --include='*.jsonl' \
    --include='*.yaml' --include='*.yml' --include='*.txt' \
    --include='*.sbatch' --include='*.pbs' \
    --include='*.png' --include='*.pdf' --include='*.svg' \
    --include='*.jpg' --include='*.jpeg' \
    --include='events.out.tfevents.*' \
    --exclude='*' \
    -- "$source_dir" "$destination"
