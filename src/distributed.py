"""Launch one fixed-size torchrun agent per scheduler node inside its container."""

import argparse
import hashlib
import os
import sys


def launch_command(nodes, gpus_per_node, scheduler, command, environ):
    """Build the node-local agent command; torchrun supplies global worker ranks.

    The scheduler starts exactly one agent per node. All agents share a master
    address and job-derived port, while Slurm/PMI supplies their node rank.
    Elastic restarts are disabled: allocation continuation owns retries.
    """
    if nodes < 2 or gpus_per_node < 1:
        raise ValueError("Distributed launch requires multiple nodes and GPUs per node")
    if not command:
        raise ValueError("Missing training command")
    rank_key, job_key = (
        ("SLURM_PROCID", "SLURM_JOB_ID")
        if scheduler == "slurm"
        else ("PMI_RANK", "PBS_JOBID")
    )
    rank = int(environ[rank_key])
    if not 0 <= rank < nodes:
        raise ValueError("Scheduler node rank is outside the requested allocation")
    master = environ["MASTER_ADDR"]
    if not master:
        raise ValueError("Empty rendezvous address")
    job_digest = hashlib.sha256(environ[job_key].encode()).hexdigest()
    port = int(environ.get("MASTER_PORT", 20000 + int(job_digest[:8], 16) % 40000))
    if not 1 <= port <= 65535:
        raise ValueError("Invalid rendezvous port")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nodes}",
        f"--nproc-per-node={gpus_per_node}",
        f"--node-rank={rank}",
        f"--master-addr={master}",
        f"--master-port={port}",
        "--max-restarts=0",
        "--no-python",
        *command,
    ]


def main(argv=None):
    """Replace the node-local launcher with torchrun inside the selected runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--scheduler", choices=("slurm", "pbs"), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    launch = launch_command(
        args.nodes, args.gpus_per_node, args.scheduler, command, os.environ
    )
    # Preserve scheduler GPU isolation; Polaris may expose all local devices.
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES", ",".join(map(str, range(args.gpus_per_node)))
    )
    os.execv(launch[0], launch)
