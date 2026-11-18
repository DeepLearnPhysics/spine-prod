"""Scheduler queries for allocation deadlines, accounting and job discovery."""

import json
import re
import subprocess
import time

from .policy import seconds


def command(argv):
    """Run a bounded scheduler query and return its captured standard output."""
    return subprocess.run(
        argv, check=True, capture_output=True, text=True, timeout=30
    ).stdout


def discover_submission(scheduler, token):
    """Reconcile an uncertain submission by its unique, exact scheduler name.

    No match is not proof of rejection (accounting can lag): leave the intent
    pending, never blindly submit again. Multiple matches require intervention.
    """
    if not re.fullmatch(r"spc-[a-f0-9]{20}", token):
        raise ValueError("Invalid continuation submission token")
    matches = set()
    if scheduler == "slurm":
        output = command(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "--starttime=1970-01-01",
                "--name=" + token,
                "--format=JobIDRaw,JobName%100",
            ]
        )
        for line in output.splitlines():
            fields = line.split("|")
            if (
                len(fields) >= 2
                and fields[1].strip() == token
                and fields[0].strip().isdigit()
            ):
                matches.add(fields[0].strip())
    else:
        jobs = json.loads(command(["qstat", "-xf", "-F", "json"]))["Jobs"]
        matches.update(
            job_id for job_id, data in jobs.items() if data.get("Job_Name") == token
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple jobs match submission intent {token}: {sorted(matches)}; reconcile manually"
        )
    return next(iter(matches), None)


def allocation(scheduler, job_id):
    """Read the allocation's actual start time and assigned deadline.

    Parameters
    ----------
    scheduler : {"slurm", "pbs"}
        Scheduler owning the job.
    job_id : str
        Scheduler job identifier.

    Returns
    -------
    tuple of float
        Start and deadline as Unix timestamps. Requested walltime is never
        substituted when the scheduler cannot establish an assigned deadline.

    Raises
    ------
    ValueError
        The scheduler reports invalid allocation timestamps.
    """
    if scheduler == "slurm":
        data = json.loads(command(["scontrol", "show", "job", job_id, "--json"]))[
            "jobs"
        ][0]

        def scalar(value):
            """Unwrap Slurm timestamp fields returned as typed JSON values."""
            return value.get("number") if isinstance(value, dict) else value

        start, end = scalar(data["start_time"]), scalar(data["end_time"])
    else:
        data = json.loads(command(["qstat", "-f", "-F", "json", job_id]))["Jobs"][
            job_id
        ]
        start = data["stime"]
        if isinstance(start, str) and not start.isdigit():
            start = time.mktime(time.strptime(start, "%a %b %d %H:%M:%S %Y"))
        start = float(start)
        end = start + seconds(data["Resource_List"]["walltime"])
    start, end = float(start), float(end)
    if not 0 < start < end or start > time.time() + 60:
        raise ValueError(
            "Scheduler did not supply a valid assigned allocation deadline"
        )
    return start, end


def accounting(scheduler, job_id):
    """Read confirmed accounting for a terminal allocation.

    Parameters
    ----------
    scheduler : {"slurm", "pbs"}
        Scheduler owning the job.
    job_id : str
        Scheduler job identifier.

    Returns
    -------
    dict or None
        Status, allocated runtime in seconds and success flag. Array runtime
        is summed across tasks; success requires every task to complete with
        exit code zero. Live jobs or missing accounting return None.
    """
    if scheduler == "slurm":
        output = command(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                job_id,
                "--starttime=1970-01-01",
                "--array",
                "--format=JobID,State,ElapsedRaw,ExitCode",
            ]
        )
        # JobID preserves ArrayJobID_ArrayTaskID, unlike per-task JobIDRaw.
        # Expand arrays and inspect every allocation before declaring success.
        terminal = {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
        }
        results = []
        for line in output.splitlines():
            fields = line.split("|")
            if len(fields) < 4 or not re.fullmatch(
                re.escape(job_id) + r"(?:_\d+)?", fields[0]
            ):
                continue
            status = fields[1].split()[0].rstrip("+")
            if status not in terminal:
                return None
            results.append(
                {
                    "status": status,
                    "seconds": int(fields[2]),
                    "success": status == "COMPLETED" and fields[3] == "0:0",
                    "exit_code": fields[3],
                }
            )
        if results:
            # Preserve a failed task's status and exit code for diagnosis.
            result = next((item for item in results if not item["success"]), results[0])
            return dict(result, seconds=sum(item["seconds"] for item in results))

    else:
        data = json.loads(command(["qstat", "-xf", "-F", "json", job_id]))["Jobs"].get(
            job_id, {}
        )
        if data.get("job_state") == "F" and "Exit_status" in data:
            walltime = data.get("resources_used", {}).get("walltime")
            # A job that never started consumes zero allocation time.
            if walltime is None and "stime" in data:
                raise ValueError("PBS completed job is missing walltime accounting")
            elapsed = 0 if walltime in (None, "00:00:00") else seconds(walltime)
            return {
                "status": str(data["Exit_status"]),
                "seconds": elapsed,
                "success": int(data["Exit_status"]) == 0,
            }
    return None


def failure_reason(result):
    """Map terminal scheduler accounting to a logical interruption reason."""
    return {
        "PREEMPTED": "preemption",
        "NODE_FAIL": "node_failure",
        "BOOT_FAIL": "node_failure",
        "TIMEOUT": "allocation_expiry",
        "DEADLINE": "allocation_expiry",
        "CANCELLED": "user_cancelled",
        "OUT_OF_MEMORY": "out_of_memory",
    }.get(result["status"], "application_failure")
