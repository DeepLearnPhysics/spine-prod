"""Execute cache registration through the templates' actual nested shells."""

import os
import subprocess
from pathlib import Path

import pytest
from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("site", ["nersc", "s3df", "anl"])
@pytest.mark.parametrize("begin_status", [0, 23])
def test_cache_begin_id_and_fail_fast(tmp_path, site, begin_status):
    suffix = "pbs" if site == "anl" else "sbatch"
    template = (ROOT / "templates" / f"job_template_{site}.{suffix}").read_text()
    # Use the real assignment and container launch construction, replacing only
    # the external scheduler/container programs with argument-forwarding mocks.
    begin = next(
        line for line in template.splitlines() if "{{ cache_begin_cmd }} begin" in line
    )
    inner = {
        "nersc": "INNER_CMD",
        "s3df": "SINGULARITY_INNER_CMD",
        "anl": "APPTAINER_INNER_CMD",
    }[site]
    launch = [line for line in template.splitlines() if line.startswith('RUN_CMD="')]
    if site == "s3df":
        setup = next(
            line
            for line in template.splitlines()
            if line.startswith('SINGULARITY_CMD="')
        )
        launch = [setup, 'RUN_CMD="$SINGULARITY_CMD"']
    script = f'{inner}=""\n' + begin + "\n"
    script += f'{inner}="${inner} touch {tmp_path}/inference-ran"\n'
    script += "\n".join(launch) + '\neval "$RUN_CMD"\n'
    rendered = Template(script).render(
        cache_publication_id_shell="publication-123",
        cache_begin_cmd=str(tmp_path / "cache-command"),
        cache_repository_shell="/tmp/cache",
        cache_stage_shell="deghosting",
        nodes=1,
        cpus_per_task=1,
    )
    command = tmp_path / "cache-command"
    command.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "'
        + str(tmp_path / "args")
        + '"\nexit '
        + str(begin_status)
        + "\n"
    )
    command.chmod(0o755)
    mocks = """
forward() { while [[ "$1" != bash ]]; do shift; done; "$@"; }
srun() { forward "$@"; }
singularity() { forward "$@"; }
apptainer() { forward "$@"; }
export -f forward srun singularity apptainer
SPINE_CONTAINER_TAG=dummy
SPINE_CONTAINER_PATH=dummy
"""
    env = dict(os.environ)
    env.pop("SPINE_CACHE_PUBLICATION_ID", None)
    result = subprocess.run(
        ["bash", "-c", mocks + rendered], env=env, capture_output=True, text=True
    )
    assert (tmp_path / "args").read_text().splitlines() == [
        "begin",
        "/tmp/cache",
        "deghosting",
        "--publication-id",
        "publication-123",
    ], result.stderr
    assert result.returncode == (1 if begin_status else 0), result.stderr
    assert (tmp_path / "inference-ran").exists() == (begin_status == 0)
