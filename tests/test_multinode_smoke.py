"""Exercise the user-facing smoke script through real dry-run submission."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("explicit_python", [False, True])
@pytest.mark.parametrize("site", ["s3df", "nersc", "polaris"])
def test_smoke_submission(workspace_root, tmp_path, site, explicit_python):
    binaries = Path(sys.executable).parent
    source = tmp_path / "generic.root"
    source.touch()
    run = tmp_path / "run"
    env = dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}")
    env.pop("FORCE_SPINE_PROD_BASEDIR", None)
    env.pop("SPINE_PROD_PYTHON", None)
    env.pop("SPINE_LOCAL_PATH", None)
    if explicit_python:
        checkout = tmp_path / "custom spine"
        (checkout / "bin").mkdir(parents=True)
        (checkout / "bin/run.py").write_text("# source checkout entry point\n")
        env["SPINE_LOCAL_PATH"] = str(checkout)
    if explicit_python:
        # The override must work even when the default python3 is unusable.
        shim_dir = tmp_path / "bin"
        shim_dir.mkdir()
        shim = shim_dir / "python3"
        shim.write_text("#!/bin/sh\nexit 99\n")
        shim.chmod(0o755)
        env["PATH"] = f"{shim_dir}:{env['PATH']}"
        env["SPINE_PROD_PYTHON"] = sys.executable
    result = subprocess.run(
        [
            "bash",
            str(workspace_root / "scripts/smoke_multinode.sh"),
            site,
            "project",
            str(source),
            str(source),
            str(run),
            "--dry-run",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((run / "latest/job_metadata.json").read_text())
    assert metadata["world_size"] == 8
    assert metadata["iterations"] == 20
    assert metadata["minibatch_size"] == 1
    assert metadata["val_num_entries"] == 16
    assert metadata["profile_config"]["nodes"] == 2
    script = next((run / "latest").glob("submit.*")).read_text()
    assert "distributed_launch.py" in script
    if explicit_python:
        assert str(checkout / "bin/run.py") in script
    assert "--set train.save_epoch=null" in script
    assert "--set train.save_step=10" in script
    if site == "nersc":
        assert "#SBATCH --qos=debug" in script
    elif site == "polaris":
        assert "--cpu-bind depth /usr/bin/env apptainer exec" in script
        assert "#PBS -q debug" in script
        assert "#PBS -l singularity_fakeroot=true" in script
