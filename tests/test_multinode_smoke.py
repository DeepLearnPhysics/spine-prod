"""Exercise the user-facing smoke script through real dry-run submission."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("site", ["s3df", "nersc", "polaris"])
def test_smoke_submission(workspace_root, tmp_path, site):
    binaries = Path(sys.executable).parent
    source = tmp_path / "generic.root"
    source.touch()
    run = tmp_path / "run"
    env = dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}")
    env.pop("FORCE_SPINE_PROD_BASEDIR", None)
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
    assert "--set train.save_epoch=null" in script
    assert "--set train.save_step=10" in script
    if site == "nersc":
        assert "#SBATCH --qos=debug" in script
    elif site == "polaris":
        assert "#PBS -q debug" in script
        assert "#PBS -l singularity_fakeroot=true" in script
