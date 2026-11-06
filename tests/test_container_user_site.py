"""Execute container shell setup to verify Python user-site isolation."""

import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from jinja2 import Environment, FileSystemLoader

PROBE = shlex.join(
    [sys.executable, "-c", "import sys; assert sys.flags.no_user_site == 1"]
)


@pytest.mark.parametrize(
    "template,variable",
    [
        ("job_template_s3df.sbatch", "SINGULARITY_INNER_CMD"),
        ("job_template_nersc.sbatch", "INNER_CMD"),
        ("job_template_anl.pbs", "APPTAINER_INNER_CMD"),
    ],
)
@pytest.mark.parametrize("with_setup", [False, True])
def test_batch_disables_user_site_after_setup(tmp_path, template, variable, with_setup):
    # A sourced environment must not undo isolation for the checker/workload.
    (tmp_path / "configure.sh").write_text("unset PYTHONNOUSERSITE\n")
    templates = Path(__file__).resolve().parents[1] / "templates"
    rendered = (
        Environment(loader=FileSystemLoader(templates))
        .get_template(template)
        .render(
            larcv_path=str(tmp_path) if with_setup else None,
            flashmatch_path=str(tmp_path) if with_setup else None,
        )
    )
    start = rendered.index(f'{variable}=""')
    end = rendered.index(
        "# Validate the configuration API inside the selected runtime."
    )
    setup = rendered[start:end]
    # Evaluate the constructed inner command in a child shell, like the runtime.
    subprocess.run(
        ["bash", "-c", setup + f'\nbash -c "${variable} {PROBE}"'],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("runtime", ["singularity", "docker"])
def test_interactive_container_disables_user_site(
    mock_submitter, monkeypatch, tmp_path, runtime
):
    image = tmp_path / "image.sif"
    if runtime == "singularity":
        image.touch()
    monkeypatch.setenv("SPINE_CONTAINER_PATH", str(image))
    with patch(
        "src.runtime.shutil.which",
        side_effect=lambda name: f"/usr/bin/{name}" if name == runtime else None,
    ):
        command = mock_submitter.runtime.build_interactive_container_command(
            PROBE, False
        )
    inner = shlex.split(command)[-1]
    subprocess.run(["bash", "-c", inner], check=True, capture_output=True, text=True)
