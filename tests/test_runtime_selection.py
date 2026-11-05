"""Container jobs must not capture SPINE installations from the host PATH."""

from unittest.mock import patch

import pytest

from src.runtime import RuntimeResolver


@pytest.mark.parametrize(
    "resolver",
    [
        "resolve_spine_command",
        "resolve_spine_report_command",
        "resolve_spine_filter_command",
        "resolve_spine_cache_command",
    ],
)
def test_container_resolution_ignores_host_path(monkeypatch, resolver):
    monkeypatch.delenv("SPINE_LOCAL_PATH", raising=False)
    with patch(
        "src.runtime.shutil.which", return_value="/home/student/.local/bin/spine"
    ) as which:
        assert getattr(RuntimeResolver, resolver)() == (None, None)
        which.assert_not_called()


def test_local_resolution_preserves_host_discovery(monkeypatch):
    monkeypatch.delenv("SPINE_LOCAL_PATH", raising=False)
    with patch(
        "src.runtime.shutil.which", return_value="/home/student/.local/bin/spine"
    ):
        assert RuntimeResolver.resolve_spine_command(allow_host_path=True) == (
            "/home/student/.local/bin/spine",
            None,
        )


@pytest.mark.parametrize("from_env", [False, True])
def test_container_resolution_preserves_explicit_override(
    monkeypatch, tmp_path, from_env
):
    binary = tmp_path / "bin" / "spine"
    binary.parent.mkdir()
    binary.touch()
    monkeypatch.delenv("SPINE_LOCAL_PATH", raising=False)
    if from_env:
        monkeypatch.setenv("SPINE_LOCAL_PATH", str(tmp_path))
    command, bind_root = RuntimeResolver.resolve_spine_command(
        None if from_env else str(tmp_path)
    )
    assert command == str(binary)
    assert bind_root == str(tmp_path)


def test_batch_script_uses_container_command(mock_submitter, monkeypatch, tmp_path):
    monkeypatch.delenv("SPINE_LOCAL_PATH", raising=False)
    source = tmp_path / "input.root"
    source.touch()
    with (
        patch(
            "src.runtime.shutil.which", return_value="/home/student/.local/bin/spine"
        ),
        patch.object(
            mock_submitter.batch,
            "get_batch_client",
            return_value=mock_submitter.batch_client,
        ),
        patch.object(mock_submitter.batch_client, "submit", return_value="12345"),
    ):
        mock_submitter.submit_job(
            config="config/infer/dune-hd-10kt-1x2x6/full_chain_260510.yaml",
            files=[str(source)],
            profile="s3df_milano",
        )
    scripts = list(mock_submitter.jobs_dir.glob("**/attempts/*/submit.sbatch"))
    assert len(scripts) == 1
    script = scripts[0].read_text()
    assert "/home/student/.local/bin/spine" not in script
    assert 'check_spine_version.py\\" spine ||' in script
    assert " exec spine " in script
