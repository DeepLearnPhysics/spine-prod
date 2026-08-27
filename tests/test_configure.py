"""Tests for the sourced shell environment configuration."""

import os
import shutil
import subprocess


def run_bash(script, env=None):
    """Run a clean Bash snippet and return its output lines."""
    result = subprocess.run(
        ["/bin/bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env or {"PATH": os.environ["PATH"]},
    )
    return result.stdout.splitlines()


def configure_fixture(tmp_path, workspace_root):
    """Create a minimal configurable spine-prod root."""
    configure = tmp_path / "configure.sh"
    shutil.copy2(workspace_root / "configure.sh", configure)
    (tmp_path / "DEFAULT_SPINE_VERSION").write_text("1.0.4\n", encoding="utf-8")
    return configure


def test_resourcing_refreshes_automatic_container_values(tmp_path, workspace_root):
    configure = configure_fixture(tmp_path, workspace_root)
    default_version = tmp_path / "DEFAULT_SPINE_VERSION"
    lines = run_bash(
        f"""
        source {configure} >/dev/null
        printf 'first|%s|%s|%s\n' "$SPINE_CONTAINER_VERSION" \
            "$SPINE_CONTAINER_TAG" "$SPINE_CONTAINER_PATH"
        printf '1.0.5\n' > {default_version}
        source {configure} >/dev/null
        printf 'second|%s|%s|%s\n' "$SPINE_CONTAINER_VERSION" \
            "$SPINE_CONTAINER_TAG" "$SPINE_CONTAINER_PATH"
        """
    )

    assert lines == [
        "first|1.0.4|docker:ghcr.io/deeplearnphysics/spine:1.0.4|"
        "/sdf/data/neutrino/images/spine_v1-0-4.sif",
        "second|1.0.5|docker:ghcr.io/deeplearnphysics/spine:1.0.5|"
        "/sdf/data/neutrino/images/spine_v1-0-5.sif",
    ]


def test_resourcing_preserves_values_modified_after_configuration(
    tmp_path, workspace_root
):
    configure = configure_fixture(tmp_path, workspace_root)
    default_version = tmp_path / "DEFAULT_SPINE_VERSION"
    lines = run_bash(
        f"""
        source {configure} >/dev/null
        export SPINE_CONTAINER_VERSION=development
        export SPINE_CONTAINER_TAG=docker:example/spine:development
        export SPINE_CONTAINER_PATH=/images/development.sif
        printf '1.0.5\n' > {default_version}
        source {configure} >/dev/null
        printf '%s|%s|%s|%s|%s|%s\n' \
            "$SPINE_CONTAINER_VERSION" "$SPINE_CONTAINER_TAG" \
            "$SPINE_CONTAINER_PATH" "$SPINE_CONTAINER_VERSION_AUTO" \
            "$SPINE_CONTAINER_TAG_AUTO" "$SPINE_CONTAINER_PATH_AUTO"
        """
    )

    assert lines == [
        "development|docker:example/spine:development|/images/development.sif|0|0|0"
    ]


def test_initial_explicit_container_values_are_preserved(tmp_path, workspace_root):
    configure = configure_fixture(tmp_path, workspace_root)
    env = {
        "PATH": os.environ["PATH"],
        "SPINE_CONTAINER_VERSION": "custom",
        "SPINE_CONTAINER_TAG": "docker:example/spine:custom",
        "SPINE_CONTAINER_PATH": "/images/custom.sif",
    }
    lines = run_bash(
        f"""
        source {configure} >/dev/null
        printf '%s|%s|%s\n' "$SPINE_CONTAINER_VERSION" \
            "$SPINE_CONTAINER_TAG" "$SPINE_CONTAINER_PATH"
        """,
        env=env,
    )

    assert lines == ["custom|docker:example/spine:custom|/images/custom.sif"]
