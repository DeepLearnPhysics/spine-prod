"""Focused tests for configuration and profile management."""

import builtins
import os
from pathlib import Path

import pytest
import yaml

from src.config_manager import ConfigManager


@pytest.fixture
def manager(tmp_path):
    templates = tmp_path / "templates"
    templates.mkdir()
    profiles = {
        "defaults": {"default_profile": "default", "time": "01:00:00"},
        "profiles": {
            "default": {"partition": "cpu"},
            "gpu": {"partition": "gpu"},
        },
        "detectors": {"icarus": {"default_profile": "gpu"}},
    }
    (templates / "profiles.yaml").write_text(yaml.safe_dump(profiles))
    return ConfigManager(tmp_path)


def test_init_requires_profiles_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Profiles not found"):
        ConfigManager(tmp_path)


def test_get_profile_auto_selects_detector_and_merges_defaults(manager):
    profile = manager.get_profile("auto", "icarus")

    assert profile == {
        "default_profile": "default",
        "time": "01:00:00",
        "partition": "gpu",
    }


def test_get_profile_auto_uses_global_default(manager):
    assert manager.get_profile("auto")["partition"] == "cpu"


def test_get_profile_rejects_unknown_name(manager):
    with pytest.raises(ValueError, match="Unknown profile: missing"):
        manager.get_profile("missing")


def test_detect_detector_returns_unknown_for_unrecognized_path(manager):
    assert manager.detect_detector("infer/icarus/base.yaml") == "icarus"
    assert manager.detect_detector("infer/other/base.yaml") == "unknown_detector"


def test_discover_modifiers_uses_versioned_directory_layout(manager, tmp_path):
    detector_dir = tmp_path / "config" / "infer" / "icarus"
    modifier_dir = detector_dir / "modifier" / "data"
    modifier_dir.mkdir(parents=True)
    common = modifier_dir / "mod_data_common.yaml"
    old = modifier_dir / "mod_data_240101.yaml"
    new = modifier_dir / "mod_data_250101.yaml"
    for path in (common, old, new):
        path.touch()

    assert manager.discover_modifiers(str(detector_dir)) == {"data": [old, new]}


def test_discover_modifiers_supports_legacy_file_names(manager, tmp_path):
    detector_dir = tmp_path / "legacy"
    detector_dir.mkdir()
    modifier = detector_dir / "legacy_data_mod.yaml"
    modifier.touch()

    assert manager.discover_modifiers(str(detector_dir)) == {"data": [modifier]}


def test_discover_modifiers_merges_legacy_yaml_and_cfg(manager, tmp_path):
    detector_dir = tmp_path / "legacy"
    detector_dir.mkdir()
    yaml_modifier = detector_dir / "legacy_data_mod.yaml"
    cfg_modifier = detector_dir / "legacy_data_mod.cfg"
    yaml_modifier.touch()
    cfg_modifier.touch()

    assert manager.discover_modifiers(str(detector_dir)) == {
        "data": [yaml_modifier, cfg_modifier]
    }


def test_discover_modifiers_accepts_config_file_and_ignores_empty_entries(
    manager, tmp_path
):
    detector_dir = tmp_path / "icarus"
    empty_modifier = detector_dir / "modifier" / "empty"
    empty_modifier.mkdir(parents=True)
    (detector_dir / "modifier" / "README.md").touch()
    config = detector_dir / "full_chain.yaml"
    config.touch()

    assert manager.discover_modifiers(str(config)) == {}


def test_resolve_modifier_version_covers_selection_rules(manager):
    versions = [Path("mod_data_240101.yaml"), Path("mod_data_250101.yaml")]

    assert manager.resolve_modifier_version("data", versions, None, None) == versions[1]
    assert (
        manager.resolve_modifier_version("data", versions, "240601", None)
        == versions[0]
    )
    assert (
        manager.resolve_modifier_version("data", versions, "230101", None)
        == versions[0]
    )
    assert (
        manager.resolve_modifier_version("data", versions, None, "240101")
        == versions[0]
    )


def test_resolve_modifier_version_reports_invalid_requests(manager):
    with pytest.raises(ValueError, match="No versions found"):
        manager.resolve_modifier_version("data", [], None, None)

    with pytest.raises(ValueError, match="version '999999' not found"):
        manager.resolve_modifier_version(
            "data", [Path("mod_data_250101.yaml")], None, "999999"
        )


def test_create_composite_config_resolves_named_and_custom_modifiers(manager, tmp_path):
    config_dir = tmp_path / "config" / "infer" / "icarus"
    modifier_dir = config_dir / "modifier" / "data"
    modifier_dir.mkdir(parents=True)
    base = config_dir / "full_chain_250101.yaml"
    named = modifier_dir / "mod_data_250101.yaml"
    custom = tmp_path / "mod_custom.yaml"
    for path in (base, named, custom):
        path.write_text("{}\n")
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    result = Path(
        manager.create_composite_config(
            "infer/icarus/full_chain_250101.yaml",
            ["data:250101", str(custom)],
            job_dir,
        )
    )

    content = result.read_text()
    assert result.name == "full_chain_250101_data_custom_composite.yaml"
    assert "infer/icarus/full_chain_250101.yaml" in content
    assert "infer/icarus/modifier/data/mod_data_250101.yaml" in content
    assert str(custom) in content


def test_create_composite_config_accepts_cwd_relative_external_config(
    manager, tmp_path, monkeypatch
):
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    (external_dir / "base.yaml").touch()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    monkeypatch.chdir(external_dir)

    result = Path(manager.create_composite_config("base.yaml", [], job_dir))

    assert "../external/base.yaml" in result.read_text()


def test_create_composite_config_accepts_cwd_path_inside_config_tree(
    manager, tmp_path, monkeypatch
):
    base = tmp_path / "config" / "infer" / "icarus" / "base.yaml"
    base.parent.mkdir(parents=True)
    base.touch()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    result = Path(
        manager.create_composite_config("config/infer/icarus/base.yaml", [], job_dir)
    )

    assert "infer/icarus/base.yaml" in result.read_text()


def test_create_composite_config_handles_cross_drive_relpath_failure(
    manager, tmp_path, monkeypatch
):
    external = tmp_path / "external.yaml"
    external.touch()
    custom = tmp_path / "custom.yaml"
    custom.touch()
    original_relpath = os.path.relpath

    def fail_custom_relpath(path, start=None):
        if Path(path) == custom:
            raise ValueError("different drive")
        return original_relpath(path, start)

    monkeypatch.setattr("src.config_manager.os.path.relpath", fail_custom_relpath)

    result = Path(
        manager.create_composite_config(str(external), [str(custom)], tmp_path)
    )

    assert str(custom) in result.read_text()


def test_create_composite_config_handles_initial_relpath_failure(
    manager, tmp_path, monkeypatch
):
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external = external_dir / "base.yaml"
    external.touch()
    original_relpath = os.path.relpath
    monkeypatch.chdir(external_dir)

    def fail_config_tree_relpath(path, start=None):
        if Path(start) == tmp_path / "config":
            raise ValueError("different drive")
        return original_relpath(path, start)

    monkeypatch.setattr("src.config_manager.os.path.relpath", fail_config_tree_relpath)

    result = Path(manager.create_composite_config("base.yaml", [], tmp_path))

    assert "external/base.yaml" in result.read_text()


def test_create_composite_config_rewrites_latest_includes(manager, tmp_path):
    existing = tmp_path / "config" / "infer" / "icarus" / "base.yaml"
    existing.parent.mkdir(parents=True)
    existing.touch()
    latest = tmp_path / "latest.yaml"
    absolute_missing = tmp_path / "absolute_missing.yaml"
    latest.write_text(
        "include:\n"
        "  - infer/icarus/base.yaml\n"
        "  - relative_missing.yaml\n"
        f"  - {absolute_missing}\n"
        "base: {}\n"
    )

    manager.create_composite_config(str(latest), [], tmp_path)

    rewritten = latest.read_text()
    assert "  - infer/icarus/base.yaml" in rewritten
    assert "relative_missing.yaml" in rewritten
    assert "absolute_missing.yaml" in rewritten
    assert "base: {}" in rewritten


def test_create_composite_config_keeps_include_when_rewrite_relpath_fails(
    manager, tmp_path, monkeypatch
):
    latest = tmp_path / "latest.yaml"
    latest.write_text("include:\n  - missing.yaml\n")
    original_relpath = os.path.relpath

    def fail_missing_relpath(path, start=None):
        if Path(path).name == "missing.yaml":
            raise ValueError("different drive")
        return original_relpath(path, start)

    monkeypatch.setattr("src.config_manager.os.path.relpath", fail_missing_relpath)

    manager.create_composite_config(str(latest), [], tmp_path)

    assert "  - missing.yaml" in latest.read_text()


def test_create_composite_config_warns_when_latest_cannot_be_read(
    manager, tmp_path, monkeypatch, capsys
):
    latest = tmp_path / "latest.yaml"
    latest.touch()
    original_open = builtins.open

    def fail_latest_read(path, mode="r", *args, **kwargs):
        if Path(path) == latest and mode == "r":
            raise OSError("read failed")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fail_latest_read)

    manager.create_composite_config(str(latest), [], tmp_path)

    assert "Could not rewrite includes" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("base", "modifiers", "message"),
    [
        ("missing.yaml", [], "Base config not found"),
        ("infer/icarus/full_chain_250101.yaml", ["missing"], "Unknown modifier"),
        (
            "infer/icarus/full_chain_250101.yaml",
            ["/missing/custom.yaml"],
            "Custom modifier file not found",
        ),
    ],
)
def test_create_composite_config_reports_invalid_inputs(
    manager, tmp_path, base, modifiers, message
):
    config = tmp_path / "config" / "infer" / "icarus" / "full_chain_250101.yaml"
    config.parent.mkdir(parents=True)
    config.touch()

    with pytest.raises((FileNotFoundError, ValueError), match=message):
        manager.create_composite_config(base, modifiers, tmp_path)


def test_create_latest_config_selects_latest_components(manager, tmp_path):
    detector_dir = tmp_path / "config" / "infer" / "icarus"
    base_dir = detector_dir / "base"
    io_dir = detector_dir / "io"
    base_dir.mkdir(parents=True)
    io_dir.mkdir()
    (base_dir / "base_common.yaml").touch()
    (base_dir / "base_240101.yaml").touch()
    latest_base = base_dir / "base_250101.yaml"
    latest_base.touch()
    (io_dir / "io_custom.yaml").touch()

    result = Path(manager.create_latest_config("icarus", tmp_path))

    assert result.name == "icarus_latest_250101_composite.yaml"
    content = result.read_text()
    assert "infer/icarus/base/base_250101.yaml" in content
    assert "infer/icarus/io/io_custom.yaml" in content
    assert "base_240101.yaml" not in content


def test_create_latest_config_reports_missing_or_empty_detector(manager, tmp_path):
    with pytest.raises(ValueError, match="Detector config directory not found"):
        manager.create_latest_config("missing", tmp_path)

    (tmp_path / "config" / "infer" / "empty").mkdir(parents=True)
    with pytest.raises(ValueError, match="No versioned components found"):
        manager.create_latest_config("empty", tmp_path)

    empty_component = tmp_path / "config" / "infer" / "empty_component" / "base"
    empty_component.mkdir(parents=True)
    with pytest.raises(ValueError, match="No versioned components found"):
        manager.create_latest_config("empty_component", tmp_path)


def test_create_latest_config_supports_only_unversioned_component(manager, tmp_path):
    base_dir = tmp_path / "config" / "infer" / "custom" / "base"
    base_dir.mkdir(parents=True)
    (base_dir / "base_custom.yaml").touch()

    result = Path(manager.create_latest_config("custom", tmp_path))

    assert result.name == "custom_latest_composite.yaml"
    assert "Latest version:" not in result.read_text()


def test_create_latest_config_uses_absolute_path_when_relpath_fails(
    manager, tmp_path, monkeypatch
):
    base = tmp_path / "config" / "infer" / "cross_drive" / "base" / "base_250101.yaml"
    base.parent.mkdir(parents=True)
    base.touch()
    monkeypatch.setattr(
        "src.config_manager.os.path.relpath",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("different drive")),
    )

    result = Path(manager.create_latest_config("cross_drive", tmp_path))

    assert f"  - {base}" in result.read_text()


def test_list_modifiers_resolves_detector_path_and_versions(manager, tmp_path):
    modifier_dir = tmp_path / "config" / "infer" / "icarus" / "modifier" / "data"
    modifier_dir.mkdir(parents=True)
    modifier = modifier_dir / "mod_data_240101.yaml"
    modifier.touch()

    result = manager.list_modifiers("infer/icarus/full_chain_250101.yaml")

    assert result["base_version"] == "250101"
    assert result["modifiers"]["data"]["selected"] == "240101"
    assert result["modifiers"]["data"]["available"] == ["240101"]


def test_list_modifiers_accepts_path_outside_infer_tree(manager, tmp_path):
    detector_dir = tmp_path / "legacy"
    detector_dir.mkdir()
    modifier = detector_dir / "legacy_data_mod.yaml"
    modifier.touch()

    result = manager.list_modifiers(str(detector_dir / "full_chain.yaml"))

    assert result["base_version"] is None
    assert result["modifiers"]["data"]["paths"] == [modifier]
