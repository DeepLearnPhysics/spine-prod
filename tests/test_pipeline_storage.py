"""Working cache routing and batch archive integration."""

from pathlib import Path

import pytest

from src.pipeline.storage import archive_plans, validate_cache_storage, working_stages


def test_relocation_preserves_references_and_avoids_cascade(tmp_path):
    root = str(tmp_path)
    stages = [
        dict(
            cache_repository=root + "/cache/train",
            source=[root + "/cache/train/manifest.json"],
            set=(f"cache.path={root}/cache/train",),
            other=True,
        ),
        dict(cache_dir=root + "/cache/new/train", similar=root + "/cache/train-extra"),
    ]
    mapped = working_stages(stages, root, root + "/cache/new")
    assert mapped[0]["cache_repository"] == root + "/cache/new/train"
    assert mapped[0]["source"] == [root + "/cache/new/train/manifest.json"]
    assert mapped[0]["set"] == [f"cache.path={root}/cache/new/train"]
    assert mapped[1]["similar"] == root + "/cache/train-extra"
    assert mapped[1]["cache_dir"] == root + "/cache/new/new/train"
    assert stages[0]["cache_repository"] == root + "/cache/train"
    assert working_stages(stages, None, None) is stages
    assert working_stages([{"source": "x"}], root, root + "/new") == [{"source": "x"}]


def test_relocation_errors(tmp_path):
    with pytest.raises(ValueError, match="workspace"):
        working_stages([], None, "cache")
    with pytest.raises(ValueError, match="under workspace"):
        working_stages([{"cache_dir": "/elsewhere"}], tmp_path, "cache")
    with pytest.raises(ValueError, match="collide"):
        working_stages(
            [
                {"cache_dir": str(tmp_path / "cache/a")},
                {"cache_dir": str(tmp_path / "a")},
            ],
            tmp_path,
            "cache",
        )


@pytest.mark.parametrize("field", ["cache_repository", "cache_dir"])
def test_nersc_cache_storage(field, tmp_path):
    validate_cache_storage({field: "/pscratch/sd/u/user/cache"}, "nersc")
    validate_cache_storage({}, "nersc")
    validate_cache_storage({field: "/sdf/cache"}, "s3df")
    validate_cache_storage({field: "/lus/cache"}, "anl")
    with pytest.raises(ValueError, match="Scratch"):
        validate_cache_storage({field: "/global/cfs/cache"}, "nersc")
    link = tmp_path / "link"
    link.symlink_to("/global/cfs/cache")
    with pytest.raises(ValueError, match="Scratch"):
        validate_cache_storage({field: str(link)}, "nersc")


def producer(tmp_path, profile="s3df_ampere"):
    return dict(
        name="cache",
        cache_repository=str(tmp_path / "cache/train"),
        cache_stage="encoder",
        run_dir=str(tmp_path / "producer"),
        profile=profile,
    )


@pytest.mark.parametrize(
    "profile,cpu",
    [
        ("s3df_ampere", "s3df_milano"),
        ("nersc_gpu", "nersc_cpu"),
        ("anl_polaris_capacity", "anl_polaris_cpu"),
    ],
)
def test_archive_plan_and_render(mock_submitter, tmp_path, profile, cpu, monkeypatch):
    stage = producer(tmp_path, profile)
    # Use the actual site profile names below; aliases are not implicit.
    plans = archive_plans(
        [stage], str(tmp_path), str(tmp_path / "archive"), mock_submitter
    )
    plan = plans["cache"]
    assert plan["profile"] == cpu
    metadata = Path(stage["run_dir"]) / "latest/job_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"cache_publication_id": "producer-identity"}')
    client = mock_submitter.batch.get_batch_client(
        mock_submitter.config_mgr.get_profile(cpu)
    )
    monkeypatch.setattr(mock_submitter.batch, "get_batch_client", lambda _: client)
    monkeypatch.setattr(client, "submit", lambda *a: "123")
    assert mock_submitter.submit_archive(
        **plan, dependency="afterok:42", dry_run=True
    ) == ["123"]
    scripts = list(Path(plan["run_dir"]).glob("attempts/*/submit.*"))
    assert len(scripts) == 1
    content = scripts[0].read_text()
    assert "producer-identity" in content
    assert "cache_archive.py archive" in content
    assert "afterok:42" in content
    metadata.write_text("{}")
    with pytest.raises(ValueError, match="publication identity"):
        mock_submitter.submit_archive(**plan, dry_run=True)


def test_archive_plan_errors(mock_submitter, tmp_path):
    stage = producer(tmp_path)
    assert archive_plans([], None, None, mock_submitter) == {}
    with pytest.raises(ValueError, match="workspace"):
        archive_plans([], None, "archive", mock_submitter)
    with pytest.raises(ValueError, match="cache-producing"):
        archive_plans([{"name": "none"}], str(tmp_path), "archive", mock_submitter)
    with pytest.raises(ValueError, match="separate"):
        archive_plans([stage], str(tmp_path), str(tmp_path / "cache"), mock_submitter)
    with pytest.raises(ValueError, match="Reserved"):
        archive_plans(
            [stage, {"name": "cache__archive"}],
            str(tmp_path),
            str(tmp_path / "archive"),
            mock_submitter,
        )
    other = dict(
        stage, name="other", cache_repository=str(tmp_path / "elsewhere/train")
    )
    with pytest.raises(ValueError, match="collide"):
        archive_plans(
            [stage, other], str(tmp_path), str(tmp_path / "archive"), mock_submitter
        )


def test_normal_pipeline_gates_consumer_on_archive(
    mock_submitter, tmp_path, monkeypatch
):
    from unittest.mock import Mock

    from tests.pipeline.test_definition import write_pipeline

    first = dict(producer(tmp_path), config="producer.yaml", source="input.root")
    second = dict(
        name="train", profile="s3df_ampere", config="train.yaml", depends_on=["cache"]
    )
    path = write_pipeline(
        tmp_path, dict(workspace=str(tmp_path), stages=[first, second])
    )
    submit = Mock(side_effect=[["producer-job"], ["consumer-job"]])
    archive = Mock(return_value=["archive-job"])
    monkeypatch.setattr(mock_submitter, "submit_job", submit)
    monkeypatch.setattr(mock_submitter, "submit_archive", archive)
    result = mock_submitter.submit_pipeline(
        str(path), cache_archive_dir=str(tmp_path / "durable")
    )
    assert archive.call_args.kwargs["dependency"] == "afterok:producer-job"
    assert submit.call_args_list[1].kwargs["dependency"] == "afterok:archive-job"
    assert result == {"cache": ["archive-job"], "train": ["consumer-job"]}


def test_archive_submission_dry_run(mock_submitter, tmp_path):
    stage = producer(tmp_path)
    plan = archive_plans(
        [stage], str(tmp_path), str(tmp_path / "archive"), mock_submitter
    )["cache"]
    metadata = Path(stage["run_dir"]) / "latest/job_metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"cache_publication_id": "dry-producer"}')
    assert mock_submitter.submit_archive(**dict(plan, account=None), dry_run=True) == []


def test_multiple_archive_producers_share_unchanged_objects(mock_submitter, tmp_path):
    stage = producer(tmp_path)
    plans = archive_plans(
        [stage, dict(stage, name="extension", cache_stage="next")],
        str(tmp_path),
        str(tmp_path / "archive"),
        mock_submitter,
    )
    assert plans["cache"]["destination"] == plans["extension"]["destination"]
