"""Fixed-rank multi-node launch contracts for Slurm and PBS."""

import os
import runpy
import subprocess
from unittest.mock import Mock

import pytest
import yaml

from src.distributed import launch_command, main
from src.sites import map_stage, validate_allocation
from src.submission.arguments import SpineCLI


@pytest.mark.parametrize(
    "scheduler,rank_key,job_key",
    [
        ("slurm", "SLURM_PROCID", "SLURM_JOB_ID"),
        ("pbs", "PMI_RANK", "PBS_JOBID"),
    ],
)
def test_agents_share_rendezvous_and_have_distinct_ranks(scheduler, rank_key, job_key):
    env = {rank_key: "0", job_key: "123.server", "MASTER_ADDR": "node0"}
    first = launch_command(2, 4, scheduler, ["spine", "--world-size", "8"], env)
    env[rank_key] = "1"
    second = launch_command(2, 4, scheduler, ["spine", "--world-size", "8"], env)
    assert first[-4:] == ["--no-python", "spine", "--world-size", "8"]
    assert "--max-restarts=0" in first
    assert "--node-rank=0" in first and "--node-rank=1" in second
    assert [x for x in first if x.startswith("--master-")] == [
        x for x in second if x.startswith("--master-")
    ]
    assert "--nnodes=2" in first and "--nproc-per-node=4" in first


@pytest.mark.parametrize(
    "nodes,gpus,changes,command,error",
    [
        (1, 4, {}, ["spine"], ValueError),
        (2, 0, {}, ["spine"], ValueError),
        (2, 4, {}, [], ValueError),
        (2, 4, {"SLURM_PROCID": "2"}, ["spine"], ValueError),
        (2, 4, {"MASTER_ADDR": ""}, ["spine"], ValueError),
        (2, 4, {"MASTER_PORT": "0"}, ["spine"], ValueError),
    ],
)
def test_invalid_launch(nodes, gpus, changes, command, error):
    env = dict(SLURM_PROCID="0", SLURM_JOB_ID="123", MASTER_ADDR="node0", **{})
    env.update(changes)
    with pytest.raises(error):
        launch_command(nodes, gpus, "slurm", command, env)


@pytest.mark.parametrize("separator", [[], ["--"]])
def test_main_executes_container_python(monkeypatch, separator):
    for key, value in {
        "SLURM_PROCID": "1",
        "SLURM_JOB_ID": "123",
        "MASTER_ADDR": "node0",
        "MASTER_PORT": "23456",
        "CUDA_VISIBLE_DEVICES": "2,3",
    }.items():
        monkeypatch.setenv(key, value)
    execute = Mock()
    monkeypatch.setattr(os, "execv", execute)
    main(
        [
            "--nodes",
            "2",
            "--gpus-per-node",
            "2",
            "--scheduler",
            "slurm",
            *separator,
            "spine",
        ]
    )
    command = execute.call_args.args[1]
    assert "--master-port=23456" in command
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_script_entrypoint(monkeypatch, workspace_root):
    call = Mock()
    monkeypatch.setattr("src.distributed.main", call)
    runpy.run_path(
        str(workspace_root / "scripts/distributed_launch.py"), run_name="__main__"
    )
    call.assert_called_once_with()


@pytest.mark.parametrize(
    "profile,site",
    [
        ("s3df_ampere_full", "s3df"),
        ("nersc_gpu_full", "nersc"),
        ("anl_polaris_capacity_full", "anl"),
    ],
)
def test_world_size_and_mapping(workspace_root, profile, site):
    catalog = yaml.safe_load((workspace_root / "templates/profiles.yaml").read_text())
    stage = {"stage": "train", "profile": "s3df_ampere_full", "time": "120:00:00"}
    mapped = map_stage(catalog, stage, site, {"nodes": 2, "account": "project"})
    effective = dict(catalog["profiles"][mapped["profile"]], **mapped)
    assert effective["nodes"] == 2
    assert SpineCLI.align_world_size(effective, None) == 8
    validate_allocation(catalog, effective)
    assert mapped["continuation"]["runtime_budget"] == "120:00:00"
    other = map_stage(
        catalog,
        {"profile": "s3df_milano", "kind": "filter"},
        site,
        {"nodes": 2, "account": "project"},
    )
    assert other.get("nodes", 1) == 1


def test_reject_nontraining_multinode(workspace_root):
    catalog = yaml.safe_load((workspace_root / "templates/profiles.yaml").read_text())
    with pytest.raises(ValueError, match="only for training"):
        map_stage(catalog, {"profile": "s3df_ampere_full", "nodes": 2}, "s3df")


@pytest.mark.parametrize(
    "profile", [{"nodes": 0}, {"nodes": 2, "site": "s3df", "gpus": 0}]
)
def test_invalid_world_size(profile):
    with pytest.raises(ValueError):
        SpineCLI.align_world_size(profile, None)


@pytest.mark.parametrize(
    "site,template",
    [
        ("s3df", "job_template_s3df.sbatch"),
        ("nersc", "job_template_nersc.sbatch"),
        ("anl", "job_template_anl.pbs"),
    ],
)
def test_render_multinode(mock_submitter, tmp_path, site, template):
    script = mock_submitter.batch_client.load_template(template).render(
        site=site,
        nodes=2,
        gpus=4,
        gpus_per_node=4,
        cpus_per_task=32,
        account="project",
        basedir="/repo",
        config="train.yaml",
        output_args="",
        spine_cli_overrides="--world-size 8",
        log_dir=str(tmp_path),
        time="04:00:00",
        qos="regular",
        queue="debug",
        mem="0",
    )
    path = tmp_path / "job.sh"
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert "distributed_launch.py" in script
    assert "--nodes 2 --gpus-per-node 4" in script
    assert "--world-size 8" in script
    if site == "anl":
        assert "mpiexec -n 2 --ppn 1" in script
        assert "#PBS -l select=2:" in script
        assert 'head -n 1 "$PBS_NODEFILE"' in script
    else:
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --ntasks-per-node=1" in script
        assert "--kill-on-bad-exit=1" in script
        assert 'scontrol show hostnames "$SLURM_JOB_NODELIST"' in script


@pytest.mark.parametrize("site", ["s3df", "nersc", "anl"])
@pytest.mark.parametrize("failure", [False, True])
def test_generated_job_launches_each_node_and_propagates_failure(
    mock_submitter, tmp_path, site, failure
):
    import json
    import sys

    binaries = tmp_path / "bin"
    binaries.mkdir()
    # Simulate scheduler fan-out and container entry, executing the actual
    # generated shell. The runtime stub records precisely what each node sees.
    stub = (
        "#!"
        + sys.executable
        + "\n"
        + """
import json, os, pathlib, subprocess, sys
name=pathlib.Path(sys.argv[0]).name
args=sys.argv[1:]
if name == 'scontrol':
    print('node0\\nnode1')
elif name in ('srun','mpiexec'):
    while args[0] not in ('singularity','shifter','apptainer'):
        args.pop(0)
    for rank in range(2):
        env=dict(os.environ, SLURM_PROCID=str(rank), PMI_RANK=str(rank))
        code=subprocess.call(args,env=env)
        if code:
            sys.exit(code)
elif name in ('singularity','shifter','apptainer'):
    args=args[args.index('bash'):]
    sys.exit(subprocess.call(args))
elif args[0].endswith('distributed_launch.py'):
    with open(os.environ['CAPTURE'], 'a') as stream:
        stream.write(json.dumps({'args':args,'rank':os.environ['SLURM_PROCID'],'master':os.environ['MASTER_ADDR']})+'\\n')
    sys.exit(7 if os.environ['FAIL'] == '1' else 0)
elif args[0].endswith('check_spine_version.py'):
    sys.exit(0)
else:
    raise RuntimeError(args)
"""
    )
    for name in (
        "scontrol",
        "srun",
        "mpiexec",
        "singularity",
        "shifter",
        "apptainer",
        "python3",
    ):
        target = binaries / name
        target.write_text(stub)
        target.chmod(0o755)
    nodefile = tmp_path / "nodes"
    nodefile.write_text("node0\nnode1\n")
    capture = tmp_path / "capture.jsonl"
    env = dict(
        os.environ,
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        CAPTURE=str(capture),
        FAIL=str(int(failure)),
        SPINE_PROD_BASEDIR="/repo",
        SPINE_CONTAINER_PATH="/image",
        SPINE_CONTAINER_TAG="image",
        SLURM_JOB_ID="123",
        SLURM_JOB_NODELIST="node[0-1]",
        PBS_JOBID="123.server",
        PBS_NODEFILE=str(nodefile),
        PBS_O_WORKDIR=str(tmp_path),
    )
    template = "job_template_" + site + (".pbs" if site == "anl" else ".sbatch")
    script = mock_submitter.batch_client.load_template(template).render(
        site=site,
        nodes=2,
        gpus=4,
        gpus_per_node=4,
        cpus_per_task=32,
        account="project",
        basedir="/repo",
        config="train.yaml",
        output_args="",
        spine_cli_overrides="--world-size 8",
        log_dir=str(tmp_path),
        time="04:00:00",
        qos="regular",
        queue="debug",
        mem="0",
    )
    result = subprocess.run(
        ["bash"], input=script, text=True, env=env, capture_output=True
    )
    assert result.returncode == (7 if failure else 0), result.stdout + result.stderr
    records = [json.loads(line) for line in capture.read_text().splitlines()]
    assert [r["rank"] for r in records] == (["0"] if failure else ["0", "1"])
    assert all(r["master"] == "node0" for r in records)
    for record in records:
        args = record["args"]
        assert args[args.index("--nodes") + 1] == "2"
        assert args[args.index("--gpus-per-node") + 1] == "4"
        assert args[args.index("--world-size") + 1] == "8"


def test_multinode_direct_inference_rejected(mock_submitter):
    with pytest.raises(ValueError, match="ordinary training"):
        mock_submitter.submit_job(
            "infer/generic/full_chain_240718.yaml", nodes=2, dry_run=True
        )


@pytest.mark.parametrize("kind", ["filter", "report"])
def test_multinode_cpu_commands_rejected(mock_submitter, tmp_path, kind):
    with pytest.raises(ValueError, match="single-node allocation"):
        if kind == "filter":
            mock_submitter.submit_filter(
                config="filter/protodune-sp/space_points_260210.yaml",
                operation="scan",
                sources=["input.root"],
                cache_dir=str(tmp_path / "counts"),
                run_dir=str(tmp_path / "run"),
                nodes=2,
            )
        else:
            config = tmp_path / "report.yaml"
            config.write_text("metadata: {}\n")
            mock_submitter.submit_report(
                config=str(config),
                input_dir=str(tmp_path / "raw"),
                output_dir=str(tmp_path / "out"),
                run_dir=str(tmp_path / "run"),
                nodes=2,
            )


def test_pipeline_nodes_override_is_training_only(
    mock_submitter, tmp_path, monkeypatch
):
    from src.continuation import controller

    start = Mock(return_value={})
    monkeypatch.setattr(controller, "start", start)
    mock_submitter.submit_pipeline(
        str(
            mock_submitter.basedir
            / "pipelines/dune-hd-10kt-1x2x6/full_chain_261022.yaml"
        ),
        workspace=str(tmp_path / "hd"),
        site="nersc",
        overrides={"account": "project", "nodes": 2},
        continuation_dir="/pscratch/sd/u/user/hd-state",
        dry_run=True,
    )
    stages = start.call_args.args[1]
    assert all(
        s.get("nodes", 1) == (2 if s.get("stage") == "train" else 1) for s in stages
    )


def test_non_main_entrypoint(monkeypatch, workspace_root):
    call = Mock()
    monkeypatch.setattr("src.distributed.main", call)
    runpy.run_path(str(workspace_root / "scripts/distributed_launch.py"))
    call.assert_not_called()


@pytest.mark.parametrize(
    "profile,expected",
    [("nersc_gpu", "nersc_gpu_full"), ("nersc_gpu_80gb", "nersc_gpu_full_80gb")],
)
def test_shared_profile_promotion(workspace_root, profile, expected):
    catalog = yaml.safe_load((workspace_root / "templates/profiles.yaml").read_text())
    stage = {"stage": "train", "profile": profile, "qos": "shared"}
    mapped = map_stage(catalog, stage, "nersc", {"account": "project", "nodes": 2})
    assert mapped["profile"] == expected
    assert "qos" not in mapped
    explicit = map_stage(
        catalog, stage, "nersc", {"account": "project", "nodes": 2, "profile": profile}
    )
    assert explicit["profile"] == profile
    with pytest.raises(ValueError, match="Node count"):
        validate_allocation(catalog, dict(catalog["profiles"][profile], **explicit))
    explicit_qos = map_stage(
        catalog, stage, "nersc", {"account": "project", "nodes": 2, "qos": "shared"}
    )
    assert explicit_qos["qos"] == "shared"


@pytest.mark.parametrize(
    "profile", ["s3df_ampere_full", "nersc_gpu_full", "anl_polaris_capacity_full"]
)
def test_training_submission_renders_global_world(mock_submitter, tmp_path, profile):
    import json
    from unittest.mock import patch

    source = tmp_path / "train.root"
    source.touch()
    run_dir = tmp_path / "run"
    with (
        patch.object(
            mock_submitter.batch,
            "get_batch_client",
            return_value=mock_submitter.batch_client,
        ),
        patch.object(mock_submitter.batch_client, "submit", return_value=None),
    ):
        mock_submitter.submit_job(
            config="train/generic/uresnet/train_240718.yaml",
            files=[str(source)],
            validation_files=[str(source)],
            stage="train",
            profile=profile,
            nodes=2,
            account="project",
            run_dir=str(run_dir),
            dry_run=True,
        )
    metadata = json.loads((run_dir / "latest" / "job_metadata.json").read_text())
    assert metadata["world_size"] == 8
    script = next(run_dir.glob("attempts/*/submit.*")).read_text()
    assert "--world-size 8" in script
    assert "distributed_launch.py" in script
