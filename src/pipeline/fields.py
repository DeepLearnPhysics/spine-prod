"""Accepted pipeline fields and variable-name syntax."""

import re

GLOBAL_FIELDS = frozenset(
    {
        "profile",
        "larcv_path",
        "larcv_basedir",
        "flashmatch_path",
        "spine_path",
        "flashmatch",
        "cvmfs",
        "expandable_segments",
        "world_size",
        "batch_size",
        "minibatch_size",
        "num_workers",
        "epochs",
        "iterations",
        "partition",
        "qos",
        "queue",
        "constraint",
        "exclude",
        "gpus_per_node",
        "gpus",
        "cpus_per_task",
        "mem_per_cpu",
        "mem_per_node",
        "nodes",
        "time",
        "account",
        "bind_paths",
    }
)

STAGE_FIELDS = GLOBAL_FIELDS | frozenset(
    {
        "name",
        "kind",
        "config",
        "files",
        "source",
        "source_list",
        "val_source",
        "val_source_list",
        "sources",
        "validation_sources",
        "module_weight",
        "warm_start",
        "weight_path",
        "export_weights",
        "input_dir",
        "output_dir",
        "checkpoint",
        "dataset",
        "dataset_selection",
        "job_name",
        "output",
        "output_suffix",
        "output_source_list",
        "in_place",
        "cache_repository",
        "cache_stage",
        "no_writer",
        "ntasks",
        "files_per_task",
        "joint_file_mode",
        "joint_validate_trees",
        "depends_on",
        "cleanup",
        "apply_mods",
        "set",
        "stage",
        "run_dir",
        "resume",
        "resume_from",
        "continuation",
        "validation_name",
        "rerun_validation",
        "tensorboard",
        "num_files",
        "val_num_files",
        "num_entries",
        "val_num_entries",
        "entry_fraction_range",
        "val_entry_fraction_range",
        "entry_filter",
        "val_entry_filter",
        "operation",
        "cache_dir",
        "workers",
        "force",
    }
)

EXCLUSIVE_GROUPS = (
    frozenset({"partition", "qos", "queue"}),
    frozenset({"gpus", "gpus_per_node"}),
    frozenset({"mem_per_cpu", "mem_per_node"}),
    frozenset({"batch_size", "minibatch_size"}),
    frozenset({"epochs", "iterations"}),
)

PROFILE_FIELDS = (
    "partition",
    "qos",
    "queue",
    "constraint",
    "exclude",
    "gpus_per_node",
    "gpus",
    "cpus_per_task",
    "mem_per_cpu",
    "mem_per_node",
    "nodes",
    "time",
    "account",
    "bind_paths",
)

VARIABLE_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\}"
)
VARIABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
STAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
