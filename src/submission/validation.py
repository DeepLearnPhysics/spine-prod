"""Validation of batch options before any filesystem or scheduler work."""

from pathlib import Path


def validate_request(
    file_handler,
    *,
    files,
    validation_files,
    named_sources,
    validation_named_sources,
    warm_start_path,
    warm_start_modules,
    weight_path,
    export_weights,
    output,
    output_suffix,
    output_source_list,
    in_place,
    cache_repository,
    cache_stage,
    ntasks,
    files_per_task,
    joint_file_mode,
    joint_validate_trees,
    num_files,
    val_num_files,
    num_entries,
    val_num_entries,
    entry_fraction_range,
    val_entry_fraction_range,
    val_entry_filter,
    stage,
    run_dir,
    resume,
    resume_from,
    validation_name,
    rerun_validation,
):
    """Reject incompatible job options before resolving inputs or writing files.

    Parameters
    ----------
    file_handler : FileHandler
        Source handler used to validate file-count limits.

    Notes
    -----
    Keyword-only parameters mirror ``BatchRunner.submit_job``. Validation is
    independent of configuration resolution and does not create run state.

    Raises
    ------
    ValueError
        Options conflict or are not meaningful for the requested stage.
    TypeError
        A boolean control has an invalid type.
    """
    if joint_file_mode not in ("broadcast", "paired", "balanced"):
        raise ValueError(
            "joint_file_mode must be either 'broadcast', 'paired' or 'balanced'"
        )

    if not isinstance(joint_validate_trees, bool):
        raise TypeError("joint_validate_trees must be a boolean")
    if joint_validate_trees and joint_file_mode != "balanced":
        raise ValueError("joint_validate_trees requires balanced joint sharding")

    if in_place and (output is not None or output_suffix is not None):
        raise ValueError(
            "--in-place cannot be combined with --output or --output-suffix"
        )
    if bool(cache_repository) != bool(cache_stage):
        raise ValueError("cache_repository and cache_stage must be provided together")
    if cache_repository and stage != "inference":
        raise ValueError("Cache publication is valid only for inference jobs")
    if output_source_list and (
        in_place
        or not files
        or named_sources
        or not output
        or not output_suffix
        or Path(output).suffix.lower() in {".h5", ".hdf5"}
    ):
        raise ValueError(
            "output_source_list requires an explicit output directory and suffix"
        )

    if stage not in ("inference", "train", "validation"):
        raise ValueError("stage must be one of: inference, train, validation")
    if stage != "inference" and not run_dir:
        raise ValueError(f"--run-dir is required for the {stage} stage")
    if stage != "train" and (resume or resume_from):
        raise ValueError("--resume and --resume-from are valid only for training")
    if weight_path and (resume or resume_from):
        raise ValueError("--weight-path cannot be combined with training resume")
    if bool(warm_start_path) != bool(warm_start_modules):
        raise ValueError(
            "warm_start_path and warm_start_modules must be provided together"
        )
    if warm_start_path and stage != "train":
        raise ValueError("Pipeline warm start is valid only for training jobs")
    if warm_start_path and weight_path:
        raise ValueError("Pipeline warm start cannot be combined with weight_path")
    if stage != "validation" and (validation_name or rerun_validation):
        raise ValueError(
            "--validation-name and --rerun-validation are valid only for validation"
        )
    if validation_files and stage != "train":
        raise ValueError(
            "--val-source and --val-source-list are valid only for training"
        )
    if validation_named_sources and stage != "train":
        raise ValueError("Named validation sources are valid only for training")
    if val_entry_fraction_range is not None and stage != "train":
        raise ValueError("--val-entry-fraction-range is valid only for training")
    if val_entry_filter is not None and stage != "train":
        raise ValueError("--val-entry-filter is valid only for training")
    if val_num_entries is not None and stage != "train":
        raise ValueError("--val-num-entries is valid only for training")
    if val_num_files is not None and stage != "train":
        raise ValueError("--val-num-files is valid only for training")
    # Validate limits independently of whether source resolution is needed.
    file_handler.limit_files([], num_files)
    file_handler.limit_files([], val_num_files)
    if num_files is not None and not (files or named_sources):
        raise ValueError("--num-files requires --source/--source-list or named sources")
    if val_num_files is not None and not (validation_files or validation_named_sources):
        raise ValueError(
            "--val-num-files requires --val-source/--val-source-list or "
            "named validation sources"
        )
    if num_entries is not None and entry_fraction_range is not None:
        raise ValueError("--num-entries cannot be combined with --entry-fraction-range")
    if val_num_entries is not None and val_entry_fraction_range is not None:
        raise ValueError(
            "--val-num-entries cannot be combined with " "--val-entry-fraction-range"
        )
    if export_weights:
        if stage != "inference":
            raise ValueError("--export-weights requires stage=inference")
        if (
            files
            or named_sources
            or validation_files
            or validation_named_sources
            or entry_fraction_range is not None
            or val_entry_fraction_range is not None
            or num_entries is not None
            or val_num_entries is not None
            or num_files is not None
            or val_num_files is not None
        ):
            raise ValueError(
                "--export-weights cannot be combined with dataset selections"
            )
        if output is not None or output_suffix is not None:
            raise ValueError(
                "--export-weights cannot be combined with writer output options"
            )
    if files and named_sources:
        raise ValueError("Flat and named training sources cannot be combined")
    if validation_files and validation_named_sources:
        raise ValueError("Flat and named validation sources cannot be combined")
    if stage != "inference" and (ntasks is not None or files_per_task is not None):
        raise ValueError("--ntasks and --files-per-task are valid only for inference")
    if in_place and stage != "inference":
        raise ValueError("--in-place is valid only for inference")
