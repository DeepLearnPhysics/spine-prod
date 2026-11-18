"""Validate stage options and lifecycle constraints before submission."""

from typing import Any, List, Mapping

from ..submission.arguments import SpineCLI
from .fields import VARIABLE_NAME_PATTERN


class StageValidator:
    """Validate stage fields without resolving files or submitting jobs."""

    __slots__ = ()

    @staticmethod
    def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
        """Require a mapping and produce a context-rich type error."""
        if not isinstance(value, Mapping):
            raise TypeError(f"{context} must be a mapping")
        return value

    @staticmethod
    def _validate_dependencies(name: str, depends_on: Any, prior_names: set) -> None:
        """Require dependencies to name earlier stages in document order."""
        if not isinstance(depends_on, list) or not all(
            isinstance(dependency, str) for dependency in depends_on
        ):
            raise TypeError(
                f"Pipeline stage '{name}' depends_on must be a list of names"
            )
        unavailable = set(depends_on) - prior_names
        if unavailable:
            raise ValueError(
                f"Pipeline stage '{name}' depends on unknown or later stage(s): "
                + ", ".join(sorted(unavailable))
            )

    @classmethod
    def _validate_sources(cls, name: str, stage: Mapping[str, Any]) -> None:
        """Reject ambiguous scalar and composite source declarations."""
        source_keys = cls._present(stage, "files", "source", "source_list")
        val_keys = cls._present(stage, "val_source", "val_source_list")
        if len(source_keys) > 1:
            raise ValueError(
                f"Pipeline stage '{name}' must specify only one of: "
                "files, source, source_list"
            )
        if len(val_keys) > 1:
            raise ValueError(
                f"Pipeline stage '{name}' must specify only one of: "
                "val_source, val_source_list"
            )
        if stage.get("sources") and source_keys:
            raise ValueError(
                f"Pipeline stage '{name}' cannot combine sources with "
                "files/source/source_list"
            )
        if stage.get("validation_sources") and val_keys:
            raise ValueError(
                f"Pipeline stage '{name}' cannot combine validation_sources "
                "with val_source/val_source_list"
            )

    @staticmethod
    def _validate_structured_fields(name: str, stage: Mapping[str, Any]) -> None:
        """Validate fields translated into repeated SPINE CLI options."""
        for field in (
            "sources",
            "validation_sources",
            "module_weight",
            "warm_start",
        ):
            value = stage.get(field)
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"Pipeline stage '{name}' {field} must be a mapping")
        module_weights = stage.get("module_weight")
        if module_weights is not None:
            StageValidator._validate_module_weights(name, module_weights)
        warm_start = stage.get("warm_start")
        if warm_start is not None:
            StageValidator._validate_warm_start(name, warm_start)
        export_weights = stage.get("export_weights")
        if export_weights is not None:
            if not isinstance(export_weights, str):
                raise TypeError(
                    f"Pipeline stage '{name}' export_weights must be a string"
                )
            if not export_weights:
                raise ValueError(
                    f"Pipeline stage '{name}' export_weights must not be empty"
                )
        for field in (
            "weight_path",
            "input_dir",
            "output_dir",
            "checkpoint",
            "dataset",
            "entry_filter",
            "val_entry_filter",
            "cache_dir",
            "cache_repository",
            "cache_stage",
        ):
            value = stage.get(field)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(
                    f"Pipeline stage '{name}' {field} must be a non-empty string"
                )
        for field in ("entry_fraction_range", "val_entry_fraction_range"):
            value = stage.get(field)
            if value is not None:
                try:
                    SpineCLI.validate_fraction_range(f"Pipeline {field}", value)
                except (TypeError, ValueError) as err:
                    raise type(err)(f"Pipeline stage '{name}': {err}") from err
        for field in ("num_files", "val_num_files"):
            value = stage.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(
                    f"Pipeline stage '{name}' {field} must be a positive integer"
                )
        for field in ("num_entries", "val_num_entries"):
            value = stage.get(field)
            if value is not None:
                try:
                    SpineCLI.format_num_entries(**{field: value})
                except ValueError as err:
                    raise ValueError(f"Pipeline stage '{name}': {err}") from err
        if (
            stage.get("num_entries") is not None
            and stage.get("entry_fraction_range") is not None
        ):
            raise ValueError(
                f"Pipeline stage '{name}' cannot combine num_entries with "
                "entry_fraction_range"
            )
        if (
            stage.get("val_num_entries") is not None
            and stage.get("val_entry_fraction_range") is not None
        ):
            raise ValueError(
                f"Pipeline stage '{name}' cannot combine val_num_entries with "
                "val_entry_fraction_range"
            )
        in_place = stage.get("in_place")
        if in_place is not None and not isinstance(in_place, bool):
            raise TypeError(f"Pipeline stage '{name}' in_place must be a boolean")

        joint_file_mode = stage.get("joint_file_mode", "broadcast")
        if joint_file_mode not in ("broadcast", "paired", "balanced"):
            raise ValueError(
                f"Pipeline stage '{name}' joint_file_mode must be "
                "broadcast, paired or balanced"
            )

        joint_validate_trees = stage.get("joint_validate_trees", False)
        if not isinstance(joint_validate_trees, bool):
            raise TypeError(
                f"Pipeline stage '{name}' joint_validate_trees must be a boolean"
            )
        if joint_validate_trees and joint_file_mode != "balanced":
            raise ValueError(
                f"Pipeline stage '{name}' joint_validate_trees requires balanced joint sharding"
            )

        cache_fields = StageValidator._present(stage, "cache_repository", "cache_stage")
        if cache_fields and len(cache_fields) != 2:
            raise ValueError(
                f"Pipeline stage '{name}' must define cache_repository and "
                "cache_stage together"
            )

    @staticmethod
    def _validate_module_weights(name: str, module_weights: Mapping[Any, Any]) -> None:
        """Require scalar destination-module checkpoint assignments."""
        for module, path in module_weights.items():
            if not isinstance(module, str) or not VARIABLE_NAME_PATTERN.match(module):
                raise ValueError(
                    f"Pipeline stage '{name}' module_weight keys must be "
                    "valid identifiers"
                )
            if not isinstance(path, str) or not path:
                raise ValueError(
                    f"Pipeline stage '{name}' module_weight path for "
                    f"'{module}' must be a non-empty string"
                )

    @staticmethod
    def _validate_warm_start(name: str, modules: Mapping[Any, Any]) -> None:
        """Validate destination-to-source checkpoint namespace mappings."""
        if not modules:
            raise ValueError(f"Pipeline stage '{name}' warm_start must not be empty")
        for module, source in modules.items():
            if not isinstance(module, str) or not VARIABLE_NAME_PATTERN.match(module):
                raise ValueError(
                    f"Pipeline stage '{name}' warm_start keys must be valid "
                    "destination module identifiers"
                )
            if not isinstance(source, str) or not source:
                raise ValueError(
                    f"Pipeline stage '{name}' warm_start source for '{module}' "
                    "must be a non-empty namespace"
                )
            if any(
                not VARIABLE_NAME_PATTERN.match(component)
                for component in source.split(".")
            ):
                raise ValueError(
                    f"Pipeline stage '{name}' warm_start source '{source}' must "
                    "be a dot-separated module namespace"
                )

    @classmethod
    def _validate_lifecycle(cls, name: str, stage: Mapping[str, Any]) -> None:
        """Validate train, validation, and inference-only controls."""
        kind = stage.get("kind", "spine")
        if "continuation" in stage:
            if kind != "spine" or stage.get("stage") != "train":
                raise ValueError("continuation is supported only for training stages")
            if not isinstance(stage["continuation"], dict):
                raise ValueError("continuation must be a mapping")
        if kind == "filter":
            cls._validate_filter(name, stage)
            return
        if kind == "report":
            cls._validate_report(name, stage)
            return
        if kind != "spine":
            raise ValueError(f"Pipeline stage '{name}' has invalid kind: {kind}")

        lifecycle = stage.get("stage", "inference")
        if lifecycle not in ("inference", "train", "validation"):
            raise ValueError(
                f"Pipeline stage '{name}' has invalid lifecycle stage: {lifecycle}"
            )
        if lifecycle != "inference" and not stage.get("run_dir"):
            raise ValueError(
                f"Pipeline stage '{name}' requires run_dir for {lifecycle}"
            )
        if lifecycle != "train" and (stage.get("resume") or stage.get("resume_from")):
            raise ValueError(
                f"Pipeline stage '{name}' can resume only when stage=train"
            )
        if lifecycle != "train" and stage.get("warm_start") is not None:
            raise ValueError(f"Pipeline stage '{name}' warm_start requires stage=train")
        if lifecycle != "validation" and (
            stage.get("validation_name") or stage.get("rerun_validation")
        ):
            raise ValueError(
                f"Pipeline stage '{name}' validation lifecycle options require "
                "stage=validation"
            )

        validation_inputs = cls._present(
            stage, "val_source", "val_source_list", "validation_sources"
        )
        source_inputs = cls._present(stage, "files", "source", "source_list", "sources")
        if stage.get("num_files") is not None and not source_inputs:
            raise ValueError(
                f"Pipeline stage '{name}' num_files requires explicit inputs"
            )
        if stage.get("val_num_files") is not None and not validation_inputs:
            raise ValueError(
                f"Pipeline stage '{name}' val_num_files requires validation inputs"
            )
        if lifecycle != "train" and validation_inputs:
            raise ValueError(
                f"Pipeline stage '{name}' validation inputs require stage=train"
            )
        if lifecycle != "train" and stage.get("val_entry_fraction_range") is not None:
            raise ValueError(
                f"Pipeline stage '{name}' validation entry range requires stage=train"
            )
        if lifecycle != "train" and stage.get("val_entry_filter") is not None:
            raise ValueError(
                f"Pipeline stage '{name}' validation entry filter requires stage=train"
            )
        if lifecycle != "train" and stage.get("val_num_entries") is not None:
            raise ValueError(
                f"Pipeline stage '{name}' validation entry count requires stage=train"
            )
        if lifecycle != "inference" and (
            stage.get("ntasks") is not None or stage.get("files_per_task") is not None
        ):
            raise ValueError(
                f"Pipeline stage '{name}' task splitting requires stage=inference"
            )
        if stage.get("joint_file_mode", "broadcast") in ("paired", "balanced"):
            sources = stage.get("sources") or {}
            if lifecycle != "inference":
                raise ValueError(
                    f"Pipeline stage '{name}' paired joint files require "
                    "stage=inference"
                )
            if not (
                "primary" in sources
                and "secondary" in sources
                and "cache" not in sources
            ):
                raise ValueError(
                    f"Pipeline stage '{name}' paired joint files require "
                    "primary and secondary sources"
                )
        if stage.get("in_place"):
            if lifecycle != "inference":
                raise ValueError(
                    f"Pipeline stage '{name}' in_place requires stage=inference"
                )
            if (
                stage.get("output") is not None
                or stage.get("output_suffix") is not None
            ):
                raise ValueError(
                    f"Pipeline stage '{name}' in_place cannot be combined with "
                    "writer output options"
                )

        if stage.get("cache_repository"):
            if lifecycle != "inference":
                raise ValueError(
                    f"Pipeline stage '{name}' cache publication requires "
                    "stage=inference"
                )
            if not (
                stage.get("source") or stage.get("source_list") or stage.get("sources")
            ):
                raise ValueError(
                    f"Pipeline stage '{name}' cache publication requires inputs"
                )

        if stage.get("export_weights"):
            if lifecycle != "inference":
                raise ValueError(
                    f"Pipeline stage '{name}' export_weights requires "
                    "stage=inference"
                )
            source_inputs = cls._present(
                stage,
                "files",
                "source",
                "source_list",
                "sources",
                "val_source",
                "val_source_list",
                "validation_sources",
                "entry_fraction_range",
                "val_entry_fraction_range",
                "num_entries",
                "val_num_entries",
                "num_files",
                "val_num_files",
                "entry_filter",
                "val_entry_filter",
            )
            if source_inputs:
                raise ValueError(
                    f"Pipeline stage '{name}' export_weights cannot be combined "
                    "with data sources"
                )
            if (
                stage.get("output") is not None
                or stage.get("output_suffix") is not None
            ):
                raise ValueError(
                    f"Pipeline stage '{name}' export_weights cannot be combined "
                    "with writer output options"
                )

    @classmethod
    def _validate_filter(cls, name: str, stage: Mapping[str, Any]) -> None:
        """Require one standalone scan or manifest-build operation."""
        required = ("run_dir", "cache_dir", "operation")
        missing = [field for field in required if not stage.get(field)]
        if missing:
            raise ValueError(
                f"Pipeline filter stage '{name}' requires: " + ", ".join(missing)
            )

        operation = stage["operation"]
        if operation not in ("scan", "build"):
            raise ValueError(
                f"Pipeline filter stage '{name}' operation must be scan or build"
            )
        source_fields = cls._present(stage, "source", "source_list")
        if len(source_fields) != 1:
            raise ValueError(
                f"Pipeline filter stage '{name}' requires exactly one of: "
                "source, source_list"
            )

        workers = stage.get("workers")
        if workers is not None and (
            isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
        ):
            raise ValueError(
                f"Pipeline filter stage '{name}' workers must be a positive integer"
            )
        force = stage.get("force")
        if force is not None and not isinstance(force, bool):
            raise TypeError(f"Pipeline filter stage '{name}' force must be a boolean")

        if operation == "scan":
            outputs = cls._present(stage, "output", "output_source_list")
            if outputs:
                raise ValueError(
                    f"Pipeline filter scan stage '{name}' cannot define build outputs"
                )
        else:
            missing_outputs = [
                field
                for field in ("output", "output_source_list")
                if not stage.get(field)
            ]
            if missing_outputs:
                raise ValueError(
                    f"Pipeline filter build stage '{name}' requires: "
                    + ", ".join(missing_outputs)
                )
            if workers is not None or force:
                raise ValueError(
                    f"Pipeline filter build stage '{name}' cannot define workers or force"
                )

        forbidden = cls._present(
            stage,
            "files",
            "val_source",
            "val_source_list",
            "sources",
            "validation_sources",
            "entry_fraction_range",
            "val_entry_fraction_range",
            "num_entries",
            "val_num_entries",
            "num_files",
            "val_num_files",
            "entry_filter",
            "val_entry_filter",
            "module_weight",
            "weight_path",
            "export_weights",
            "input_dir",
            "output_dir",
            "checkpoint",
            "dataset",
            "dataset_selection",
            "in_place",
            "ntasks",
            "files_per_task",
            "set",
            "stage",
            "resume",
            "resume_from",
            "validation_name",
            "rerun_validation",
            "tensorboard",
        )
        if forbidden:
            raise ValueError(
                f"Pipeline filter stage '{name}' cannot use SPINE/report field(s): "
                + ", ".join(forbidden)
            )

    @classmethod
    def _validate_report(cls, name: str, stage: Mapping[str, Any]) -> None:
        """Require a standalone report contract without SPINE runtime fields."""
        required = ("run_dir", "input_dir", "output_dir")
        missing = [field for field in required if not stage.get(field)]
        if missing:
            raise ValueError(
                f"Pipeline report stage '{name}' requires: " + ", ".join(missing)
            )

        dataset_selection = stage.get("dataset_selection")
        if dataset_selection is not None and not isinstance(dataset_selection, Mapping):
            raise TypeError(
                f"Pipeline report stage '{name}' dataset_selection must be a mapping"
            )

        forbidden = cls._present(
            stage,
            "files",
            "source",
            "source_list",
            "val_source",
            "val_source_list",
            "sources",
            "validation_sources",
            "entry_fraction_range",
            "val_entry_fraction_range",
            "num_entries",
            "val_num_entries",
            "num_files",
            "val_num_files",
            "entry_filter",
            "val_entry_filter",
            "module_weight",
            "weight_path",
            "export_weights",
            "output",
            "output_suffix",
            "in_place",
            "ntasks",
            "files_per_task",
            "set",
            "stage",
            "resume",
            "resume_from",
            "validation_name",
            "rerun_validation",
            "tensorboard",
            "operation",
            "cache_dir",
            "workers",
            "force",
        )
        if forbidden:
            raise ValueError(
                f"Pipeline report stage '{name}' cannot use SPINE field(s): "
                + ", ".join(forbidden)
            )

    @staticmethod
    def _present(stage: Mapping[str, Any], *keys: str) -> List[str]:
        """Return configured keys while retaining their declared order."""
        return [key for key in keys if key in stage]
