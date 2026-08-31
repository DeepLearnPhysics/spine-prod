"""Loading, validation, and execution of multi-stage production pipelines."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from .component import SubmissionComponent

GLOBAL_FIELDS = frozenset(
    {
        "profile",
        "larcv_path",
        "larcv_basedir",
        "flashmatch_path",
        "spine_path",
        "flashmatch",
        "cvmfs",
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
        "config",
        "files",
        "source",
        "source_list",
        "val_source",
        "val_source_list",
        "sources",
        "validation_sources",
        "module_weight",
        "job_name",
        "output",
        "output_suffix",
        "no_writer",
        "ntasks",
        "files_per_task",
        "depends_on",
        "cleanup",
        "apply_mods",
        "set",
        "stage",
        "run_dir",
        "resume",
        "resume_from",
        "validation_name",
        "rerun_validation",
        "tensorboard",
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


@dataclass(frozen=True)
class PipelineDefinition:
    """A fully validated pipeline, ready for ordered submission.

    Attributes
    ----------
    stages : tuple of mappings
        Stages after defaults, stage values, and CLI overrides are resolved.
    """

    stages: Tuple[Dict[str, Any], ...]

    @classmethod
    def load(
        cls,
        pipeline_path: str,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "PipelineDefinition":
        """Load and validate a pipeline without contacting a scheduler.

        Configuration precedence is ``defaults < stage < CLI overrides``.
        Every stage is validated before the definition is returned, preventing
        a malformed later stage from leaving a partially submitted workflow.
        """
        with Path(pipeline_path).open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)

        if not isinstance(document, Mapping):
            raise TypeError("Pipeline YAML must contain a mapping")
        unknown = set(document) - {"defaults", "stages"}
        if unknown:
            raise ValueError("Unknown pipeline field(s): " + ", ".join(sorted(unknown)))

        defaults = cls._require_mapping(
            document.get("defaults", {}), "Pipeline defaults"
        )
        unknown = set(defaults) - GLOBAL_FIELDS
        if unknown:
            raise ValueError(
                "Pipeline defaults contain stage-specific or unknown field(s): "
                + ", ".join(sorted(unknown))
            )

        override_values = cls._require_mapping(overrides or {}, "Pipeline overrides")
        unknown = set(override_values) - GLOBAL_FIELDS
        if unknown:
            raise ValueError(
                "Unknown pipeline override field(s): " + ", ".join(sorted(unknown))
            )

        raw_stages = document.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("Pipeline must define a non-empty stages list")

        default_layer = cls._normalize_aliases(defaults, "Pipeline defaults")
        override_layer = cls._normalize_aliases(override_values, "Pipeline overrides")
        stages = []
        prior_names = set()
        for index, raw_stage in enumerate(raw_stages, start=1):
            stage = cls._resolve_stage(
                raw_stage,
                index,
                default_layer,
                override_layer,
                prior_names,
            )
            stages.append(stage)
            prior_names.add(stage["name"])

        return cls(tuple(stages))

    @staticmethod
    def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
        """Require a mapping and produce a context-rich type error."""
        if not isinstance(value, Mapping):
            raise TypeError(f"{context} must be a mapping")
        return value

    @staticmethod
    def _normalize_aliases(values: Mapping[str, Any], context: str) -> Dict[str, Any]:
        """Convert legacy aliases before values enter precedence resolution."""
        normalized = dict(values)
        if "larcv_basedir" in normalized:
            if "larcv_path" in normalized:
                raise ValueError(
                    f"{context} cannot specify both larcv_path and larcv_basedir"
                )
            normalized["larcv_path"] = normalized.pop("larcv_basedir")
        return normalized

    @staticmethod
    def _merge_layers(*layers: Mapping[str, Any]) -> Dict[str, Any]:
        """Merge precedence layers, replacing mutually exclusive settings.

        A higher-layer ``gpus`` setting, for example, removes a lower-layer
        ``gpus_per_node`` setting instead of passing both to the batch runner.
        """
        merged: Dict[str, Any] = {}
        for layer in layers:
            for group in EXCLUSIVE_GROUPS:
                if group.intersection(layer):
                    for key in group:
                        merged.pop(key, None)
            merged.update(layer)
        return merged

    @classmethod
    def _resolve_stage(
        cls,
        raw_stage: Any,
        index: int,
        defaults: Mapping[str, Any],
        overrides: Mapping[str, Any],
        prior_names: set,
    ) -> Dict[str, Any]:
        """Resolve and validate one stage against all preceding stages."""
        if not isinstance(raw_stage, Mapping):
            raise TypeError(f"Pipeline stage {index} must be a mapping")

        unknown = set(raw_stage) - STAGE_FIELDS
        if unknown:
            raise ValueError(
                f"Pipeline stage {index} contains unknown field(s): "
                + ", ".join(sorted(unknown))
            )
        if not raw_stage.get("name") or not raw_stage.get("config"):
            raise ValueError(f"Pipeline stage {index} must define name and config")

        name = raw_stage["name"]
        if not isinstance(name, str):
            raise TypeError(f"Pipeline stage {index} name must be a string")
        if name in prior_names:
            raise ValueError(f"Duplicate pipeline stage name: {name}")

        depends_on = raw_stage.get("depends_on", [])
        cls._validate_dependencies(name, depends_on, prior_names)

        stage_layer = cls._normalize_aliases(raw_stage, f"Pipeline stage '{name}'")
        stage = cls._merge_layers(defaults, stage_layer, overrides)
        cls._validate_sources(name, stage)
        cls._validate_structured_fields(name, stage)
        cls._validate_lifecycle(name, stage)
        return stage

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
        for field in ("sources", "validation_sources", "module_weight"):
            value = stage.get(field)
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"Pipeline stage '{name}' {field} must be a mapping")

    @classmethod
    def _validate_lifecycle(cls, name: str, stage: Mapping[str, Any]) -> None:
        """Validate train, validation, and inference-only controls."""
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
        if lifecycle != "train" and validation_inputs:
            raise ValueError(
                f"Pipeline stage '{name}' validation inputs require stage=train"
            )
        if lifecycle != "inference" and (
            stage.get("ntasks") is not None or stage.get("files_per_task") is not None
        ):
            raise ValueError(
                f"Pipeline stage '{name}' task splitting requires stage=inference"
            )

    @staticmethod
    def _present(stage: Mapping[str, Any], *keys: str) -> List[str]:
        """Return configured keys while retaining their declared order."""
        return [key for key in keys if key in stage]


class PipelineRunner(SubmissionComponent):
    """Submit a validated pipeline through the single-job batch interface."""

    def submit_pipeline(
        self,
        pipeline_path: str,
        dry_run: bool = False,
        preload: bool = False,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """Submit an ordered multi-stage production pipeline.

        The complete document is validated before the first scheduler call.
        Stages receive ``afterok`` dependencies on declared predecessors, while
        cleanup jobs wait for direct consumers of the artifacts they remove.

        Parameters
        ----------
        pipeline_path : str
            Path to the pipeline YAML document.
        dry_run : bool, optional
            Render submissions without contacting the scheduler.
        preload : bool, optional
            Materialize ``!download`` assets before each stage submission.
        overrides : mapping, optional
            Launch overrides with precedence over all YAML settings.

        Returns
        -------
        dict
            Mapping from stage names to scheduler job IDs.
        """
        stages = PipelineDefinition.load(pipeline_path, overrides).stages
        print(f"Loading pipeline: {pipeline_path}")
        print(f"Stages: {len(stages)}\n")

        job_map: Dict[str, List[str]] = {}
        cleanup_map: Dict[str, List[str]] = {}
        for stage in stages:
            name = stage["name"]
            print(f"Stage: {name}")

            dependency = self._dependency(stage.get("depends_on", []), job_map)
            options = self._submission_options(stage, dependency)
            job_map[name] = self.context.submit_job(
                dry_run=dry_run,
                preload=preload,
                **options,
            )

            cleanup = self._as_list(stage.get("cleanup"))
            if cleanup:
                cleanup_map[name] = cleanup
                print(f"  Cleanup scheduled for: {', '.join(cleanup)}")
            print()

        self._schedule_cleanup(stages, job_map, cleanup_map, dry_run)
        return job_map

    @classmethod
    def _submission_options(
        cls, stage: Mapping[str, Any], dependency: Optional[str]
    ) -> Dict[str, Any]:
        """Translate one stage to ``BatchRunner.submit_job`` options."""
        source_key, files = cls._source(stage, "files", "source", "source_list")
        val_key, val_files = cls._source(stage, "val_source", "val_source_list")

        # Profile fields are passed through; the remaining pipeline vocabulary
        # maps explicitly to the stable single-job API.
        options = {key: stage[key] for key in PROFILE_FIELDS if key in stage}
        options.update(
            {
                "config": stage["config"],
                "files": files,
                "source_type": (
                    "source_list" if source_key == "source_list" else "source"
                ),
                "validation_files": val_files,
                "validation_source_type": (
                    "source_list" if val_key == "val_source_list" else "source"
                ),
                "named_sources": stage.get("sources"),
                "validation_named_sources": stage.get("validation_sources"),
                "module_weights": stage.get("module_weight"),
                "profile": stage.get("profile", "auto"),
                "job_name": stage.get("job_name", stage["name"]),
                "output": stage.get("output"),
                "output_suffix": stage.get("output_suffix"),
                "ntasks": stage.get("ntasks"),
                "files_per_task": stage.get("files_per_task"),
                "dependency": dependency,
                "larcv_path": stage.get("larcv_path"),
                "flashmatch_path": stage.get("flashmatch_path"),
                "spine_path": stage.get("spine_path"),
                "flashmatch": stage.get("flashmatch", False),
                "cvmfs": stage.get("cvmfs", False),
                "apply_mods": cls._as_list(stage.get("apply_mods")),
                "no_writer": stage.get("no_writer", False),
                "set_overrides": cls._as_list(stage.get("set")),
                "stage": stage.get("stage", "inference"),
                "run_dir": stage.get("run_dir"),
                "resume": stage.get("resume", False),
                "resume_from": stage.get("resume_from"),
                "validation_name": stage.get("validation_name"),
                "rerun_validation": stage.get("rerun_validation", False),
                "tensorboard": stage.get("tensorboard", False),
                "world_size": stage.get("world_size"),
                "batch_size": stage.get("batch_size"),
                "minibatch_size": stage.get("minibatch_size"),
                "num_workers": stage.get("num_workers"),
                "epochs": stage.get("epochs"),
                "iterations": stage.get("iterations"),
            }
        )
        return options

    @staticmethod
    def _source(
        stage: Mapping[str, Any], *keys: str
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """Extract and normalize one scalar/list source declaration."""
        configured = [key for key in keys if key in stage]
        key = configured[0] if configured else None
        value = stage.get(key) if key else None
        return key, PipelineRunner._as_list(value)

    @staticmethod
    def _as_list(value: Any) -> Optional[List[Any]]:
        """Normalize an optional scalar to the downstream list form."""
        if value is None:
            return None
        return value if isinstance(value, list) else [value]

    @staticmethod
    def _dependency(
        depends_on: Sequence[str], job_map: Mapping[str, List[str]]
    ) -> Optional[str]:
        """Resolve stage names to a scheduler ``afterok`` expression."""
        job_ids = [
            job_id
            for stage_name in depends_on
            for job_id in job_map.get(stage_name, [])
        ]
        return f"afterok:{':'.join(job_ids)}" if job_ids else None

    def _schedule_cleanup(
        self,
        stages: Sequence[Mapping[str, Any]],
        job_map: Mapping[str, List[str]],
        cleanup_map: Mapping[str, List[str]],
        dry_run: bool,
    ) -> None:
        """Remove artifacts after each producer's direct consumers finish."""
        if not cleanup_map:
            return

        print("\nScheduling cleanup jobs...")
        for producer, paths in cleanup_map.items():
            consumers = [
                stage["name"]
                for stage in stages
                if producer in stage.get("depends_on", [])
            ]
            dependency = self._dependency(consumers, job_map)
            if not consumers:
                print(f"  {producer}: no cleanup (no dependent stages found)")
                continue
            if not dependency:
                # Never delete inputs when a consumer did not return a job ID.
                continue

            print(f"  {producer}: cleanup after {', '.join(consumers)} complete")
            self.batch_client.submit_cleanup_job(
                paths_to_clean=paths,
                job_name=f"cleanup_{producer}",
                dependency=dependency,
                dry_run=dry_run,
            )
