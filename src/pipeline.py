"""Loading, validation, and execution of multi-stage production pipelines."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from .component import SubmissionComponent
from .run_manager import RunManager
from .spine_cli import SpineCLI

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
        "weight_path",
        "export_weights",
        "input_dir",
        "output_dir",
        "checkpoint",
        "dataset",
        "job_name",
        "output",
        "output_suffix",
        "in_place",
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
        "entry_fraction_range",
        "val_entry_fraction_range",
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


class PipelineDefinition:
    """A fully validated pipeline, ready for ordered submission.

    Attributes
    ----------
    stages : tuple of mappings
        Stages after defaults, stage values, and CLI overrides are resolved.
    """

    __slots__ = ("stages", "workspace")
    stages: Tuple[Mapping[str, Any], ...]
    workspace: Optional[str]

    def __init__(
        self,
        stages: Sequence[Mapping[str, Any]],
        workspace: Optional[str] = None,
    ):
        """Store the resolved workspace and immutable submission order."""
        self.stages = tuple(stages)
        self.workspace = workspace

    @classmethod
    def load(
        cls,
        pipeline_path: str,
        overrides: Optional[Mapping[str, Any]] = None,
        workspace_override: Optional[str] = None,
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
        unknown = set(document) - {
            "workspace",
            "variables",
            "collections",
            "defaults",
            "stages",
        }
        if unknown:
            raise ValueError("Unknown pipeline field(s): " + ", ".join(sorted(unknown)))

        workspace = cls._resolve_workspace(document, workspace_override)
        variables = cls._resolve_variables(workspace, document.get("variables", {}))
        collections = cls._resolve_collections(
            document.get("collections", {}), variables
        )
        defaults = cls._require_mapping(
            cls._expand_variables(
                document.get("defaults", {}), variables, "Pipeline defaults"
            ),
            "Pipeline defaults",
        )
        unknown = set(defaults) - GLOBAL_FIELDS
        if unknown:
            raise ValueError(
                "Pipeline defaults contain stage-specific or unknown field(s): "
                + ", ".join(sorted(unknown))
            )

        override_values = cls._require_mapping(
            cls._expand_variables(overrides or {}, variables, "Pipeline overrides"),
            "Pipeline overrides",
        )
        unknown = set(override_values) - GLOBAL_FIELDS
        if unknown:
            raise ValueError(
                "Unknown pipeline override field(s): " + ", ".join(sorted(unknown))
            )

        raw_stages = cls._expand_stage_templates(
            document.get("stages"), variables, collections
        )
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

        return cls(tuple(stages), workspace)

    @staticmethod
    def _resolve_workspace(
        document: Mapping[str, Any], workspace_override: Optional[str]
    ) -> Optional[str]:
        """Resolve the launch workspace before expanding pipeline variables.

        An explicit ``workspace: null`` marks the pipeline as portable and
        requires the submitter to choose its output root. Pipelines that omit
        the field remain valid for workflows without a shared workspace.
        """
        if workspace_override is not None:
            if not isinstance(workspace_override, str):
                raise TypeError("Pipeline workspace override must be a string")
            if not workspace_override:
                raise ValueError("Pipeline workspace override must not be empty")
            return workspace_override

        workspace = document.get("workspace")
        if "workspace" in document and workspace is None:
            raise ValueError(
                "Pipeline workspace is null; specify --workspace when submitting"
            )
        return workspace

    @classmethod
    def _resolve_variables(cls, workspace: Any, raw_variables: Any) -> Dict[str, str]:
        """Validate and resolve pipeline-local string substitutions.

        ``workspace`` is exposed as the reserved ``${workspace}`` variable.
        User variables may reference the workspace or one another, regardless
        of declaration order. Cycles and missing references are rejected before
        stage validation or scheduler interaction.
        """
        if workspace is not None:
            if not isinstance(workspace, str):
                raise TypeError("Pipeline workspace must be a string")
            if not workspace:
                raise ValueError("Pipeline workspace must not be empty")

        variables = cls._require_mapping(raw_variables, "Pipeline variables")
        if "workspace" in variables:
            raise ValueError(
                "Pipeline variable 'workspace' is reserved for the top-level "
                "workspace field"
            )

        unresolved = dict(variables)
        if workspace is not None:
            unresolved["workspace"] = workspace
        for name, value in unresolved.items():
            if not isinstance(name, str) or not VARIABLE_NAME_PATTERN.match(name):
                raise ValueError(
                    "Pipeline variable names must be valid identifiers: " + repr(name)
                )
            if not isinstance(value, str):
                raise TypeError(f"Pipeline variable '{name}' must be a string")

        resolved: Dict[str, str] = {}

        def resolve(name: str, stack: Tuple[str, ...]) -> str:
            """Resolve one variable while retaining its dependency stack."""
            if name in resolved:
                return resolved[name]
            if name not in unresolved:
                raise ValueError(f"Undefined pipeline variable: {name}")
            if name in stack:
                cycle = " -> ".join(stack + (name,))
                raise ValueError(f"Cyclic pipeline variable reference: {cycle}")

            value = unresolved[name]
            next_stack = stack + (name,)
            resolved[name] = VARIABLE_PATTERN.sub(
                lambda match: resolve(match.group(1), next_stack), value
            )
            return resolved[name]

        for name in unresolved:
            resolve(name, ())
        return resolved

    @classmethod
    def _resolve_collections(
        cls, raw_collections: Any, variables: Mapping[str, str]
    ) -> Dict[str, Tuple[Mapping[str, str], ...]]:
        """Validate reusable collections used by stage ``for_each`` blocks.

        Collections are deliberately small data tables: each is a non-empty
        list of flat string mappings. Global pipeline variables are expanded
        in their values before the entries become iteration-local variables.
        """
        collections = cls._require_mapping(raw_collections, "Pipeline collections")
        resolved: Dict[str, Tuple[Mapping[str, str], ...]] = {}
        for name, raw_items in collections.items():
            if not isinstance(name, str) or not VARIABLE_NAME_PATTERN.match(name):
                raise ValueError(
                    "Pipeline collection names must be valid identifiers: " + repr(name)
                )
            if not isinstance(raw_items, list) or not raw_items:
                raise ValueError(
                    f"Pipeline collection '{name}' must be a non-empty list"
                )

            items = []
            for index, raw_item in enumerate(raw_items, start=1):
                context = f"Pipeline collection '{name}' item {index}"
                item = cls._require_mapping(raw_item, context)
                if not item:
                    raise ValueError(f"{context} must not be empty")
                for key, value in item.items():
                    if not isinstance(key, str) or not VARIABLE_NAME_PATTERN.match(key):
                        raise ValueError(
                            f"{context} keys must be valid identifiers: {key!r}"
                        )
                    if not isinstance(value, str):
                        raise TypeError(f"{context} value '{key}' must be a string")
                items.append(cls._expand_variables(item, variables, context))
            resolved[name] = tuple(items)
        return resolved

    @classmethod
    def _expand_stage_templates(
        cls,
        raw_stages: Any,
        variables: Mapping[str, str],
        collections: Mapping[str, Sequence[Mapping[str, str]]],
    ) -> List[Any]:
        """Expand ``for_each`` templates into ordinary concrete stages."""
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("Pipeline must define a non-empty stages list")

        expanded = []
        for index, raw_stage in enumerate(raw_stages, start=1):
            if not isinstance(raw_stage, Mapping):
                raise TypeError(f"Pipeline stage {index} must be a mapping")

            for_each = raw_stage.get("for_each")
            if for_each is None:
                expanded.append(
                    cls._expand_variables(
                        raw_stage, variables, f"Pipeline stage {index}"
                    )
                )
                continue

            context = f"Pipeline stage {index} for_each"
            loop = cls._require_mapping(for_each, context)
            unknown = set(loop) - {"collection", "as"}
            if unknown:
                raise ValueError(
                    f"{context} contains unknown field(s): "
                    + ", ".join(sorted(unknown))
                )
            collection_name = loop.get("collection")
            alias = loop.get("as")
            if not isinstance(collection_name, str) or not collection_name:
                raise TypeError(f"{context} collection must be a non-empty string")
            if collection_name not in collections:
                raise ValueError(
                    f"{context} references unknown collection: {collection_name}"
                )
            if not isinstance(alias, str) or not VARIABLE_NAME_PATTERN.match(alias):
                raise ValueError(f"{context} as must be a valid identifier")
            if alias in variables:
                raise ValueError(
                    f"{context} alias '{alias}' conflicts with a pipeline variable"
                )

            template = {
                key: value for key, value in raw_stage.items() if key != "for_each"
            }
            for item_index, item in enumerate(collections[collection_name], start=1):
                local_variables = dict(variables)
                local_variables.update(
                    {f"{alias}.{key}": value for key, value in item.items()}
                )
                expanded.append(
                    cls._expand_variables(
                        template,
                        local_variables,
                        f"Pipeline stage {index} iteration {item_index}",
                    )
                )
        return expanded

    @classmethod
    def _expand_variables(cls, value: Any, variables: Mapping[str, str], context: str):
        """Recursively expand pipeline variables in strings and containers."""
        if isinstance(value, str):

            def replace(match):
                name = match.group(1)
                if name not in variables:
                    raise ValueError(
                        f"Undefined pipeline variable '{name}' in {context}"
                    )
                return variables[name]

            return VARIABLE_PATTERN.sub(replace, value)
        if isinstance(value, Mapping):
            return {
                key: cls._expand_variables(item, variables, context)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._expand_variables(item, variables, context) for item in value]
        if isinstance(value, tuple):
            return tuple(
                cls._expand_variables(item, variables, context) for item in value
            )
        return value

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
        if not STAGE_NAME_PATTERN.match(name):
            raise ValueError(
                f"Pipeline stage {index} name may contain only letters, "
                "numbers, '.', '_', and '-'"
            )
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
        in_place = stage.get("in_place")
        if in_place is not None and not isinstance(in_place, bool):
            raise TypeError(f"Pipeline stage '{name}' in_place must be a boolean")

    @classmethod
    def _validate_lifecycle(cls, name: str, stage: Mapping[str, Any]) -> None:
        """Validate train, validation, and inference-only controls."""
        kind = stage.get("kind", "spine")
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
        if lifecycle != "train" and stage.get("val_entry_fraction_range") is not None:
            raise ValueError(
                f"Pipeline stage '{name}' validation entry range requires stage=train"
            )
        if lifecycle != "inference" and (
            stage.get("ntasks") is not None or stage.get("files_per_task") is not None
        ):
            raise ValueError(
                f"Pipeline stage '{name}' task splitting requires stage=inference"
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
    def _validate_report(cls, name: str, stage: Mapping[str, Any]) -> None:
        """Require a standalone report contract without SPINE runtime fields."""
        required = ("run_dir", "input_dir", "output_dir")
        missing = [field for field in required if not stage.get(field)]
        if missing:
            raise ValueError(
                f"Pipeline report stage '{name}' requires: " + ", ".join(missing)
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


class PipelineRunner(SubmissionComponent):
    """Submit a validated pipeline through the single-job batch interface."""

    def submit_pipeline(
        self,
        pipeline_path: str,
        dry_run: bool = False,
        preload: bool = False,
        overrides: Optional[Mapping[str, Any]] = None,
        workspace: Optional[str] = None,
        from_stage: Optional[str] = None,
        to_stage: Optional[str] = None,
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
        workspace : str, optional
            Shared output root exposed to the pipeline as ``${workspace}``.
            This overrides a concrete YAML value and satisfies a required
            ``workspace: null`` declaration.
        from_stage : str, optional
            Restart at this stage in pipeline order. Earlier stages are
            treated as completed, and selected stage directories are retried
            without discarding prior submission records.
        to_stage : str, optional
            Stop after this stage in pipeline order. This can bound a restart
            to the stages that must be regenerated.

        Returns
        -------
        dict
            Mapping from stage names to scheduler job IDs.
        """
        definition = PipelineDefinition.load(
            pipeline_path,
            overrides,
            workspace_override=workspace,
        )
        all_stages = definition.stages
        stages, skipped, deferred = self._select_stages(
            all_stages,
            from_stage,
            to_stage,
        )
        print(f"Loading pipeline: {pipeline_path}")
        if definition.workspace is not None:
            print(f"Workspace: {definition.workspace}")
        print(f"Stages: {len(stages)}")
        if skipped:
            print(f"Skipped as completed: {', '.join(skipped)}")
        if deferred:
            print(f"Not selected after stop: {', '.join(deferred)}")
        print()

        if definition.workspace is not None:
            self._prepare_log_index(definition.workspace, stages)

        job_map: Dict[str, List[str]] = {}
        cleanup_map: Dict[str, List[str]] = {}
        for stage in stages:
            name = stage["name"]
            print(f"Stage: {name}")

            dependency = self._dependency(stage.get("depends_on", []), job_map)
            skipped_dependencies = [
                dependency_name
                for dependency_name in stage.get("depends_on", [])
                if dependency_name in skipped
            ]
            if skipped_dependencies:
                print(
                    "  Reusing completed dependencies: "
                    + ", ".join(skipped_dependencies)
                )
            options = self._submission_options(
                stage,
                dependency,
                retry=from_stage is not None,
            )
            if stage.get("kind", "spine") == "report":
                job_map[name] = self.context.submit_report(
                    dry_run=dry_run,
                    **options,
                )
            else:
                job_map[name] = self.context.submit_job(
                    dry_run=dry_run,
                    preload=preload,
                    **options,
                )
            if definition.workspace is not None and stage.get("run_dir"):
                self._link_stage_attempt(
                    definition.workspace,
                    name,
                    stage["run_dir"],
                )

            cleanup = self._as_list(stage.get("cleanup"))
            if cleanup:
                cleanup_map[name] = cleanup
                print(f"  Cleanup scheduled for: {', '.join(cleanup)}")
            print()

        self._schedule_cleanup(stages, job_map, cleanup_map, dry_run)
        return job_map

    @staticmethod
    def _prepare_log_index(workspace: str, stages: Sequence[Mapping[str, Any]]) -> None:
        """Validate the workspace's shallow stage-attempt index up front."""
        log_dir = Path(workspace).expanduser().resolve() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        for stage in stages:
            link = log_dir / stage["name"]
            if link.exists() and not link.is_symlink():
                raise ValueError(f"Pipeline log index path is not a symlink: {link}")

    @staticmethod
    def _link_stage_attempt(workspace: str, name: str, run_dir: str) -> None:
        """Link ``workspace/logs/<stage>`` to that stage's latest attempt."""
        link = Path(workspace).expanduser().resolve() / "logs" / name
        latest = Path(run_dir).expanduser().resolve() / "latest"
        target = Path(os.path.relpath(str(latest), str(link.parent)))
        RunManager.replace_symlink(link, target)

    @staticmethod
    def _select_stages(
        stages: Sequence[Mapping[str, Any]],
        from_stage: Optional[str],
        to_stage: Optional[str],
    ) -> Tuple[Sequence[Mapping[str, Any]], List[str], List[str]]:
        """Select an inclusive ordered range and describe omitted stages."""
        names = [stage["name"] for stage in stages]
        start = 0
        stop = len(stages)
        if from_stage is not None:
            if not isinstance(from_stage, str) or not from_stage:
                raise ValueError("Pipeline from_stage must be a non-empty string")
            if from_stage not in names:
                raise ValueError(f"Unknown pipeline restart stage: {from_stage}")
            start = names.index(from_stage)
        if to_stage is not None:
            if not isinstance(to_stage, str) or not to_stage:
                raise ValueError("Pipeline to_stage must be a non-empty string")
            if to_stage not in names:
                raise ValueError(f"Unknown pipeline stop stage: {to_stage}")
            stop = names.index(to_stage) + 1
        if stop <= start:
            raise ValueError("Pipeline --to-stage must not precede --from-stage")
        return stages[start:stop], names[:start], names[stop:]

    @classmethod
    def _submission_options(
        cls,
        stage: Mapping[str, Any],
        dependency: Optional[str],
        retry: bool = False,
    ) -> Dict[str, Any]:
        """Translate one stage to ``BatchRunner.submit_job`` options."""
        if stage.get("kind", "spine") == "report":
            options = {key: stage[key] for key in PROFILE_FIELDS if key in stage}
            options.update(
                {
                    "config": stage["config"],
                    "input_dir": stage["input_dir"],
                    "output_dir": stage["output_dir"],
                    "run_dir": stage["run_dir"],
                    "checkpoint": stage.get("checkpoint"),
                    "dataset": stage.get("dataset"),
                    "profile": stage.get("profile", "s3df_milano"),
                    "job_name": stage.get("job_name", stage["name"]),
                    "dependency": dependency,
                    "spine_path": stage.get("spine_path"),
                    "cvmfs": stage.get("cvmfs", False),
                    "retry": retry,
                }
            )
            return options

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
                "weight_path": stage.get("weight_path"),
                "export_weights": stage.get("export_weights"),
                "profile": stage.get("profile", "auto"),
                "job_name": stage.get("job_name", stage["name"]),
                "output": stage.get("output"),
                "output_suffix": stage.get("output_suffix"),
                "in_place": stage.get("in_place", False),
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
                # Exact files produced by declared predecessors do not exist
                # yet; scheduler dependencies make them available at runtime.
                "allow_missing_inputs": bool(stage.get("depends_on")),
                "retry": retry,
                "world_size": stage.get("world_size"),
                "batch_size": stage.get("batch_size"),
                "minibatch_size": stage.get("minibatch_size"),
                "num_workers": stage.get("num_workers"),
                "epochs": stage.get("epochs"),
                "iterations": stage.get("iterations"),
                "entry_fraction_range": stage.get("entry_fraction_range"),
                "val_entry_fraction_range": stage.get("val_entry_fraction_range"),
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
