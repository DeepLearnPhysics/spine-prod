"""Resolve pipeline defaults, variables, templates and stage declarations."""

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from .fields import (
    EXCLUSIVE_GROUPS,
    GLOBAL_FIELDS,
    STAGE_FIELDS,
    STAGE_NAME_PATTERN,
    VARIABLE_NAME_PATTERN,
    VARIABLE_PATTERN,
)
from .validation import StageValidator


class PipelineDefinition(StageValidator):
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
        stage_module_weights: Optional[Mapping[str, Mapping[str, str]]] = None,
        source_list: Optional[str] = None,
        val_source_list: Optional[str] = None,
    ) -> "PipelineDefinition":
        """Load and validate a pipeline without contacting a scheduler.

        Configuration precedence is ``defaults < stage < global CLI overrides
        < stage-specific module weights``. Every stage is validated before the
        definition is returned, preventing a malformed later stage from
        leaving a partially submitted workflow.
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
        raw_variables = dict(
            cls._require_mapping(document.get("variables", {}), "Pipeline variables")
        )
        for flag, name, value in [
            ("--source-list", "train_source_list", source_list),
            ("--val-source-list", "validation_source_list", val_source_list),
        ]:
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Pipeline {flag} requires a non-empty list path")
            if name not in raw_variables:
                raise ValueError(
                    f"Pipeline must declare variables.{name} to use {flag}"
                )
            raw_variables[name] = os.path.abspath(os.path.expanduser(value))
        variables = cls._resolve_variables(workspace, raw_variables)
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

        cls._apply_stage_module_weights(stages, stage_module_weights or {})
        return cls(tuple(stages), workspace)

    @staticmethod
    def parse_stage_module_weights(
        values: Optional[Sequence[Sequence[str]]],
    ) -> Dict[str, Dict[str, str]]:
        """Parse repeated ``STAGE MODULE=PATH`` command-line assignments.

        Parameters
        ----------
        values : sequence of two-string sequences, optional
            Raw values collected by ``argparse`` for repeated
            ``--stage-module-weight`` options.

        Returns
        -------
        dict
            Stage names mapped to destination modules and checkpoint paths.
        """
        result: Dict[str, Dict[str, str]] = {}
        for value in values or ():
            if isinstance(value, (str, bytes)) or len(value) != 2:
                raise ValueError("--stage-module-weight requires STAGE and MODULE=PATH")
            stage, assignment = value
            if not isinstance(stage, str) or not STAGE_NAME_PATTERN.match(stage):
                raise ValueError(
                    "--stage-module-weight STAGE must be a valid pipeline stage name"
                )
            if not isinstance(assignment, str) or "=" not in assignment:
                raise ValueError(
                    "--stage-module-weight assignment must use MODULE=PATH"
                )
            module, path = assignment.split("=", 1)
            if not module or not VARIABLE_NAME_PATTERN.match(module):
                raise ValueError(
                    "--stage-module-weight MODULE must be a valid identifier"
                )
            if not path:
                raise ValueError("--stage-module-weight PATH must not be empty")
            stage_weights = result.setdefault(stage, {})
            if module in stage_weights:
                raise ValueError(
                    "Duplicate --stage-module-weight assignment for "
                    f"{stage}:{module}"
                )
            stage_weights[module] = path
        return result

    @classmethod
    def _apply_stage_module_weights(
        cls,
        stages: Sequence[Dict[str, Any]],
        overrides: Mapping[str, Mapping[str, str]],
    ) -> None:
        """Merge validated launch-time weights into named SPINE stages."""
        stage_map = {stage["name"]: stage for stage in stages}
        unknown = set(overrides) - set(stage_map)
        if unknown:
            raise ValueError(
                "Unknown pipeline stage module-weight override(s): "
                + ", ".join(sorted(unknown))
            )

        # Validate every target before mutating any resolved stage.
        for name, module_weights in overrides.items():
            stage = stage_map[name]
            if stage.get("kind", "spine") != "spine":
                raise ValueError(
                    f"Pipeline stage '{name}' cannot receive module weights"
                )
            if not isinstance(module_weights, Mapping):
                raise TypeError(
                    f"Pipeline stage '{name}' module-weight override must be a mapping"
                )
            cls._validate_module_weights(name, module_weights)

        for name, module_weights in overrides.items():
            stage = stage_map[name]
            merged = dict(stage.get("module_weight") or {})
            merged.update(module_weights)
            stage["module_weight"] = merged

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
    ) -> Dict[str, Tuple[Mapping[str, Any], ...]]:
        """Validate reusable collections used by stage ``for_each`` blocks.

        Collections are deliberately small data tables: each is a non-empty
        list of flat scalar mappings. Global pipeline variables are expanded
        in string values before entries become iteration-local variables.
        """
        collections = cls._require_mapping(raw_collections, "Pipeline collections")
        resolved: Dict[str, Tuple[Mapping[str, Any], ...]] = {}
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
                    if not isinstance(value, (str, int, float, bool)):
                        raise TypeError(f"{context} value '{key}' must be a scalar")
                items.append(cls._expand_variables(item, variables, context))
            resolved[name] = tuple(items)
        return resolved

    @classmethod
    def _expand_stage_templates(
        cls,
        raw_stages: Any,
        variables: Mapping[str, str],
        collections: Mapping[str, Sequence[Mapping[str, Any]]],
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

            exact = VARIABLE_PATTERN.fullmatch(value)
            if exact:
                name = exact.group(1)
                if name not in variables:
                    raise ValueError(
                        f"Undefined pipeline variable '{name}' in {context}"
                    )
                return variables[name]

            def replace(match):
                """Resolve one variable match while reporting its configuration context."""
                name = match.group(1)
                if name not in variables:
                    raise ValueError(
                        f"Undefined pipeline variable '{name}' in {context}"
                    )
                replacement = variables[name]
                if not isinstance(replacement, str):
                    raise TypeError(
                        f"Non-string pipeline variable '{name}' cannot be embedded "
                        f"in a larger string in {context}"
                    )
                return replacement

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
