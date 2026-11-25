"""Submit resolved pipeline stages and manage dependency gates."""

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..runs import RunManager
from ..sites import allocation_limits, map_stage, validate_allocation
from ..submission.base import SubmissionComponent
from .definition import PipelineDefinition
from .fields import PROFILE_FIELDS


class PipelineRunner(SubmissionComponent):
    """Submit a validated pipeline through the single-job batch interface."""

    def submit_pipeline(
        self,
        pipeline_path: str,
        dry_run: bool = False,
        preload: bool = False,
        overrides: Optional[Mapping[str, Any]] = None,
        workspace: Optional[str] = None,
        continuation_dir: Optional[str] = None,
        from_stage: Optional[str] = None,
        to_stage: Optional[str] = None,
        select_stages: Optional[Sequence[str]] = None,
        stage_module_weights: Optional[Sequence[Sequence[str]]] = None,
        warm_start: Optional[str] = None,
        source_list: Optional[str] = None,
        val_source_list: Optional[str] = None,
        skip_stages: Optional[Sequence[str]] = None,
        site: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_archive_dir: Optional[str] = None,
        audit_dir: Optional[str] = None,
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
        continuation_dir : str, optional
            Separate shared continuation state root; NERSC requires Scratch.
            Training logs, checkpoints and outputs stay in the workspace.
        cache_dir : str, optional
            Relocate declared working caches and their downstream references.
        cache_archive_dir : str, optional
            Durable cache snapshots copied by CPU jobs after each producer.
        audit_dir : str, optional
            Historical continuation records; defaults to workspace/monitoring.
        from_stage : str, optional
            Restart at this stage in pipeline order. Earlier stages are
            treated as completed, and selected stage directories are retried
            without discarding prior submission records.
        to_stage : str, optional
            Stop after this stage in pipeline order. This can bound a restart
            to the stages that must be regenerated.
        select_stages : sequence of str, optional
            Submit only these named stages. Their pipeline order is retained,
            and dependencies are contracted through omitted stages.
        skip_stages : sequence of str, optional
            Exclude these stages from the selected range, reusing existing
            artifacts and contracting dependencies through omitted stages.
        stage_module_weights : sequence, optional
            Repeated launch-time ``STAGE MODULE=PATH`` assignments. These
            override matching module weights in the pipeline document.
        warm_start : str, optional
            Full-chain checkpoint used to initialize every selected training
            stage that declares a ``warm_start`` module mapping.

        site : str, optional
            Map allocation roles to a site before validating all selected requests.
        source_list : str, optional
            Override the declared ``train_source_list`` pipeline variable.
        val_source_list : str, optional
            Override the declared ``validation_source_list`` pipeline variable.

        Returns
        -------
        dict
            Mapping from stage names to scheduler job IDs.
        """
        # Source-list overrides expand with variables before any stage is submitted.
        parsed_stage_weights = PipelineDefinition.parse_stage_module_weights(
            stage_module_weights
        )
        # Preserve source walltimes/profiles until site mapping captures training budgets.
        definition = PipelineDefinition.load(
            pipeline_path,
            {
                key: value
                for key, value in (overrides or {}).items()
                if site is None or key not in ("time", "profile")
            },
            workspace_override=workspace,
            stage_module_weights=parsed_stage_weights,
            source_list=source_list,
            val_source_list=val_source_list,
        )
        from .storage import archive_plans, validate_cache_storage, working_stages

        all_stages = working_stages(definition.stages, definition.workspace, cache_dir)
        if warm_start is not None:
            if not isinstance(warm_start, str) or not warm_start:
                raise ValueError("Pipeline warm-start path must be a non-empty string")
            if not any(stage.get("warm_start") for stage in all_stages):
                raise ValueError(
                    "--warm-start was provided, but the pipeline declares no "
                    "warm_start mappings"
                )
        stages, skipped, deferred = self._select_stages(
            all_stages,
            from_stage,
            to_stage,
            select_stages,
            skip_stages,
        )
        # Validate every selected allocation before creating directories or jobs.
        if site is not None:
            stages = [
                map_stage(self.profiles, stage, site, overrides) for stage in stages
            ]
        for stage in stages:
            profile = self.config_mgr.get_profile(stage.get("profile", "auto"))
            profile.update({key: stage[key] for key in PROFILE_FIELDS if key in stage})
            try:
                validate_cache_storage(stage, profile.get("site"))
                if stage.get("continuation") is not None:
                    from ..continuation.policy import profile_policy
                    from ..continuation.state import validate_storage

                    profile_policy(self.profiles, profile, stage["continuation"])
                    validate_storage(continuation_dir, profile.get("site"))
                validate_allocation(self.profiles, profile)
            except ValueError as exc:
                raise ValueError(f"Stage {stage['name']}: {exc}") from exc
            if site is not None:
                print(
                    f"{stage['name']}: {stage['profile']} "
                    f"GPUs={profile.get('gpus', profile.get('gpus_per_node', 0))} "
                    f"CPUs={profile['cpus_per_task']} time={profile['time']} "
                    f"memory={profile.get('mem_per_node', profile.get('mem_per_cpu', profile.get('mem', 'node default')))} "
                    f"queue={profile.get('queue', profile.get('qos', profile.get('partition')))} "
                    f"max_time={allocation_limits(self.profiles, profile).get('max_time') or 'unknown'}"
                )
        selected_names = {stage["name"] for stage in stages}
        inactive_weight_stages = set(parsed_stage_weights) - selected_names
        print(f"Loading pipeline: {pipeline_path}")
        if definition.workspace is not None:
            print(f"Workspace: {definition.workspace}")
        print(f"Stages: {len(stages)}")
        if skipped and select_stages is None:
            print(f"Skipped as completed: {', '.join(skipped)}")
        elif skipped:
            print(f"Not selected: {', '.join(skipped)}")
        if deferred:
            print(f"Not selected after stop: {', '.join(deferred)}")
        if inactive_weight_stages:
            print(
                "Inactive module-weight overrides: "
                + ", ".join(sorted(inactive_weight_stages))
            )
        print()

        archives = archive_plans(
            stages, definition.workspace, cache_archive_dir, self.context
        )
        if definition.workspace is not None:
            self._prepare_log_index(definition.workspace, stages)

        if any("continuation" in stage for stage in (*all_stages, *stages)):
            from ..continuation.controller import start

            if from_stage or select_stages or skip_stages or to_stage:
                raise ValueError(
                    "Continuation pipelines use the durable controller; stage slicing is unsupported"
                )
            return start(
                self.context,
                stages,
                definition.workspace,
                dry_run,
                preload,
                warm_start,
                continuation_dir,
                cache_archives=archives,
                audit_dir=audit_dir,
            )

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
                retry=(
                    from_stage is not None
                    or select_stages is not None
                    or skip_stages is not None
                ),
                warm_start_path=warm_start,
            )
            kind = stage.get("kind", "spine")
            if kind == "filter":
                job_map[name] = self.context.submit_filter(
                    dry_run=dry_run,
                    **options,
                )
            elif kind == "report":
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
            if name in archives:
                archive_dependency = self._dependency([name], job_map)
                job_map[name] = self.context.submit_archive(
                    **archives[name],
                    dependency=archive_dependency,
                    dry_run=dry_run,
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
        select_stages: Optional[Sequence[str]] = None,
        skip_stages: Optional[Sequence[str]] = None,
    ) -> Tuple[Sequence[Mapping[str, Any]], List[str], List[str]]:
        """Select an ordered range or dependency-aware sparse stage set."""
        names = [stage["name"] for stage in stages]
        if skip_stages is not None:
            if select_stages is not None:
                raise ValueError(
                    "Pipeline skip_stages cannot be combined with select_stages"
                )
            if not skip_stages:
                raise ValueError("Pipeline skip_stages must not be empty")
            if any(not isinstance(name, str) or not name for name in skip_stages):
                raise ValueError("Pipeline skip_stages must contain non-empty strings")
            if len(set(skip_stages)) != len(skip_stages):
                raise ValueError("Pipeline skip_stages must not contain duplicates")
            unknown = set(skip_stages) - set(names)
            if unknown:
                raise ValueError(
                    "Unknown skipped pipeline stage(s): " + ", ".join(sorted(unknown))
                )
            stage_range, _, deferred = PipelineRunner._select_stages(
                stages, from_stage, to_stage
            )
            selected_names = [
                stage["name"]
                for stage in stage_range
                if stage["name"] not in skip_stages
            ]
            if not selected_names:
                raise ValueError("Pipeline skip_stages leaves no stages selected")
            selected, omitted, _ = PipelineRunner._select_stages(
                stages, None, None, select_stages=selected_names
            )
            return (
                selected,
                [name for name in omitted if name not in deferred],
                deferred,
            )
        if select_stages is not None:
            if from_stage is not None or to_stage is not None:
                raise ValueError(
                    "Pipeline select_stages cannot be combined with stage boundaries"
                )
            if not select_stages:
                raise ValueError("Pipeline select_stages must not be empty")
            if any(not isinstance(name, str) or not name for name in select_stages):
                raise ValueError(
                    "Pipeline select_stages must contain non-empty strings"
                )
            if len(set(select_stages)) != len(select_stages):
                raise ValueError("Pipeline select_stages must not contain duplicates")
            unknown = set(select_stages) - set(names)
            if unknown:
                raise ValueError(
                    "Unknown selected pipeline stage(s): " + ", ".join(sorted(unknown))
                )

            selected_names = set(select_stages)
            stage_map = {stage["name"]: stage for stage in stages}

            def selected_dependencies(name: str) -> List[str]:
                """Find the nearest selected ancestors of one selected stage."""
                result: List[str] = []

                def visit(dependency: str) -> None:
                    """Collect selected predecessors in dependency order."""
                    if dependency in selected_names:
                        if dependency not in result:
                            result.append(dependency)
                        return
                    for ancestor in stage_map[dependency].get("depends_on", []):
                        visit(ancestor)

                for dependency in stage_map[name].get("depends_on", []):
                    visit(dependency)
                return result

            selected = []
            for stage in stages:
                if stage["name"] in selected_names:
                    resolved = dict(stage)
                    resolved["depends_on"] = selected_dependencies(stage["name"])
                    selected.append(resolved)
            omitted = [name for name in names if name not in selected_names]
            return selected, omitted, []

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
        warm_start_path: Optional[str] = None,
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
                    "dataset_selection": stage.get("dataset_selection"),
                    "profile": stage.get("profile", "s3df_milano"),
                    "job_name": stage.get("job_name", stage["name"]),
                    "dependency": dependency,
                    "spine_path": stage.get("spine_path"),
                    "cvmfs": stage.get("cvmfs", False),
                    "retry": retry,
                }
            )
            return options

        if stage.get("kind", "spine") == "filter":
            options = {key: stage[key] for key in PROFILE_FIELDS if key in stage}
            options.update(
                {
                    "config": stage["config"],
                    "operation": stage["operation"],
                    "run_dir": stage["run_dir"],
                    "sources": cls._as_list(stage.get("source")),
                    "source_list": stage.get("source_list"),
                    "cache_dir": stage["cache_dir"],
                    "output": stage.get("output"),
                    "output_source_list": stage.get("output_source_list"),
                    "workers": stage.get("workers"),
                    "force": stage.get("force", False),
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
                "warm_start_path": (
                    warm_start_path if stage.get("warm_start") else None
                ),
                "warm_start_modules": (
                    stage.get("warm_start") if warm_start_path else None
                ),
                "weight_path": stage.get("weight_path"),
                "export_weights": stage.get("export_weights"),
                "profile": stage.get("profile", "auto"),
                "job_name": stage.get("job_name", stage["name"]),
                "output": stage.get("output"),
                "output_suffix": stage.get("output_suffix"),
                "output_source_list": stage.get("output_source_list"),
                "in_place": stage.get("in_place", False),
                "cache_repository": stage.get("cache_repository"),
                "cache_stage": stage.get("cache_stage"),
                "ntasks": stage.get("ntasks"),
                "files_per_task": stage.get("files_per_task"),
                "joint_file_mode": stage.get("joint_file_mode", "broadcast"),
                "joint_validate_trees": stage.get("joint_validate_trees", False),
                "dependency": dependency,
                "larcv_path": stage.get("larcv_path"),
                "flashmatch_path": stage.get("flashmatch_path"),
                "spine_path": stage.get("spine_path"),
                "flashmatch": stage.get("flashmatch", False),
                "cvmfs": stage.get("cvmfs", False),
                "expandable_segments": stage.get("expandable_segments", False),
                "apply_mods": cls._as_list(stage.get("apply_mods")),
                "no_writer": stage.get("no_writer", False),
                "set_overrides": cls._as_list(stage.get("set")),
                "stage": stage.get("stage", "inference"),
                "run_dir": stage.get("run_dir"),
                "resume": stage.get("resume", False),
                "resume_from": stage.get("resume_from"),
                **(
                    {"continuation": stage["continuation"]}
                    if "continuation" in stage
                    else {}
                ),
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
                "num_files": stage.get("num_files"),
                "val_num_files": stage.get("val_num_files"),
                "num_entries": stage.get("num_entries"),
                "val_num_entries": stage.get("val_num_entries"),
                "entry_fraction_range": stage.get("entry_fraction_range"),
                "val_entry_fraction_range": stage.get("val_entry_fraction_range"),
                "entry_filter": stage.get("entry_filter"),
                "val_entry_filter": stage.get("val_entry_filter"),
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
