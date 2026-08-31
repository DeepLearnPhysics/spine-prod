"""Formatting and validation for SPINE command-line arguments."""

import shlex
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


class SpineCLI:
    """Pure helpers for producing deterministic SPINE CLI fragments."""

    @staticmethod
    def format_set_overrides(set_overrides: Optional[List[str]]) -> str:
        """Format SPINE ``--set`` overrides for shell execution."""
        if not set_overrides:
            return ""

        formatted = []
        for override in set_overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid --set override '{override}'. Expected KEY=VALUE."
                )
            if override.split("=", 1)[0].strip() == "base.world_size":
                raise ValueError(
                    "base.world_size is managed from the GPU allocation; "
                    "use --world-size only as a matching assertion"
                )
            if any(char.isspace() for char in override) or any(
                char in override for char in ("'", '"')
            ):
                raise ValueError(
                    f"Invalid --set override '{override}'. Whitespace and quotes "
                    "are not supported in submit.py --set values."
                )
            formatted.append(f"--set {override}")

        return " ".join(formatted)

    @staticmethod
    def format_named_sources(
        sources: Optional[Mapping[str, Mapping[str, Any]]],
        validation: bool = False,
    ) -> str:
        """Format target-qualified composite dataset source overrides."""
        if not sources:
            return ""

        direct_option = "--val-source" if validation else "--source"
        list_option = "--val-source-list" if validation else "--source-list"
        direct_values = []
        list_values = []
        for target, source_cfg in sources.items():
            if not isinstance(source_cfg, Mapping):
                raise TypeError(f"Named source '{target}' must be a mapping")
            selectors = [key for key in ("source", "source_list") if key in source_cfg]
            if len(selectors) != 1:
                raise ValueError(
                    f"Named source '{target}' must specify exactly one of: "
                    "source, source_list"
                )

            selector = selectors[0]
            values = source_cfg[selector]
            if selector == "source":
                if not isinstance(values, list):
                    values = [values]
                if not values:
                    raise ValueError(f"Named source '{target}' cannot be empty")
                direct_values.extend(
                    shlex.quote(f"{target}={value}") for value in values
                )
            else:
                if isinstance(values, list):
                    if len(values) != 1:
                        raise ValueError(
                            f"Named source-list '{target}' accepts exactly one file"
                        )
                    values = values[0]
                list_values.append(shlex.quote(f"{target}={values}"))

        parts = []
        if direct_values:
            parts.append(f"{direct_option} {' '.join(direct_values)}")
        if list_values:
            parts.append(f"{list_option} {' '.join(list_values)}")
        return " ".join(parts)

    @staticmethod
    def format_module_weights(
        module_weights: Optional[Mapping[str, str]],
    ) -> str:
        """Format module-specific checkpoint overrides."""
        if not module_weights:
            return ""
        values = []
        for module, path in module_weights.items():
            if not module or not path:
                raise ValueError("Module weight assignments require a module and path")
            values.append(shlex.quote(f"{module}={path}"))
        return f"--module-weight {' '.join(values)}"

    @staticmethod
    def format_runtime_options(
        world_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        minibatch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        epochs: Optional[float] = None,
        iterations: Optional[int] = None,
    ) -> str:
        """Format first-class SPINE runtime CLI overrides."""
        options = [
            ("--world-size", world_size),
            ("--batch-size", batch_size),
            ("--minibatch-size", minibatch_size),
            ("--num-workers", num_workers),
            ("--epochs", epochs),
            ("--iterations", iterations),
        ]
        return " ".join(
            f"{flag} {value}" for flag, value in options if value is not None
        )

    @staticmethod
    def align_world_size(
        profile_config: Dict, requested_world_size: Optional[int]
    ) -> Optional[int]:
        """Align SPINE process count with the scheduler GPU allocation."""
        site = profile_config.get("site", "s3df")
        if site == "s3df" and "gpus" in profile_config:
            allocated_gpus = int(profile_config["gpus"])
        elif site != "s3df" and "gpus_per_node" in profile_config:
            nodes = int(profile_config.get("nodes", 1))
            if nodes != 1:
                raise ValueError(
                    "Multi-node GPU submissions are not supported; use a single node"
                )
            allocated_gpus = int(profile_config["gpus_per_node"])
        else:
            return requested_world_size

        if requested_world_size is not None and requested_world_size != allocated_gpus:
            raise ValueError(
                f"--world-size {requested_world_size} conflicts with the "
                f"scheduler allocation of {allocated_gpus} GPU(s)"
            )
        return allocated_gpus

    @staticmethod
    def default_writer_output_settings(
        job_dir: Path, config: str, suffix: Optional[str] = None
    ) -> Tuple[str, str]:
        """Return default HDF5 writer directory and suffix settings."""
        return str(job_dir / "output"), suffix or Path(config).stem

    @staticmethod
    def format_output_args(output: Optional[str], directory: str, suffix: str) -> str:
        """Format SPINE output arguments for explicit or derived writer naming."""
        if output:
            output_path = Path(output)
            if output_path.suffix:
                return f"--output {shlex.quote(output)}"

            directory = output

        return " ".join(
            [
                f"--output-dir {shlex.quote(directory)}",
                f"--output-suffix {shlex.quote(suffix)}",
            ]
        )

    @staticmethod
    def warn_no_writer_deprecated() -> None:
        """Warn that the deprecated no-writer option is ignored."""
        print(
            "WARNING: --no-writer is deprecated and ignored with SPINE "
            "v0.15.3+. Output options are still passed; SPINE safely ignores "
            "them when the config has no io.writer block."
        )
