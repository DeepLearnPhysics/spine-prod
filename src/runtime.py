"""Software setup, executable, bind, and container runtime resolution."""

import os
import shlex
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from .component import SubmissionComponent


class RuntimeResolver(SubmissionComponent):
    """Resolve local and container execution environments for SPINE."""

    @staticmethod
    def resolve_setup_path(
        path: Optional[str], option_name: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve a software setup path to a source command and bind root."""
        if not path:
            return None, None

        root = Path(path).expanduser()
        configure_script = root / "configure.sh"
        if not configure_script.is_file():
            raise RuntimeError(
                f"{option_name} must point to a directory containing configure.sh: {root}"
            )

        return f"source {shlex.quote(str(configure_script))}", str(root)

    @staticmethod
    def default_container_version() -> str:
        """Return the repository default SPINE container version."""
        version_path = Path(__file__).resolve().parents[1] / "DEFAULT_SPINE_VERSION"
        with open(version_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def container_version() -> str:
        """Return configured container version, or the repo default fallback."""
        if os.environ.get("SPINE_PROD_CONFIGURED") == "1":
            version = os.environ.get("SPINE_CONTAINER_VERSION")
            if version:
                return version

        return RuntimeResolver.default_container_version()

    @staticmethod
    def default_container_path() -> str:
        """Build the default local SIF path from the configured SPINE version."""
        version = RuntimeResolver.container_version()
        version = version[1:] if version.startswith("v") else version
        path_version = version.replace(".", "-")
        return f"/sdf/data/neutrino/images/spine_v{path_version}.sif"

    @staticmethod
    def resolve_spine_command(
        spine_path: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve the SPINE command and any extra bind root it requires."""
        configured = spine_path or os.environ.get("SPINE_LOCAL_PATH")
        if configured:
            configured_path = Path(configured).expanduser()
            if configured_path.is_dir():
                bind_root = configured_path
                candidate_commands = [
                    configured_path / "bin" / "spine",
                    configured_path / "bin" / "run.py",
                ]
            else:
                if configured_path.parent.name == "bin":
                    bind_root = configured_path.parent.parent
                else:
                    bind_root = configured_path.parent
                candidate_commands = [configured_path]

            for candidate in candidate_commands:
                if candidate.exists() and candidate.is_file():
                    quoted_candidate = shlex.quote(str(candidate))
                    if candidate.suffix == ".py":
                        return f"python3 {quoted_candidate}", str(bind_root)
                    return quoted_candidate, str(bind_root)

            option_name = "--spine-path" if spine_path else "SPINE_LOCAL_PATH"
            raise RuntimeError(
                f"{option_name} is set, but no local SPINE executable was found. "
                "Expected a file path or a directory containing bin/spine or "
                "bin/run.py."
            )

        local_spine = shutil.which("spine")
        if local_spine:
            return shlex.quote(local_spine), None

        return None, None

    @staticmethod
    def merge_bind_paths(
        bind_paths: Optional[str], extra_paths: Optional[List[str]] = None
    ) -> Optional[str]:
        """Merge user bind roots with required extra paths."""
        resolved_paths = []
        seen_paths = set()

        for raw_value in [bind_paths, *(extra_paths or [])]:
            if not raw_value:
                continue

            for path in str(raw_value).split(","):
                stripped = path.strip()
                if not stripped or stripped in seen_paths:
                    continue
                seen_paths.add(stripped)
                resolved_paths.append(stripped)

        return ",".join(resolved_paths) if resolved_paths else None

    @staticmethod
    def default_bind_paths_for_site(site: Optional[str]) -> Optional[str]:
        """Return the implicit bind roots required by a site template."""
        if site == "s3df":
            return "/sdf/"
        return None

    @staticmethod
    def container_tag_for_cli() -> str:
        """Return the configured container tag in Docker/Podman CLI form."""
        version = RuntimeResolver.container_version()
        version = version[1:] if version.startswith("v") else version
        tag = os.environ.get(
            "SPINE_CONTAINER_TAG", f"docker:ghcr.io/deeplearnphysics/spine:{version}"
        )
        return tag[len("docker:") :] if tag.startswith("docker:") else tag

    @staticmethod
    def sif_runtime_executable() -> Optional[str]:
        """Return the Singularity/Apptainer executable for local SIF execution."""
        configured = os.environ.get("SPINE_CONTAINER_RUNTIME_BIN")
        if configured:
            runtime = shutil.which(configured)
            if not runtime:
                raise RuntimeError(
                    "SPINE_CONTAINER_RUNTIME_BIN is set, but no executable was found "
                    f"for: {configured}"
                )
            return runtime

        return shutil.which("singularity") or shutil.which("apptainer")

    @staticmethod
    def sif_runtime_args() -> str:
        """Return additional args for interactive Singularity/Apptainer execution."""
        configured = os.environ.get("SPINE_CONTAINER_RUNTIME_ARGS", "")
        if not configured:
            return ""

        return " ".join(shlex.quote(arg) for arg in shlex.split(configured))

    @staticmethod
    def interactive_bind_paths(bind_paths: Optional[str] = None) -> str:
        """Return the bind roots for interactive Singularity/Apptainer execution."""
        resolved_paths = {str(Path.cwd())}
        if bind_paths:
            resolved_paths.update(
                path.strip() for path in bind_paths.split(",") if path.strip()
            )
        return ",".join(sorted(resolved_paths))

    def build_interactive_container_command(
        self, inner_cmd: str, cvmfs: bool, bind_paths: Optional[str] = None
    ) -> str:
        """Build an interactive container command for local smoke tests."""
        container_path = os.environ.get(
            "SPINE_CONTAINER_PATH", self.default_container_path()
        )

        if Path(container_path).exists():
            singularity = self.sif_runtime_executable()
            if singularity:
                runtime_args = self.sif_runtime_args()
                runtime_args = f" {runtime_args}" if runtime_args else ""
                resolved_bind_paths = self.interactive_bind_paths(bind_paths)
                resolved_bind_path_set = set(resolved_bind_paths.split(","))
                resolved_bind_path_set.add(str(self.basedir))
                if cvmfs:
                    resolved_bind_path_set.add("/cvmfs")
                bind_arg = ",".join(sorted(resolved_bind_path_set))
                return (
                    f"{shlex.quote(singularity)} exec{runtime_args} --bind {shlex.quote(bind_arg)} "
                    f"--nv {shlex.quote(container_path)} bash -c "
                    f"{shlex.quote(inner_cmd)}"
                )

        docker = shutil.which("docker") or shutil.which("podman")
        if docker:
            workdir = str(Path.cwd())
            platform = os.environ.get("SPINE_CONTAINER_PLATFORM", "linux/amd64")
            platform_arg = f"--platform {shlex.quote(platform)} " if platform else ""
            volume_args = [f"-v {shlex.quote(f'{workdir}:{workdir}')}"]
            basedir = str(self.basedir)
            if basedir != workdir:
                volume_args.append(f"-v {shlex.quote(f'{basedir}:{basedir}')}")
            if cvmfs and Path("/cvmfs").exists():
                volume_args.append("-v /cvmfs:/cvmfs:ro")

            env_args = [
                f"-e SPINE_PROD_BASEDIR={shlex.quote(basedir)}",
                f"-e SPINE_CONFIG_PATH={shlex.quote(os.environ.get('SPINE_CONFIG_PATH', str(self.basedir / 'config')))}",
            ]
            for data_dir_var in ("ICARUS_DATA_DIR", "SBND_DATA_DIR"):
                if os.environ.get(data_dir_var):
                    data_dir = shlex.quote(os.environ[data_dir_var])
                    env_args.append(f"-e {data_dir_var}={data_dir}")

            return (
                f"{shlex.quote(docker)} run --rm {platform_arg}{' '.join(volume_args)} "
                f"-w {shlex.quote(workdir)} {' '.join(env_args)} "
                f"{shlex.quote(self.container_tag_for_cli())} bash -c "
                f"{shlex.quote(inner_cmd)}"
            )

        raise RuntimeError(
            "Interactive container runtime requested, but no usable runtime was "
            "found. Install spine on PATH, provide a readable SPINE_CONTAINER_PATH "
            "with singularity/apptainer, or make docker/podman available with "
            "SPINE_CONTAINER_TAG."
        )
