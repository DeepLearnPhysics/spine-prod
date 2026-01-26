#!/usr/bin/env python3
"""
SPINE Production SLURM Submission System

A modern, flexible job submission orchestrator for SPINE reconstruction
on SLURM-based HPC systems.

Usage:
    ./submit.py --config infer/icarus/latest.cfg --source-list file_list.txt
    ./submit.py --config infer/icarus/latest_data.cfg --source data/*.root --profile s3df_ampere
    ./submit.py --pipeline pipelines/icarus_production.yaml
    ./submit.py --config ... --source ... --local-output
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from jinja2 import Template

from version import __version__


class SlurmSubmitter:
    """Orchestrates SLURM job submissions for SPINE production"""

    def __init__(self, basedir: Optional[Path] = None, local_output: bool = False):
        if local_output:
            self.basedir = Path(os.getcwd())
        else:
            self.basedir = basedir or Path(__file__).parent
        self.profiles = self._load_profiles()
        self.template = self._load_template()
        self.jobs_dir = self.basedir / "jobs"
        self.jobs_dir.mkdir(exist_ok=True)

        # Check environment
        if not os.getenv("SPINE_PROD_BASEDIR") and not local_output:
            print("WARNING: SPINE_PROD_BASEDIR not set. Did you source configure.sh?")

    def _load_profiles(self) -> Dict:
        """Load resource profiles from YAML.

        Returns
        -------
        Dict
            Resource profiles configuration
        """
        profiles_path = self.basedir / "templates" / "profiles.yaml"
        if not profiles_path.exists():
            raise FileNotFoundError(f"Profiles not found: {profiles_path}")

        with open(profiles_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_template(self) -> Template:
        """Load SBATCH job template.

        Returns
        -------
        Template
            Jinja2 template for SBATCH script
        """
        template_path = self.basedir / "templates" / "job_template.sbatch"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            return Template(f.read())

    def _detect_detector(self, config: str) -> str:
        """Auto-detect detector from config path.

        Parameters
        ----------
        config : str
            Path to configuration file

        Returns
        -------
        str
            Detector name or 'generic'
        """
        config_path = Path(config)
        for detector in self.profiles["detectors"]:
            if detector in str(config_path):
                return detector
        return "unknown_detector"

    def _get_profile(self, profile_name: str, detector: Optional[str] = None) -> Dict:
        """Get resource profile with detector defaults.

        Parameters
        ----------
        profile_name : str
            Name of the profile or 'auto'
        detector : Optional[str], optional
            Detector name for auto profile selection, by default None

        Returns
        -------
        Dict
            Resource profile configuration
        """
        if profile_name == "auto":
            if detector and detector in self.profiles["detectors"]:
                profile_name = self.profiles["detectors"][detector]["default_profile"]
            else:
                profile_name = "s3df_ampere"

        if profile_name not in self.profiles["profiles"]:
            raise ValueError(f"Unknown profile: {profile_name}")

        return self.profiles["profiles"][profile_name]

    def _parse_files(
        self, file_input: List[str], source_type: str = "source"
    ) -> List[str]:
        """Parse file input (paths, globs, or txt file).

        Parameters
        ----------
        file_input : List[str]
            List of file paths, glob patterns, or text file path
        source_type : str, optional
            Either 'source' (direct paths/globs) or 'source_list' (text file),
            by default 'source'

        Returns
        -------
        List[str]
            Expanded list of file paths
        """
        files = []

        if source_type == "source_list":
            # Read from a single text file (one file path per line)
            if len(file_input) != 1:
                raise ValueError("--source-list/-S accepts exactly one text file")
            source_list_path = file_input[0]
            with open(source_list_path, "r", encoding="utf-8") as f:
                files.extend([line.strip() for line in f if line.strip()])
        else:
            # Handle direct sources (paths, globs)
            for item in file_input:
                if "*" in item or "?" in item:
                    # Expand glob
                    files.extend(glob.glob(item))
                else:
                    # Direct file path
                    if os.path.exists(item):
                        files.append(item)
                    else:
                        print(f"WARNING: File not found: {item}")

        return files

    def _discover_modifiers(self, config: str) -> Dict[str, List[Path]]:
        """Discover available modifier configs in the same directory as the base config.

        Parameters
        ----------
        config : str
            Path to the base configuration file or detector directory

        Returns
        -------
        Dict[str, List[Path]]
            Dict mapping modifier names to lists of available version paths
            (excluding *_common.yaml).
            Example: {'data': [Path('mod_data_240719.yaml'), Path('mod_data_250625.yaml')]}
        """
        config_path = Path(config).resolve()
        # If path is a directory, use it directly; if file, use its parent
        if config_path.is_dir():
            config_dir = config_path
        else:
            config_dir = config_path.parent
        detector = config_dir.name

        modifiers = {}

        # First check for modifier/ subdirectory (new YAML structure)
        modifier_dir = config_dir / "modifier"
        if modifier_dir.exists():
            # Look for versioned modifier files in subdirectories
            for subdir in modifier_dir.iterdir():
                if subdir.is_dir():
                    mod_name = subdir.name
                    # Find all versioned modifiers (exclude *_common.yaml)
                    yaml_files = [
                        f
                        for f in subdir.glob("mod_*.yaml")
                        if not f.stem.endswith("_common")
                    ]
                    if yaml_files:
                        modifiers[mod_name] = sorted(yaml_files, key=lambda p: p.stem)

        # Fall back to old pattern: <detector>_*_mod.{yaml,cfg} in same directory
        if not modifiers:
            for pattern in [f"{detector}_*_mod.yaml", f"{detector}_*_mod.cfg"]:
                modifier_files = list(config_dir.glob(pattern))
                for mod_file in modifier_files:
                    # Extract modifier name: "2x2_data_mod.yaml" -> "data"
                    mod_name = mod_file.stem.replace(f"{detector}_", "").replace(
                        "_mod", ""
                    )
                    if mod_name not in modifiers:
                        modifiers[mod_name] = []
                    modifiers[mod_name].append(mod_file)

        return modifiers

    def _extract_version(self, config_path: Path) -> Optional[str]:
        """Extract version number from config filename (YYMMDD format).

        Parameters
        ----------
        config_path : Path
            Path to config file

        Returns
        -------
        Optional[str]
            Version string (e.g., '250625') or None if no version found
        """
        # Look for 6-digit date pattern (YYMMDD)
        match = re.search(r"_(\d{6})(?:_|\.|$)", config_path.stem)
        return match.group(1) if match else None

    def _resolve_modifier_version(
        self,
        mod_name: str,
        available_versions: List[Path],
        base_version: Optional[str],
        explicit_version: Optional[str],
    ) -> Path:
        """Resolve which version of a modifier to use.

        Parameters
        ----------
        mod_name : str
            Name of the modifier (e.g., 'data')
        available_versions : List[Path]
            List of available version paths, sorted by version
        base_version : Optional[str]
            Version from base config (e.g., '250625')
        explicit_version : Optional[str]
            User-specified version (e.g., '240719')

        Returns
        -------
        Path
            Path to the selected modifier file
        """
        if not available_versions:
            raise ValueError(f"No versions found for modifier '{mod_name}'")

        # If user specified a version explicitly
        if explicit_version:
            for version_path in available_versions:
                if explicit_version in version_path.stem:
                    print(f"  {mod_name}: Using explicit version {explicit_version}")
                    return version_path
            raise ValueError(
                f"Modifier '{mod_name}' version '{explicit_version}' not found. "
                f"Available: {[self._extract_version(p) or p.stem for p in available_versions]}"
            )

        # If base config has a version, try to match or find closest earlier
        if base_version:
            # Look for exact match
            for version_path in available_versions:
                if base_version in version_path.stem:
                    print(f"  {mod_name}: Using matching version {base_version}")
                    return version_path

            # Find closest earlier version
            earlier_versions = [
                v
                for v in available_versions
                if (self._extract_version(v) or "") <= base_version
            ]
            if earlier_versions:
                selected = earlier_versions[-1]  # Latest of the earlier versions
                selected_ver = self._extract_version(selected) or selected.stem
                print(
                    f"  {mod_name}: No exact match for {base_version}, using closest earlier "
                    f"version {selected_ver}"
                )
                return selected
            else:
                print(
                    f"  {mod_name}: WARNING - No earlier version found for {base_version}, "
                    "using oldest available"
                )
                return available_versions[0]

        # No version in base config - use latest
        selected = available_versions[-1]
        selected_ver = self._extract_version(selected) or selected.stem
        print(
            f"  {mod_name}: No version in base config, using latest version {selected_ver}"
        )
        return selected

    def _create_composite_config(
        self,
        base_config: str,
        modifiers: List[str],
        job_dir: Path,
        detector: Optional[str] = None,
    ) -> str:
        """Create a composite config that includes base + modifiers.

        Parameters
        ----------
        base_config : str
            Path to the base configuration file
        modifiers : List[str]
            List of modifier specs to apply (e.g., ['data', 'lite:250625', '/path/to/custom.yaml'])
        job_dir : Path
            Job directory to save the composite config
        detector : Optional[str], optional
            Detector name (for generated configs in job_dir), by default None

        Returns
        -------
        str
            Path to the generated composite config file
        """
        config_path = Path(base_config).resolve()
        base_name = config_path.stem
        base_version = self._extract_version(config_path)

        # Discover available modifiers
        # If detector is provided, look in the detector's config directory
        if detector:
            modifier_search_path = str(self.basedir / "config" / "infer" / detector)
        else:
            modifier_search_path = base_config

        available_mods = self._discover_modifiers(modifier_search_path)

        # Parse modifier specs and resolve versions
        resolved_mods = []
        mod_names = []  # For naming the composite file

        print(
            f"Resolving modifiers for base config version: {base_version or 'unversioned'}"
        )

        for mod_spec in modifiers:
            # Check if it's a file path
            if (
                "/" in mod_spec
                or mod_spec.endswith(".yaml")
                or mod_spec.endswith(".cfg")
            ):
                mod_path = Path(mod_spec)
                if not mod_path.exists():
                    raise ValueError(f"Custom modifier file not found: {mod_spec}")
                resolved_mods.append(mod_path.resolve())
                mod_names.append(mod_path.stem.replace("mod_", ""))
                print(f"  custom: Using custom file {mod_path}")
                continue

            # Parse modifier:version syntax
            if ":" in mod_spec:
                mod_name, explicit_version = mod_spec.split(":", 1)
            else:
                mod_name = mod_spec
                explicit_version = None

            # Validate modifier exists
            if mod_name not in available_mods:
                raise ValueError(
                    f"Unknown modifier: {mod_name}. "
                    f"Available: {', '.join(available_mods.keys())}"
                )

            # Resolve version
            selected_path = self._resolve_modifier_version(
                mod_name, available_mods[mod_name], base_version, explicit_version
            )
            resolved_mods.append(selected_path)
            mod_names.append(mod_name)

        # Create composite config content
        composite_content = "# Auto-generated composite configuration\n"
        composite_content += f"# Base: {base_config}\n"
        composite_content += f"# Modifiers: {', '.join(modifiers)}\n"
        composite_content += f"# Generated: {datetime.now().isoformat()}\n\n"

        # Save composite config first to determine its location
        composite_name = f"{base_name}_{'_'.join(mod_names)}_composite.yaml"
        composite_path = job_dir / composite_name

        # Make all paths relative to the composite config location
        composite_content += "include:\n"
        try:
            rel_base = os.path.relpath(config_path, composite_path.parent)
            composite_content += f"  - {rel_base}\n"
        except ValueError:
            # On Windows, relpath fails if paths are on different drives
            composite_content += f"  - {config_path}\n"

        for mod_path in resolved_mods:
            try:
                rel_mod = os.path.relpath(mod_path, composite_path.parent)
                composite_content += f"  - {rel_mod}\n"
            except ValueError:
                composite_content += f"  - {mod_path}\n"

        # Write the composite config
        with open(composite_path, "w", encoding="utf-8") as f:
            f.write(composite_content)

        print(f"Created composite config: {composite_path}")
        print(f"  Base: {Path(base_config).name}")
        print(f"  Modifiers: {', '.join(modifiers)}")

        return str(composite_path)

    def _chunk_files(
        self, files: List[str], max_array_size: int, files_per_task: int
    ) -> List[List[str]]:
        """Split files into chunks for array jobs.

        Parameters
        ----------
        files : List[str]
            List of file paths to process
        max_array_size : int
            Maximum array size for SLURM job arrays
        files_per_task : int
            Number of files to process per array task

        Returns
        -------
        List[List[str]]
            List of file chunks, each chunk is a list of comma-separated file groups
        """
        # Group files by files_per_task
        file_groups = []
        for i in range(0, len(files), files_per_task):
            group = files[i : i + files_per_task]
            file_groups.append(",".join(group))

        # Split into chunks that fit array size limit
        chunks = []
        for i in range(0, len(file_groups), max_array_size):
            chunks.append(file_groups[i : i + max_array_size])

        return chunks

    def _create_job_dir(self, job_name: str) -> Path:
        """Create directory for job artifacts.

        Parameters
        ----------
        job_name : str
            Name of the job

        Returns
        -------
        Path
            Path to created job directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir = self.jobs_dir / f"{timestamp}_{job_name}"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "logs").mkdir(exist_ok=True)
        (job_dir / "output").mkdir(exist_ok=True)
        return job_dir

    def _save_job_metadata(self, job_dir: Path, metadata: Dict):
        """Save job metadata for tracking.

        Parameters
        ----------
        job_dir : Path
            Job directory path
        metadata : Dict
            Metadata dictionary to save
        """
        metadata_path = job_dir / "job_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _submit_sbatch(self, script_path: Path, dry_run: bool = False) -> Optional[str]:
        """Submit job to SLURM and return job ID.

        Parameters
        ----------
        script_path : Path
            Path to SBATCH script
        dry_run : bool, optional
            If True, print script without submitting, by default False

        Returns
        -------
        Optional[str]
            Job ID if submitted, None otherwise
        """
        if dry_run:
            print(f"[DRY RUN] Would submit: {script_path}")
            with open(script_path, "r", encoding="utf-8") as f:
                print(f.read())
            return None

        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print(f"ERROR: sbatch failed: {result.stderr}")
            return None

        # Extract job ID from "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        return job_id

    def _submit_cleanup_job(
        self,
        paths_to_clean: List[str],
        job_name: str,
        dependency: str,
        dry_run: bool = False,
    ) -> Optional[str]:
        """Submit a cleanup job to remove intermediate files.

        Parameters
        ----------
        paths_to_clean : List[str]
            List of file/directory paths to remove
        job_name : str
            Name for the cleanup job
        dependency : str
            SLURM dependency specification
        dry_run : bool, optional
            If True, print without submitting, by default False

        Returns
        -------
        Optional[str]
            Job ID if submitted, None otherwise
        """
        if dry_run:
            print(f"[DRY RUN] Would schedule cleanup of: {', '.join(paths_to_clean)}")
            return None

        # Create cleanup script
        cleanup_script = f"""#!/bin/bash
#SBATCH --account=neutrino:ml
#SBATCH --partition=milano
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1g
#SBATCH --time=0:10:00
#SBATCH --job-name={job_name}
#SBATCH --dependency={dependency}

echo "Cleanup job started: $(date)"
echo "Removing intermediate files..."

"""

        for path in paths_to_clean:
            cleanup_script += f"""if [ -e "{path}" ]; then
    echo "  Removing: {path}"
    rm -rf "{path}"
else
    echo "  Not found (already removed?): {path}"
fi

"""

        cleanup_script += """echo "Cleanup completed: $(date)"
"""

        # Write cleanup script
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleanup_dir = self.jobs_dir / f"{timestamp}_cleanup_{job_name}"
        cleanup_dir.mkdir(parents=True, exist_ok=True)

        script_path = cleanup_dir / "cleanup.sbatch"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(cleanup_script)
        script_path.chmod(0o755)

        # Submit
        result = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            print(f"WARNING: Cleanup job submission failed: {result.stderr}")
            return None

        job_id = result.stdout.strip().split()[-1]
        print(f"  Cleanup job ID: {job_id}")
        return job_id

    def _create_latest_config(self, detector: str, job_dir: Path) -> str:
        """Create a composite config from the latest versions of all components.

        Parameters
        ----------
        detector : str
            Detector name (e.g., 'icarus', 'sbnd')
        job_dir : Path
            Job directory to save the generated config

        Returns
        -------
        str
            Path to the generated latest config file
        """
        config_dir = self.basedir / "config" / "infer" / detector

        if not config_dir.exists():
            raise ValueError(f"Detector config directory not found: {config_dir}")

        # Component subdirectories to check (in order)
        component_dirs = ["base", "io", "model", "post"]
        latest_components = {}
        latest_version = None

        print(f"Composing latest config for {detector}:")

        for component in component_dirs:
            comp_dir = config_dir / component
            if not comp_dir.exists():
                continue

            # Find all versioned files (exclude *_common.yaml)
            yaml_files = [
                f
                for f in comp_dir.glob(f"{component}_*.yaml")
                if not f.stem.endswith("_common")
            ]

            if yaml_files:
                # Sort by version and take the latest
                latest = sorted(
                    yaml_files, key=lambda p: self._extract_version(p) or ""
                )[-1]
                latest_components[component] = latest
                version = self._extract_version(latest)
                version_str = version or latest.stem
                print(f"  {component:8} -> {version_str}")

                # Track the latest version number for naming
                if version and (not latest_version or version > latest_version):
                    latest_version = version

        if not latest_components:
            raise ValueError(
                f"No versioned components found for {detector}. "
                f"Expected files like base_YYMMDD.yaml, io_YYMMDD.yaml, etc."
            )

        # Create composite config with version in filename
        composite_content = "# Auto-generated 'latest' configuration\n"
        composite_content += f"# Detector: {detector}\n"
        composite_content += f"# Generated: {datetime.now().isoformat()}\n"
        composite_content += f"# Components: {', '.join(latest_components.keys())}\n"
        if latest_version:
            composite_content += f"# Latest version: {latest_version}\n"
        composite_content += "\n"

        composite_content += "include:\n"
        for component in component_dirs:
            if component in latest_components:
                comp_path = latest_components[component]
                try:
                    rel_path = os.path.relpath(comp_path, job_dir)
                    composite_content += f"  - {rel_path}\n"
                except ValueError:
                    composite_content += f"  - {comp_path}\n"

        # Save the config with version in filename
        composite_name = f"{detector}_latest"
        if latest_version:
            composite_name += f"_{latest_version}"
        composite_name += "_composite.yaml"
        composite_path = job_dir / composite_name

        with open(composite_path, "w", encoding="utf-8") as f:
            f.write(composite_content)

        print(f"Created latest config: {composite_path}")
        return str(composite_path)

    def list_modifiers(self, config_path: str) -> Dict:
        """List available modifiers for a given configuration file.

        Parameters
        ----------
        config_path : str
            Path to the base configuration file

        Returns
        -------
        Dict
            Dictionary with 'base_version', 'config_name', and 'modifiers' keys.
            Each modifier contains 'selected', 'available', and 'paths' information.
        """
        config_path_obj = Path(config_path)
        base_version = self._extract_version(config_path_obj)
        modifiers = self._discover_modifiers(config_path)

        result = {
            "base_version": base_version,
            "config_name": config_path_obj.name,
            "modifiers": {},
        }

        for mod_name, version_paths in sorted(modifiers.items()):
            versions = [
                self._extract_version(p)
                or p.stem.replace("mod_", "").replace(f"{mod_name}_", "")
                for p in version_paths
            ]
            selected = self._resolve_modifier_version(
                mod_name, version_paths, base_version, None
            )
            selected_ver = self._extract_version(selected) or selected.stem

            result["modifiers"][mod_name] = {
                "selected": selected_ver,
                "available": versions,
                "paths": version_paths,
            }

        return result

    def run_interactive(
        self,
        config: str,
        files: List[str],
        source_type: str = "source",
        output: Optional[str] = None,
        files_per_task: int = 1,
        task_id: int = 1,
        larcv_basedir: Optional[str] = None,
        flashmatch: bool = False,
        apply_mods: Optional[List[str]] = None,
    ) -> int:
        """Run SPINE processing interactively (no SLURM submission).

        This mode performs all config composition and file preparation like
        submit_job(), but executes the SPINE command directly in the current
        shell instead of submitting to SLURM. Useful for testing configs.

        Parameters
        ----------
        config : str
            Path to SPINE configuration file
        files : List[str]
            List of input files (direct paths/globs or source list path)
        source_type : str, optional
            Either 'source' (direct paths/globs) or 'source_list' (text file)
        output : str, optional
            Output file path
        files_per_task : int, optional
            Files to process per task, by default 1
        task_id : int, optional
            Which task to run (1-indexed), by default 1
        larcv_basedir : str, optional
            Custom LArCV installation path
        flashmatch : bool, optional
            Enable flash matching
        apply_mods : List[str], optional
            List of modifiers to apply

        Returns
        -------
        int
            Exit code from SPINE execution
        """
        # Parse input files
        file_list = self._parse_files(files, source_type)
        if not file_list:
            raise ValueError("No input files found")

        print(f"Found {len(file_list)} file(s) to process")

        # Detect detector
        detector = self._detect_detector(config)
        config_path = Path(config)
        config_name = config_path.stem

        # Check if this is a "latest" request
        is_latest = config_name == "latest" or "latest" in config_path.parts

        job_name = f"interactive_{detector}_{config_name}"
        job_dir = self._create_job_dir(job_name)

        # Handle "latest" config generation
        if is_latest:
            print(f"\nDetected 'latest' config request for {detector}")
            config = self._create_latest_config(detector, job_dir)

        # Apply modifiers if specified
        if apply_mods:
            config = self._create_composite_config(
                config, apply_mods, job_dir, detector=detector if is_latest else None
            )

        # Determine output path
        if not output:
            output = str(job_dir / "output" / f"{job_name}.h5")

        # Ensure output directory exists
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        # Chunk files for processing
        max_array_size = self.profiles["defaults"]["max_array_size"]
        file_chunks = self._chunk_files(file_list, max_array_size, files_per_task)

        # Validate task_id
        if task_id < 1 or task_id > len(file_chunks):
            raise ValueError(f"Task ID {task_id} out of range (1-{len(file_chunks)})")

        # Get the file group for this task (it's a list of comma-separated file strings)
        file_group_list = file_chunks[task_id - 1]

        # Create temporary file list for this task
        task_file_list = job_dir / f"interactive_files_task_{task_id}.txt"
        with open(task_file_list, "w", encoding="utf-8") as f:
            for file_group in file_group_list:
                # Each file_group is a comma-separated string of files
                for file_path in file_group.split(","):
                    f.write(f"{file_path}\n")

        # Count total files for display
        total_files = sum(len(fg.split(",")) for fg in file_group_list)
        print(f"\nRunning task {task_id}/{len(file_chunks)}")
        print(f"Processing {total_files} file(s):")
        for file_group in file_group_list:
            for file_path in file_group.split(","):
                print(f"  {file_path}")

        # Build command
        cmd_parts = []

        # Add environment setup if needed
        if larcv_basedir:
            cmd_parts.append(f"source {larcv_basedir}/configure.sh")

        if flashmatch:
            cmd_parts.append("source $FMATCH_BASEDIR/configure.sh")

        # Build SPINE command
        spine_cmd = (
            f"python3 $SPINE_BASEDIR/bin/run.py "
            f"-S {task_file_list} "
            f"-o {output} "
            f"-c {config}"
        )
        cmd_parts.append(spine_cmd)

        # Join with && for proper sequencing
        full_cmd = " && ".join(cmd_parts)

        print("\nExecuting:")
        print(f"  {full_cmd}\n")
        print("=" * 80)

        # Execute directly
        result = subprocess.run(full_cmd, shell=True)

        print("=" * 80)
        print(f"\nInteractive execution completed with exit code: {result.returncode}")
        print(f"Job directory: {job_dir}")
        if result.returncode == 0:
            print(f"Output: {output}")

        return result.returncode

    def submit_job(
        self,
        config: str,
        files: List[str],
        source_type: str = "source",
        profile: str = "auto",
        job_name: Optional[str] = None,
        output: Optional[str] = None,
        ntasks: Optional[int] = None,
        files_per_task: int = 1,
        dependency: Optional[str] = None,
        larcv_basedir: Optional[str] = None,
        flashmatch: bool = False,
        apply_mods: Optional[List[str]] = None,
        dry_run: bool = False,
        **profile_overrides,
    ) -> List[str]:
        """Submit SLURM job for SPINE processing.

        Parameters
        ----------
        config : str
            Path to SPINE configuration file
        files : List[str]
            List of input files (direct paths/globs or source list path)
        source_type : str, optional
            Either 'source' (direct paths/globs) or 'source_list' (text file),
            by default 'source'
        profile : str, optional
            Resource profile name or 'auto', by default 'auto'
        job_name : str, optional
            Custom job name, by default None
        output : str, optional
            Output file path, by default None
        ntasks : int, optional
            Number of parallel tasks (default: auto), by default None
        files_per_task : int, optional
            Files to process per task, by default 1
        dependency : str, optional
            SLURM dependency string, by default None
        larcv_basedir : str, optional
            Custom LArCV installation path, by default None
        flashmatch : bool, optional
            Enable flash matching, by default False
        apply_mods : List[str], optional
            List of modifiers to apply (e.g., ['data', 'flash']), by default None
        dry_run : bool, optional
            Show what would be submitted without submitting, by default False
        **profile_overrides
            Override profile settings

        Returns
        -------
        List[str]
            List of submitted job IDs
        """
        # Parse input files
        file_list = self._parse_files(files, source_type)
        if not file_list:
            raise ValueError("No input files found")

        print(f"Found {len(file_list)} file(s) to process")

        # Detect detector first
        detector = self._detect_detector(config)

        # Create job directory first (needed for composite/latest config generation)
        config_path = Path(config)
        config_name = config_path.stem

        # Check if this is a "latest" request
        is_latest = config_name == "latest" or "latest" in config_path.parts

        if not job_name:
            job_name = f"spine_{detector}_{config_name}"

        job_dir = self._create_job_dir(job_name)

        # Handle "latest" config generation
        if is_latest:
            print(f"\nDetected 'latest' config request for {detector}")
            config = self._create_latest_config(detector, job_dir)
            config_name = Path(config).stem

        # Apply modifiers if specified
        original_config = config
        if apply_mods:
            # Pass detector if config was generated (to find modifiers in config dir)
            config = self._create_composite_config(
                config, apply_mods, job_dir, detector=detector if is_latest else None
            )

        # Detect detector and get profile
        profile_config = self._get_profile(profile, detector)
        profile_config.update(profile_overrides)

        # Get account
        account = profile_config.get("account")
        if not account and detector in self.profiles["detectors"]:
            account = self.profiles["detectors"][detector].get(
                "account", self.profiles["defaults"]["account"]
            )

        # Determine output path
        if not output:
            output = str(job_dir / "output" / f"{job_name}.h5")

        # Chunk files for array jobs
        max_array_size = self.profiles["defaults"]["max_array_size"]
        file_chunks = self._chunk_files(file_list, max_array_size, files_per_task)

        print(f"Splitting into {len(file_chunks)} array job(s)")

        job_ids = []
        for chunk_idx, chunk in enumerate(file_chunks):
            # Create file list for this chunk
            file_list_path = job_dir / f"files_chunk_{chunk_idx}.txt"
            with open(file_list_path, "w", encoding="utf-8") as f:
                for _, file_group in enumerate(chunk, 1):
                    f.write(f"{file_group}\n")

            # Render SBATCH script
            array_spec = f"1-{len(chunk)}"
            if ntasks and ntasks < len(chunk):
                array_spec += f"%{ntasks}"

            script_content = self.template.render(
                account=account,
                partition=profile_config["partition"],
                cpus_per_task=profile_config["cpus_per_task"],
                mem_per_cpu=profile_config["mem_per_cpu"],
                time=profile_config["time"],
                gpus=profile_config["gpus"],
                array_spec=array_spec,
                job_name=(
                    f"{job_name}_{chunk_idx}" if len(file_chunks) > 1 else job_name
                ),
                log_dir=str(job_dir / "logs"),
                dependency=dependency,
                basedir=str(self.basedir),
                file_list=str(file_list_path),
                config=config,
                output=output,
                larcv_basedir=larcv_basedir,
                flashmatch=flashmatch,
            )

            # Write script
            script_path = job_dir / f"submit_chunk_{chunk_idx}.sbatch"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)
            script_path.chmod(0o755)

            # Submit
            print(f"\nSubmitting chunk {chunk_idx + 1}/{len(file_chunks)}:")
            print(f"  Script: {script_path}")
            print(f"  Files: {len(chunk)}")
            print(f"  Profile: {profile} ({profile_config['description']})")

            job_id = self._submit_sbatch(script_path, dry_run)
            if job_id:
                job_ids.append(job_id)
                print(f"  Job ID: {job_id}")

        # Save metadata
        metadata = {
            "spine_prod_version": __version__,
            "job_name": job_name,
            "detector": detector,
            "config": config,
            "original_config": original_config if apply_mods else config,
            "applied_modifiers": apply_mods or [],
            "profile": profile,
            "profile_config": profile_config,
            "num_files": len(file_list),
            "num_chunks": len(file_chunks),
            "job_ids": job_ids,
            "output": output,
            "submitted": datetime.now().isoformat(),
            "command": " ".join(sys.argv),
        }
        self._save_job_metadata(job_dir, metadata)

        print(f"\nJob directory: {job_dir}")
        print(f"Job metadata: {job_dir}/job_metadata.json")

        return job_ids

    def submit_pipeline(
        self, pipeline_path: str, dry_run: bool = False
    ) -> Dict[str, List[str]]:
        """
        Submit multi-stage pipeline with dependencies

        Args:
            pipeline_path: Path to pipeline YAML file
            dry_run: Show what would be submitted

        Returns:
            Dict mapping stage names to job IDs
        """
        with open(pipeline_path, "r", encoding="utf-8") as f:
            pipeline = yaml.safe_load(f)

        print(f"Loading pipeline: {pipeline_path}")
        print(f"Stages: {len(pipeline['stages'])}\n")

        job_map = {}
        cleanup_map = {}  # Track stages that need cleanup

        for stage in pipeline["stages"]:
            stage_name = stage["name"]
            print(f"Stage: {stage_name}")

            # Build dependency string
            depends_on = stage.get("depends_on", [])
            dependency = None
            if depends_on:
                dep_jobs = []
                for dep_stage in depends_on:
                    if dep_stage in job_map:
                        dep_jobs.extend(job_map[dep_stage])
                if dep_jobs:
                    dependency = f"afterok:{':'.join(dep_jobs)}"

            # Submit stage
            job_ids = self.submit_job(
                config=stage["config"],
                files=stage["files"],
                profile=stage.get("profile", "auto"),
                job_name=stage.get("job_name", stage_name),
                output=stage.get("output"),
                ntasks=stage.get("ntasks"),
                files_per_task=stage.get("files_per_task", 1),
                dependency=dependency,
                larcv_basedir=stage.get("larcv_basedir"),
                flashmatch=stage.get("flashmatch", False),
                dry_run=dry_run,
            )

            job_map[stage_name] = job_ids

            # Store cleanup info if requested
            cleanup = stage.get("cleanup", [])
            if cleanup:
                if not isinstance(cleanup, list):
                    cleanup = [cleanup]
                cleanup_map[stage_name] = {"paths": cleanup, "job_ids": job_ids}
                print(f"  Cleanup scheduled for: {', '.join(cleanup)}")

            print()

        # Schedule cleanup jobs for stages that have downstream dependencies
        if cleanup_map:
            print("\nScheduling cleanup jobs...")
            for stage_name, cleanup_info in cleanup_map.items():
                # Find all stages that depend on this one
                dependent_stages = []
                for stage in pipeline["stages"]:
                    if stage_name in stage.get("depends_on", []):
                        dependent_stages.append(stage["name"])

                if dependent_stages:
                    # Wait for all dependent stages to complete
                    dep_jobs = []
                    for dep_stage in dependent_stages:
                        if dep_stage in job_map:
                            dep_jobs.extend(job_map[dep_stage])

                    if dep_jobs:
                        dependency = f"afterok:{':'.join(dep_jobs)}"
                        print(
                            f"  {stage_name}: cleanup after {', '.join(dependent_stages)} complete"
                        )

                        self._submit_cleanup_job(
                            paths_to_clean=cleanup_info["paths"],
                            job_name=f"cleanup_{stage_name}",
                            dependency=dependency,
                            dry_run=dry_run,
                        )
                else:
                    print(f"  {stage_name}: no cleanup (no dependent stages found)")

        return job_map


def main():
    parser = argparse.ArgumentParser(
        description="SPINE Production SLURM Submission System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic submission with source list (recommended)
  %(prog)s --config infer/icarus/icarus_full_chain_co_250625.yaml --source-list file_list.txt

  # Direct sources with glob pattern
  %(prog)s --config infer/icarus/icarus_full_chain_co_250625.yaml --source data/*.root

  # Interactive mode (test locally without SLURM)
  %(prog)s --interactive --config infer/icarus/icarus_full_chain_co_250625.yaml --source test.root
  %(prog)s -I --config infer/icarus/icarus_full_chain_co_250625.yaml --source-list files.txt --task-id 2

  # With custom profile
  %(prog)s --config infer/icarus/icarus_full_chain_co_250625.yaml --source data/*.root --profile s3df_ampere

  # Apply modifiers at runtime
  %(prog)s --config infer/icarus/icarus_full_chain_co_250625.yaml --source data/*.root --apply-mods data
  %(prog)s --config infer/icarus/icarus_full_chain_co_250625.yaml --source data/*.root --apply-mods data lite

  # List available modifiers for a config
  %(prog)s --list-mods infer/icarus/icarus_full_chain_co_250625.yaml

  # Multiple files per task
  %(prog)s --config infer/icarus/icarus_full_chain_co_250625.yaml --source-list files.txt --files-per-task 5 --ntasks 20

  # Pipeline mode
  %(prog)s --pipeline pipelines/icarus_production.yaml

  # Dry run
  %(prog)s --config infer/icarus/icarus_full_chain_co_250625.yaml --source test.root --dry-run
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--config", "-c", help="SPINE configuration file")
    mode_group.add_argument("--pipeline", "-P", help="Pipeline YAML file")
    mode_group.add_argument(
        "--list-mods",
        metavar="CONFIG",
        help="List available modifiers for a configuration",
    )

    # Input files (required for --config mode) - mutually exclusive
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source",
        "-s",
        nargs="+",
        help="Input files as direct paths or glob patterns (e.g., data/*.root)",
    )
    source_group.add_argument(
        "--source-list",
        "-S",
        nargs=1,
        help="Text file containing input file paths (one per line)",
    )

    # Configuration modifiers
    parser.add_argument(
        "--apply-mods",
        "-m",
        nargs="+",
        help='Apply modifier configs (e.g., "data flash" to compose with data_mod and flash_mod)',
    )

    # Resource configuration
    parser.add_argument(
        "--profile",
        "-p",
        default="auto",
        help="Resource profile (default: auto-detect)",
    )
    parser.add_argument(
        "--ntasks", "-n", type=int, help="Number of parallel tasks (default: all)"
    )
    parser.add_argument(
        "--files-per-task", type=int, default=1, help="Files per task (default: 1)"
    )

    # Job configuration
    parser.add_argument("--job-name", "-j", help="Custom job name")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--account", "-A", help="SLURM account")
    parser.add_argument("--dependency", "-d", help="SLURM dependency string")
    parser.add_argument(
        "--local-output",
        action="store_true",
        help="Write job directories and logs to the current working directory "
        "instead of the default jobs directory in the spine-prod base.",
    )

    # Software paths
    parser.add_argument(
        "--larcv", "-l", dest="larcv_basedir", help="Custom LArCV installation path"
    )
    parser.add_argument(
        "--flashmatch", "-F", action="store_true", help="Enable flash matching"
    )

    # Profile overrides
    parser.add_argument("--partition", help="Override partition")
    parser.add_argument("--gpus", type=int, help="Override GPU count")
    parser.add_argument("--cpus-per-task", type=int, help="Override CPUs per task")
    parser.add_argument("--mem-per-cpu", help="Override memory per CPU")
    parser.add_argument("--time", "-t", help="Override time limit")

    # Execution
    parser.add_argument(
        "--interactive",
        "-I",
        action="store_true",
        help="Run interactively without submitting to SLURM (useful for testing)",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=1,
        help="Task ID to run in interactive mode (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without submitting",
    )

    args = parser.parse_args()

    # Initialize submitter
    submitter = SlurmSubmitter(local_output=getattr(args, "local_output", False))

    # Handle --list-mods
    if args.list_mods:
        try:
            result = submitter.list_modifiers(args.list_mods)

            print(
                f"Available modifiers for {result['config_name']} "
                "(version: {result['base_version'] or 'unversioned'}):"
            )
            if result["modifiers"]:
                for mod_name, mod_info in result["modifiers"].items():
                    versions_str = ", ".join(mod_info["available"])
                    print(
                        f"  {mod_name:12} -> {mod_info['selected']} (available: {versions_str})"
                    )

                mod_names = list(result["modifiers"].keys())
                print("\nUsage examples:")
                print(f"  --apply-mods {mod_names[0]}")
                print(f"  --apply-mods {mod_names[0]}:240719")
                print("  --apply-mods /path/to/custom_mod.yaml")
                if len(mod_names) > 1:
                    print(f"  --apply-mods {' '.join(mod_names[:2])}")
            else:
                config_path = Path(args.list_mods)
                print("  (none found)")
                print(
                    "\nNote: Modifiers should be in "
                    f"{config_path.parent.name}/modifier/ subdirectories or named "
                    f"{config_path.parent.name}_*_mod.yaml"
                )
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        return 0

    # Validate arguments
    if args.config and not (args.source or args.source_list):
        parser.error("--config requires either --source/-s or --source-list/-S")

    # Interactive and dry-run are mutually exclusive
    if args.interactive and args.dry_run:
        parser.error("--interactive and --dry-run are mutually exclusive")

    # Interactive mode not supported for pipelines
    if args.interactive and args.pipeline:
        parser.error("--interactive mode is not supported with --pipeline")

    # Build profile overrides
    profile_overrides = {}
    for key in ["partition", "gpus", "cpus_per_task", "mem_per_cpu", "time", "account"]:
        value = getattr(args, key, None)
        if value is not None:
            profile_overrides[key] = value

    try:
        if args.pipeline:
            # Pipeline mode
            job_map = submitter.submit_pipeline(args.pipeline, dry_run=args.dry_run)
            print("\n=== Pipeline submitted ===")
            for stage, job_ids in job_map.items():
                print(f"{stage}: {', '.join(job_ids)}")
        elif args.interactive:
            # Interactive mode - run directly without SLURM
            files = args.source if args.source else args.source_list
            source_type = "source" if args.source else "source_list"

            exit_code = submitter.run_interactive(
                config=args.config,
                files=files,
                source_type=source_type,
                output=args.output,
                files_per_task=args.files_per_task,
                task_id=args.task_id,
                larcv_basedir=args.larcv_basedir,
                flashmatch=args.flashmatch,
                apply_mods=args.apply_mods,
            )
            return exit_code
        else:
            # Single job mode (batch submission)
            # Determine which source type was provided
            files = args.source if args.source else args.source_list
            source_type = "source" if args.source else "source_list"

            job_ids = submitter.submit_job(
                config=args.config,
                files=files,
                source_type=source_type,
                profile=args.profile,
                job_name=args.job_name,
                output=args.output,
                ntasks=args.ntasks,
                files_per_task=args.files_per_task,
                dependency=args.dependency,
                larcv_basedir=args.larcv_basedir,
                flashmatch=args.flashmatch,
                apply_mods=args.apply_mods,
                dry_run=args.dry_run,
                **profile_overrides,
            )

            if job_ids and not args.dry_run:
                print(f"\n=== Submitted job IDs: {', '.join(job_ids)} ===")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
