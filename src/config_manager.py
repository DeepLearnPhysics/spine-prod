"""Configuration and profile management for SPINE production."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class ConfigManager:
    """Manages configuration files, profiles, and modifiers."""

    def __init__(self, basedir: Path):
        """Initialize ConfigManager.

        Parameters
        ----------
        basedir : Path
            Base directory of spine-prod
        """
        self.basedir = basedir
        self.profiles = self._load_profiles()

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

    def detect_detector(self, config: str) -> str:
        """Auto-detect detector from config path.

        Parameters
        ----------
        config : str
            Path to configuration file

        Returns
        -------
        str
            Detector name or 'unknown_detector'
        """
        config_path = Path(config)
        for detector in self.profiles["detectors"]:
            if detector in str(config_path):
                return detector
        return "unknown_detector"

    def get_profile(self, profile_name: str, detector: Optional[str] = None) -> Dict:
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
            Profile configuration dictionary

        Raises
        ------
        ValueError
            If profile not found
        """
        if profile_name == "auto":
            # Auto-select based on detector
            if detector and detector in self.profiles["detectors"]:
                profile_name = self.profiles["detectors"][detector].get(
                    "default_profile", "s3df_ampere"  # fallback to s3df_ampere
                )
            else:
                # Fallback to s3df_ampere if no default_profile in defaults
                profile_name = self.profiles["defaults"].get(
                    "default_profile", "s3df_ampere"
                )

        # Get profile
        if profile_name not in self.profiles["profiles"]:
            available = ", ".join(self.profiles["profiles"].keys())
            raise ValueError(f"Unknown profile: {profile_name}. Available: {available}")

        # Start with defaults, then overlay profile-specific settings
        profile_config = self.profiles["defaults"].copy()
        profile_config.update(self.profiles["profiles"][profile_name])

        return profile_config

    def discover_modifiers(self, config: str) -> Dict[str, List[Path]]:
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

    def extract_version(self, config_path: Path) -> Optional[str]:
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

    def resolve_modifier_version(
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
                f"Available: {[self.extract_version(p) or p.stem for p in available_versions]}"
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
                if (self.extract_version(v) or "") <= base_version
            ]
            if earlier_versions:
                selected = earlier_versions[-1]  # Latest of the earlier versions
                selected_ver = self.extract_version(selected) or selected.stem
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
        selected_ver = self.extract_version(selected) or selected.stem
        print(
            f"  {mod_name}: No version in base config, using latest version {selected_ver}"
        )
        return selected

    def create_composite_config(
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
        base_version = self.extract_version(config_path)

        # Always use the detector's config directory for modifier discovery if detector is known
        if detector:
            modifier_search_path = str(self.basedir / "config" / "infer" / detector)
        else:
            # Try to infer detector from base_config path if possible
            config_path_parts = Path(base_config).parts
            try:
                infer_idx = config_path_parts.index("infer")
                detector_guess = config_path_parts[infer_idx + 1]
                modifier_search_path = str(
                    self.basedir / "config" / "infer" / detector_guess
                )
            except (ValueError, IndexError):
                modifier_search_path = base_config

        available_mods = self.discover_modifiers(modifier_search_path)

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
            selected_path = self.resolve_modifier_version(
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

        # Make all paths relative to the config directory (SPINE_CONFIG_PATH)
        # Find the config directory (should be .../config)
        config_dir = self.basedir / "config"

        composite_content += "include:\n"

        # If the base config is a 'latest' (i.e., built on the fly, not in config_dir),
        # use absolute path. Otherwise, keep as relative to config_dir
        # If the base config exists under SPINE_CONFIG_PATH, use as given (relative path)
        # Otherwise (e.g. latest built on the fly), use absolute path
        config_path_in_config = (
            config_dir / os.path.relpath(config_path, self.basedir)
        ).resolve()
        if config_path_in_config.exists():
            composite_content += f"  - {os.path.relpath(config_path, self.basedir)}\n"
        else:
            # For 'latest' (built) configs, use a relative path from the composite file's directory
            rel_to_composite = os.path.relpath(config_path, composite_path.parent)
            composite_content += f"  - {rel_to_composite}\n"

        for mod_path in resolved_mods:
            try:
                rel_mod = os.path.relpath(mod_path, config_dir)
                composite_content += f"  - {rel_mod}\n"
            except ValueError:
                composite_content += f"  - {mod_path}\n"

        # If this is a 'latest' composite (i.e., base config is a built composite),
        # rewrite all includes in the base composite to be relative to SPINE_CONFIG_PATH
        if base_name.endswith("_composite") or "latest" in base_name:
            # Patch the just-written base composite file to rewrite its includes
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                with open(config_path, "w", encoding="utf-8") as f:
                    for line in lines:
                        if line.strip().startswith("- "):
                            inc_path = line.strip()[2:]
                            # If absolute or relative, always rewrite as relative to config_dir
                            inc_path_abs = (
                                (Path(config_path).parent / inc_path).resolve()
                                if not os.path.isabs(inc_path)
                                else Path(inc_path)
                            )
                            try:
                                rel_inc = os.path.relpath(inc_path_abs, config_dir)
                                f.write(f"  - {rel_inc}\n")
                            except (ValueError, OSError):
                                f.write(line)
                        else:
                            f.write(line)

            except (OSError, IOError) as e:
                print(f"WARNING: Could not rewrite includes in base composite: {e}")

        # Write the composite config
        with open(composite_path, "w", encoding="utf-8") as f:
            f.write(composite_content)

        print(f"Created composite config: {composite_path}")
        print(f"  Base: {Path(base_config).name}")
        print(f"  Modifiers: {', '.join(modifiers)}")

        return str(composite_path)

    def create_latest_config(self, detector: str, job_dir: Path) -> str:
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
                    yaml_files, key=lambda p: self.extract_version(p) or ""
                )[-1]
                latest_components[component] = latest
                version = self.extract_version(latest)
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
                # Use path relative to SPINE_CONFIG_PATH (config dir)
                # This allows SPINE to resolve includes via SPINE_CONFIG_PATH
                try:
                    rel_path = os.path.relpath(comp_path, self.basedir / "config")
                    composite_content += f"  - {rel_path}\n"
                except ValueError:
                    # Fallback to absolute path if on different drives/filesystems
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
        base_version = self.extract_version(config_path_obj)

        # Always use the detector's config directory for modifier discovery if possible
        config_path_parts = config_path_obj.parts
        modifier_search_path = config_path
        try:
            infer_idx = config_path_parts.index("infer")
            detector_guess = config_path_parts[infer_idx + 1]
            modifier_search_path = str(
                self.basedir / "config" / "infer" / detector_guess
            )
        except (ValueError, IndexError):
            pass

        modifiers = self.discover_modifiers(modifier_search_path)

        result = {
            "base_version": base_version,
            "config_name": config_path_obj.name,
            "modifiers": {},
        }

        for mod_name, version_paths in sorted(modifiers.items()):
            versions = [
                self.extract_version(p)
                or p.stem.replace("mod_", "").replace(f"{mod_name}_", "")
                for p in version_paths
            ]
            selected = self.resolve_modifier_version(
                mod_name, version_paths, base_version, None
            )
            selected_ver = self.extract_version(selected) or selected.stem

            result["modifiers"][mod_name] = {
                "selected": selected_ver,
                "available": versions,
                "paths": version_paths,
            }

        return result
