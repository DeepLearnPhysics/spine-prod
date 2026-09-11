"""File handling and chunking for SPINE production."""

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping


class FileHandler:
    """Handles file parsing, validation, and chunking."""

    def parse_files(
        self,
        file_input: List[str],
        source_type: str = "source",
        allow_missing: bool = False,
    ) -> List[str]:
        """Parse file input (paths, globs, or txt file).

        Parameters
        ----------
        file_input : List[str]
            List of file paths, glob patterns, or text file path
        source_type : str, optional
            Either 'source' (direct paths/globs) or 'source_list' (text file),
            by default 'source'
        allow_missing : bool, optional
            Preserve missing direct paths or glob patterns that an upstream
            pipeline stage will create before this job starts. Source-list
            files must still exist when the submission is prepared.

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
                listed_files = [line.strip() for line in f if line.strip()]
            duplicate_paths = self._duplicates(listed_files)
            if duplicate_paths:
                examples = ", ".join(duplicate_paths[:3])
                raise ValueError(
                    f"Source list '{source_list_path}' contains repeated file "
                    f"paths ({len(duplicate_paths)} unique duplicate(s)); each "
                    f"source must appear exactly once. Examples: {examples}"
                )
            files.extend(listed_files)
        else:
            # Handle direct sources (paths, globs)
            for item in file_input:
                if "*" in item or "?" in item:
                    # Expand glob
                    matches = sorted(glob.glob(item))
                    if matches:
                        files.extend(matches)
                    elif allow_missing:
                        # The dependent job lets SPINE expand this after the
                        # scheduler has materialized the upstream files.
                        files.append(item)
                else:
                    # Direct file path
                    if os.path.exists(item) or allow_missing:
                        files.append(item)
                    else:
                        print(f"WARNING: File not found: {item}")

        return files

    def parse_named_sources(
        self,
        sources: Mapping[str, Mapping[str, Any]],
        allow_missing: bool = False,
    ) -> Dict[str, List[str]]:
        """Resolve every target in a composite dataset source mapping.

        Ordinary targets must resolve to the same number of files. The
        canonical ``cache`` role is one shared repository and may resolve to
        one path while the ``primary`` source is partitioned across tasks.
        """
        resolved = {}
        expected_count = None
        for target, source_cfg in sources.items():
            selectors = [key for key in ("source", "source_list") if key in source_cfg]
            if len(selectors) != 1:
                raise ValueError(
                    f"Named source '{target}' must specify exactly one of: "
                    "source, source_list"
                )
            selector = selectors[0]
            values = source_cfg[selector]
            if not isinstance(values, list):
                values = [values]
            files = self.parse_files(
                values,
                selector,
                allow_missing=allow_missing,
            )
            if not files:
                raise ValueError(f"Named source '{target}' contains no input files")
            if target == "cache":
                if len(files) != 1:
                    raise ValueError(
                        "Named source 'cache' requires one repository path"
                    )
            elif expected_count is None:
                expected_count = len(files)
            elif len(files) != expected_count:
                raise ValueError(
                    "Named sources must contain aligned file counts; "
                    f"'{target}' has {len(files)}, expected {expected_count}"
                )
            resolved[target] = files

        if expected_count is None:
            raise ValueError(
                "Named cache sources require another target to drive task splitting"
            )

        return resolved

    @staticmethod
    def stage_cache_output_paths(
        source_files: List[str], output_dir: str, suffix: str
    ) -> List[str]:
        """Predict source-routed stage-cache paths for an output manifest."""
        directory = Path(output_dir)
        paths = [
            str(directory / f"{Path(source).stem}_{suffix}.h5")
            for source in source_files
        ]
        duplicate_paths = FileHandler._duplicates(paths)
        if duplicate_paths:
            colliding_names = ", ".join(Path(path).name for path in duplicate_paths[:3])
            raise ValueError(
                "Stage-cache output names collide because distinct source paths "
                "share a basename. SPINE split-output caches currently require "
                "globally unique source basenames. Colliding output example(s): "
                f"{colliding_names}"
            )
        return paths

    @staticmethod
    def _duplicates(values: List[str]) -> List[str]:
        """Return repeated values once, preserving first duplicate order."""
        seen = set()
        repeated = set()
        duplicates = []
        for value in values:
            if value in seen and value not in repeated:
                duplicates.append(value)
                repeated.add(value)
            seen.add(value)
        return duplicates

    def chunk_files(
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
            List of file chunks, each chunk is a list of file groups,
            where each group contains files_per_task individual files
        """
        # Group files by files_per_task
        file_groups = []
        for i in range(0, len(files), files_per_task):
            group = files[i : i + files_per_task]
            file_groups.append(group)

        # Split into chunks that fit array size limit
        chunks = []
        for i in range(0, len(file_groups), max_array_size):
            chunks.append(file_groups[i : i + max_array_size])

        return chunks
