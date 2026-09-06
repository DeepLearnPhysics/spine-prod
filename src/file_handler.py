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
                files.extend([line.strip() for line in f if line.strip()])
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

        All targets must resolve to the same number of files. This lets an
        inference array preserve the one-to-one correspondence between, for
        example, a LArCV file and its stage-cache sidecar.
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
            if expected_count is None:
                expected_count = len(files)
            elif len(files) != expected_count:
                raise ValueError(
                    "Named sources must contain aligned file counts; "
                    f"'{target}' has {len(files)}, expected {expected_count}"
                )
            resolved[target] = files

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
        if len(paths) != len(set(paths)):
            raise ValueError(
                "Stage-cache output names collide; source basenames must be unique"
            )
        return paths

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
