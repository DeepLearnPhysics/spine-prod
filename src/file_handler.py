"""File handling and chunking for SPINE production."""

import glob
import os
from typing import List


class FileHandler:
    """Handles file parsing, validation, and chunking."""

    def parse_files(
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
