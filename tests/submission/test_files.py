"""Submission integration tests for files behavior."""

import pytest


class TestFileHandling:
    """Tests for file parsing and handling."""

    def test_parse_files_single_file(self, mock_submitter, tmp_path):
        """Test parsing single file."""
        test_file = tmp_path / "test.root"
        test_file.touch()

        files = mock_submitter.file_handler.parse_files([str(test_file)])
        assert len(files) == 1
        assert files[0] == str(test_file)

    def test_parse_files_glob_pattern(self, mock_submitter, tmp_path):
        """Test parsing glob patterns."""
        # Create test files
        for i in range(3):
            (tmp_path / f"data_{i}.root").touch()

        pattern = str(tmp_path / "data_*.root")
        files = mock_submitter.file_handler.parse_files([pattern])
        assert len(files) == 3
        assert all(f.endswith(".root") for f in files)

    def test_parse_files_from_list(self, mock_submitter, tmp_path):
        """Test parsing files from a source list file."""
        # Create test files
        test_files = []
        for i in range(3):
            f = tmp_path / f"file_{i}.root"
            f.touch()
            test_files.append(str(f))

        # Create file list
        file_list = tmp_path / "files.txt"
        file_list.write_text("\n".join(test_files))

        files = mock_submitter.file_handler.parse_files(
            [str(file_list)], source_type="source_list"
        )
        assert len(files) == 3
        assert all(f.endswith(".root") for f in files)

    def test_parse_files_direct_paths(self, mock_submitter, tmp_path):
        """Test parsing direct file paths."""
        # Create test files
        files_to_create = []
        for i in range(3):
            f = tmp_path / f"file_{i}.root"
            f.touch()
            files_to_create.append(str(f))

        files = mock_submitter.file_handler.parse_files(
            files_to_create, source_type="source"
        )
        assert len(files) == 3
        assert all(f.endswith(".root") for f in files)


class TestFileChunking:
    """Tests for file chunking logic."""

    def test_chunk_files_basic(self, mock_submitter):
        """Test basic file chunking."""
        files = [f"file_{i}.root" for i in range(10)]

        chunks = mock_submitter.file_handler.chunk_files(
            files, max_array_size=99, files_per_task=2
        )
        # 10 files / 2 per task = 5 groups, all fit in one chunk
        assert len(chunks) == 1
        assert len(chunks[0]) == 5  # 5 groups

    def test_chunk_files_multiple_chunks(self, mock_submitter):
        """Test chunking with array size limit."""
        # Create enough files to exceed max_array_size
        files = [f"file_{i}.root" for i in range(50)]

        chunks = mock_submitter.file_handler.chunk_files(
            files, max_array_size=10, files_per_task=1
        )
        # 50 files / 1 per task = 50 groups, split into chunks of 10
        assert len(chunks) == 5
        assert all(len(chunk) <= 10 for chunk in chunks)

    def test_chunk_files_per_task(self, mock_submitter):
        """Test multiple files per task."""
        files = [f"file_{i}.root" for i in range(9)]

        chunks = mock_submitter.file_handler.chunk_files(
            files, max_array_size=99, files_per_task=3
        )
        # 9 files / 3 per task = 3 groups
        assert len(chunks) == 1
        assert len(chunks[0]) == 3
        # Each group should have 3 files as a list
        assert len(chunks[0][0]) == 3
        assert isinstance(chunks[0][0], list)

    def test_resolve_files_per_task_defaults_to_all_files(self, mock_submitter):
        """Test omitted splitting flags collapse explicit inputs into one task."""
        assert mock_submitter.batch.resolve_files_per_task(9) == 9

    def test_resolve_files_per_task_uses_ntasks_for_even_split(self, mock_submitter):
        """Test ntasks alone distributes files roughly evenly across tasks."""
        assert mock_submitter.batch.resolve_files_per_task(10, ntasks=3) == 4

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"ntasks": 0}, "--ntasks must be >= 1"),
            ({"files_per_task": 0}, "--files-per-task must be >= 1"),
        ],
    )
    def test_resolve_files_per_task_rejects_nonpositive_values(
        self, mock_submitter, kwargs, message
    ):
        with pytest.raises(ValueError, match=message):
            mock_submitter.batch.resolve_files_per_task(10, **kwargs)

    def test_parse_named_sources_preserves_aligned_file_order(
        self, mock_submitter, tmp_path
    ):
        """Composite source lists resolve into aligned target sequences."""
        larcv = [tmp_path / f"input_{index}.root" for index in range(2)]
        hdf5 = [tmp_path / f"input_{index}_cache.h5" for index in range(2)]
        for path in [*larcv, *hdf5]:
            path.touch()
        larcv_list = tmp_path / "larcv.txt"
        hdf5_list = tmp_path / "hdf5.txt"
        larcv_list.write_text("\n".join(map(str, larcv)), encoding="utf-8")
        hdf5_list.write_text("\n".join(map(str, hdf5)), encoding="utf-8")

        resolved = mock_submitter.file_handler.parse_named_sources(
            {
                "larcv": {"source_list": str(larcv_list)},
                "hdf5": {"source_list": str(hdf5_list)},
            }
        )
        assert resolved == {
            "larcv": [str(path) for path in larcv],
            "hdf5": [str(path) for path in hdf5],
        }

    @pytest.mark.parametrize(
        ("sources", "message"),
        [
            ({"larcv": {}}, "must specify exactly one"),
            ({"larcv": {"source": []}}, "contains no input files"),
        ],
    )
    def test_parse_named_sources_rejects_invalid_targets(
        self, mock_submitter, sources, message
    ):
        with pytest.raises(ValueError, match=message):
            mock_submitter.file_handler.parse_named_sources(sources)

    def test_parse_named_sources_rejects_unaligned_counts(
        self, mock_submitter, tmp_path
    ):
        """Every mixed source target must contribute one file per pair."""
        paths = [tmp_path / f"input_{index}.root" for index in range(2)]
        for path in paths:
            path.touch()
        with pytest.raises(ValueError, match="aligned file counts"):
            mock_submitter.file_handler.parse_named_sources(
                {
                    "larcv": {"source": [str(path) for path in paths]},
                    "hdf5": {"source": str(paths[0])},
                }
            )
