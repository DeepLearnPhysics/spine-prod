# SPINE Production - Code Organization

## Overview

The SPINE production submission system uses a thin command-line entry point backed by focused modules for configuration management, file handling, scheduler integration, preloading, and job orchestration.

## Structure

```
spine-prod/
├── submit.py                # Main CLI entry point
├── src/                     # Source code modules
│   ├── __init__.py         # Package initialization
│   ├── client/             # Batch scheduler clients
│   │   ├── base.py         # Shared template/job metadata helpers
│   │   ├── slurm.py        # SLURM sbatch client
│   │   └── pbs.py          # PBS qsub client
│   ├── batch.py            # Single-job planning and scheduler submission
│   ├── component.py        # Shared component-to-façade contract
│   ├── config_manager.py   # Configuration and profile management
│   ├── file_handler.py     # File parsing and chunking
│   ├── interactive.py      # Direct local/container execution workflow
│   ├── pipeline.py         # Pipeline loading, validation, and execution
│   ├── preload.py          # SPINE download preloading
│   ├── runtime.py          # Executable, setup, container, and bind resolution
│   ├── run_manager.py      # Persistent training and validation lifecycle
│   ├── spine_cli.py        # Pure SPINE CLI argument formatting
│   └── submitter.py        # Thin public façade and component wiring
├── templates/              # Batch job templates
├── config/                 # SPINE configurations
└── runs/                   # Automatic inference run directories
```

## Module Responsibilities

### `submit.py`
- Command-line argument parsing
- Main entry point
- Minimal orchestration logic

### `src/config_manager.py`
- Load and manage profiles from YAML
- Detector auto-detection
- Modifier discovery and version resolution
- Composite config generation
- "Latest" config assembly

### `src/file_handler.py`
- Parse file inputs (globs, lists, direct paths)
- Chunk files for array jobs
- File validation

### `src/client/`
- Load batch job templates
- Submit jobs via scheduler-specific clients
- Create job directories
- Save job metadata
- Parse scheduler-specific job IDs
- Cleanup job management for SLURM

### `src/submitter.py`
- Stable public `Submitter` API
- Shared manager and scheduler-client construction
- Focused component wiring and compatibility forwarding

### `src/batch.py`
- Single-job input and lifecycle validation
- Scheduler profile and array-job planning
- Template rendering, submission, and metadata recording

### `src/interactive.py`
- Interactive input preparation
- Direct local or container-backed SPINE execution

### `src/pipeline.py`
- Whole-document loading and validation before scheduler interaction
- Default/stage/CLI precedence resolution
- Dependency-ordered stage and cleanup submission

### `src/spine_cli.py`
- Pure formatting and validation of SPINE command-line options
- GPU allocation and world-size alignment

### `src/runtime.py`
- Software setup and executable discovery
- Container selection and bind-path construction

### `src/run_manager.py`
- Create and validate persistent training runs
- Resolve numeric resume checkpoints
- Plan incremental and named validation suites
- Create immutable attempt directories and stable latest/log links

## Benefits

1. **Readability**: Each module has a clear, focused purpose
2. **Maintainability**: Changes are easier to locate and implement
3. **Testability**: Individual components can be tested in isolation
4. **Extensibility**: New features can be added without modifying unrelated code
5. **Reusability**: Components can be imported and used programmatically

## Usage

```bash
# Basic usage
./submit.py --config infer/icarus/latest --source-list files.txt

# Additional modes
./submit.py --pipeline pipelines/icarus_production_example.yaml
./submit.py --interactive --config ... --source test.root
./submit.py --list-mods infer/icarus/full_chain_co_260501.yaml
```

## Imports for Programmatic Use

The refactored code can now be easily used programmatically:

```python
from src import Submitter
from src.config_manager import ConfigManager
from src.file_handler import FileHandler
from src.client import PBSClient, SlurmClient

# Example: Use components directly
submitter = Submitter()
job_ids = submitter.submit_job(config="...", files=["..."])
```
