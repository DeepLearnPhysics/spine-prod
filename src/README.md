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
│   ├── config_manager.py   # Configuration and profile management
│   ├── file_handler.py     # File parsing and chunking
│   ├── preload.py          # SPINE download preloading
│   └── submitter.py        # Main orchestration class
├── templates/              # Batch job templates
├── config/                 # SPINE configurations
└── jobs/                   # Job output directories
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
- Main `Submitter` class
- Orchestrates all components
- Handles job submission workflow
- Pipeline management
- Interactive execution mode

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
