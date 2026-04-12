# SPINE Production - Code Organization

## Overview

The SPINE production submission system has been refactored into a modular structure for better maintainability and readability. The original 1,500+ line monolithic `submit.py` has been reorganized into focused modules.

## Structure

```
spine-prod/
├── submit.py                # Main CLI entry point (295 lines)
├── src/                     # Source code modules
│   ├── __init__.py         # Package initialization
│   ├── client/             # Batch scheduler clients
│   │   ├── base.py         # Shared template/job metadata helpers
│   │   ├── slurm.py        # SLURM sbatch client
│   │   └── pbs.py          # PBS qsub client
│   ├── config_manager.py   # Configuration and profile management (560 lines)
│   ├── file_handler.py     # File parsing and chunking (81 lines)
│   └── submitter.py        # Main orchestration class (522 lines)
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

No changes to user-facing commands - all existing scripts and workflows continue to work:

```bash
# Basic usage (unchanged)
./submit.py --config infer/icarus/latest.yaml --source-list files.txt

# All features still available
./submit.py --pipeline pipelines/production.yaml
./submit.py --interactive --config ... --source test.root
./submit.py --list-mods infer/icarus/config.yaml
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
