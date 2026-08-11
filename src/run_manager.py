"""Run-directory and training lifecycle helpers."""

import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class RunManager:
    """Manage persistent training runs and their stage submissions."""

    METADATA_NAME = "run_metadata.json"
    CHECKPOINT_RE = re.compile(r"^snapshot-(\d+)\.ckpt$")

    @staticmethod
    def config_digest(config: str) -> str:
        """Return a stable digest for a resolved configuration file."""
        return hashlib.sha256(Path(config).read_bytes()).hexdigest()

    @staticmethod
    def checkpoint_digest(checkpoint: Path) -> Optional[str]:
        """Verify a checkpoint sidecar when present and return its digest.

        SPINE versions before v0.17.0 did not create checksum sidecars, so a
        missing sidecar intentionally leaves the checkpoint unverified rather
        than making it unusable.
        """
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        sidecar = Path(f"{checkpoint}.sha256")
        if not sidecar.exists():
            return None

        fields = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"Malformed checkpoint checksum sidecar: {sidecar}")
        expected, filename = fields
        filename = filename.lstrip(" *")
        if filename != checkpoint.name:
            raise ValueError(
                f"Checkpoint checksum sidecar names '{filename}', "
                f"expected '{checkpoint.name}': {sidecar}"
            )

        actual = hashlib.sha256()
        with open(checkpoint, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                actual.update(chunk)
        digest = actual.hexdigest()
        if digest != expected:
            raise ValueError(f"Checkpoint checksum does not match: {checkpoint}")
        return digest

    @staticmethod
    def _read_json(path: Path) -> Dict:
        """Read a JSON object from disk."""
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def _write_json(path: Path, payload: Dict) -> None:
        """Write a JSON object to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)

    @classmethod
    def prepare_training_run(
        cls,
        run_dir: Path,
        config: str,
        resume: bool = False,
        resume_from: Optional[str] = None,
    ) -> Optional[Path]:
        """Create a training run or resolve the checkpoint used to resume it."""
        metadata_path = run_dir / cls.METADATA_NAME
        wants_resume = resume or resume_from is not None

        if wants_resume:
            if not metadata_path.is_file():
                raise ValueError(f"Cannot resume: no training run found at {run_dir}")
            metadata = cls._read_json(metadata_path)
            digest = cls.config_digest(config)
            if metadata.get("training_config_sha256") != digest:
                raise ValueError(
                    "Cannot resume with a different training configuration. "
                    "Use runtime --set overrides for intentional changes such as "
                    "extending the total epoch count."
                )
            checkpoint = (
                cls.resolve_resume_checkpoint(run_dir, resume_from)
                if resume_from is not None
                else cls.latest_checkpoint(run_dir)
            )
            if checkpoint is None:
                raise ValueError(f"Cannot resume: no checkpoints found under {run_dir}")
            cls.checkpoint_digest(checkpoint)
            return checkpoint

        if metadata_path.exists():
            metadata = cls._read_json(metadata_path)
            if (
                metadata.get("training_config_sha256") == cls.config_digest(config)
                and not metadata.get("job_ids")
                and cls.latest_checkpoint(run_dir) is None
                and not any(run_dir.glob("train*_log-*.csv"))
            ):
                return None
            raise ValueError(
                f"Training run already exists at {run_dir}; pass --resume to continue it"
            )
        existing = (
            []
            if not run_dir.exists()
            else [path for path in run_dir.iterdir() if path.name != ".spine-prod"]
        )
        if existing:
            raise ValueError(f"New training run directory is not empty: {run_dir}")

        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "weights").mkdir(exist_ok=True)
        (run_dir / "tensorboard" / "train").mkdir(parents=True, exist_ok=True)
        (run_dir / "tensorboard" / "validation").mkdir(parents=True, exist_ok=True)
        cls._write_json(
            metadata_path,
            {
                "kind": "training",
                "created": datetime.now().isoformat(),
                "training_config": str(Path(config).resolve()),
                "training_config_sha256": cls.config_digest(config),
            },
        )
        return None

    @classmethod
    def record_training_jobs(cls, run_dir: Path, job_ids: List[str]) -> None:
        """Record successfully submitted scheduler jobs on a training run."""
        if not job_ids:
            return
        metadata_path = run_dir / cls.METADATA_NAME
        metadata = cls._read_json(metadata_path)
        metadata["job_ids"] = job_ids
        metadata["last_submitted"] = datetime.now().isoformat()
        cls._write_json(metadata_path, metadata)

    @classmethod
    def checkpoints(cls, run_dir: Path) -> List[Tuple[int, Path]]:
        """Return run checkpoints ordered by their numeric iteration suffix."""
        checkpoints = []
        for path in (run_dir / "weights").glob("snapshot-*.ckpt"):
            match = cls.CHECKPOINT_RE.match(path.name)
            if match:
                checkpoints.append((int(match.group(1)), path))
        return sorted(checkpoints)

    @classmethod
    def latest_checkpoint(cls, run_dir: Path) -> Optional[Path]:
        """Return the checkpoint with the greatest numeric iteration."""
        checkpoints = cls.checkpoints(run_dir)
        return checkpoints[-1][1] if checkpoints else None

    @classmethod
    def resolve_resume_checkpoint(cls, run_dir: Path, resume_from: str) -> Path:
        """Validate and return an explicit checkpoint within a training run."""
        checkpoint = Path(resume_from).resolve()
        weights_dir = (run_dir / "weights").resolve()
        if checkpoint.parent != weights_dir or not checkpoint.is_file():
            raise ValueError(
                "--resume-from must name an existing checkpoint in the run's "
                f"weights directory: {weights_dir}"
            )
        if cls.CHECKPOINT_RE.match(checkpoint.name) is None:
            raise ValueError(f"Invalid checkpoint name: {checkpoint.name}")
        return checkpoint

    @staticmethod
    def validation_log_dir(run_dir: Path, validation_name: Optional[str]) -> Path:
        """Return the CSV log directory for a validation suite."""
        if validation_name:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", validation_name):
                raise ValueError(
                    "--validation-name may contain only letters, numbers, '.', '_', and '-'"
                )
            return run_dir / "validation" / validation_name
        return run_dir

    @staticmethod
    def _validation_log_complete(path: Path) -> bool:
        """Return whether a validation CSV contains a header and at least one row."""
        try:
            with open(path, "r", encoding="utf-8", newline="") as stream:
                rows = csv.reader(stream)
                header = next(rows, None)
                row = next(rows, None)
            return bool(header and row)
        except (OSError, csv.Error):
            return False

    @classmethod
    def checkpoint_is_validated(cls, log_dir: Path, iteration: int) -> bool:
        """Return whether a complete validation log exists for a checkpoint."""
        start = iteration + 1
        pattern = f"inference*_log-{start:07d}.csv"
        return any(cls._validation_log_complete(path) for path in log_dir.glob(pattern))

    @classmethod
    def prepare_validation(
        cls,
        run_dir: Path,
        config: str,
        validation_name: Optional[str] = None,
        rerun: bool = False,
    ) -> Tuple[Path, List[Path]]:
        """Validate a training run and return the log directory and checkpoints."""
        metadata_path = run_dir / cls.METADATA_NAME
        if not metadata_path.is_file():
            raise ValueError(f"No training run found at {run_dir}")

        checkpoints = cls.checkpoints(run_dir)
        if not checkpoints:
            raise ValueError(f"No checkpoints found under {run_dir}")

        log_dir = cls.validation_log_dir(run_dir, validation_name)
        log_dir.mkdir(parents=True, exist_ok=True)
        suite_name = validation_name or "primary"
        state_path = run_dir / ".spine-prod" / "validation" / f"{suite_name}.json"
        digest = cls.config_digest(config)
        if state_path.exists():
            state = cls._read_json(state_path)
            if state.get("config_sha256") != digest and not rerun:
                raise ValueError(
                    f"Validation suite '{suite_name}' was created with a different "
                    "configuration; use another --validation-name or "
                    "--rerun-validation"
                )

        cls._write_json(
            state_path,
            {
                "name": suite_name,
                "config": str(Path(config).resolve()),
                "config_sha256": digest,
                "updated": datetime.now().isoformat(),
            },
        )
        selected = [
            path
            for iteration, path in checkpoints
            if rerun or not cls.checkpoint_is_validated(log_dir, iteration)
        ]
        for checkpoint in selected:
            cls.checkpoint_digest(checkpoint)
        return log_dir, selected

    @staticmethod
    def create_submission_dir(
        run_dir: Path, stage: str, name: Optional[str] = None
    ) -> Path:
        """Create a timestamped scheduler-artifact directory for a run stage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        parts = [run_dir, "submissions", stage]
        if name:
            parts.append(name)
        submission_dir = Path(*parts) / timestamp
        (submission_dir / "logs").mkdir(parents=True)
        return submission_dir
