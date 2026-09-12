"""Checkpoint state management with atomic disk writes for Milestone 3."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict, field


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""
    pass


class CheckpointStatus:
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    UPLOADING = "UPLOADING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    ALL = {NOT_STARTED, RUNNING, UPLOADING, PROCESSING, COMPLETED, FAILED}


@dataclass
class CheckpointData:
    """Represents the schema of state/checkpoint.json."""
    job_id: str | None = None
    session_id: str | None = None
    current_block_id: str | None = None
    last_confirmed_source_index: int | None = None
    next_source_index: int | None = None
    status: str = CheckpointStatus.NOT_STARTED
    file_name: str | None = None
    file_id: str | None = None
    total_source_segments: int | None = None
    transcript_file: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    error_message: str | None = None
    confirmed_segments: list[dict[str, Any]] = field(default_factory=list)
    block_metrics: list[dict[str, Any]] = field(default_factory=list)  # Milestone 3.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointData":
        allowed_fields = {
            "job_id",
            "session_id",
            "current_block_id",
            "last_confirmed_source_index",
            "next_source_index",
            "status",
            "file_name",
            "file_id",
            "total_source_segments",
            "transcript_file",
            "created_at",
            "updated_at",
            "error_message",
            "confirmed_segments",
            "block_metrics",  # Milestone 3.1
        }
        filtered = {k: v for k, v in data.items() if k in allowed_fields}
        # Compute next_source_index if missing
        if filtered.get("last_confirmed_source_index") is not None and filtered.get("next_source_index") is None:
            filtered["next_source_index"] = filtered["last_confirmed_source_index"] + 1
        return cls(**filtered)


def generate_job_id(file_name: str) -> str:
    """Generate a unique job ID based on current UTC timestamp and file base name."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sanitized_name = Path(file_name).stem.replace(" ", "_").lower()
    return f"{timestamp}_{sanitized_name}"


def generate_session_id() -> str:
    """Generate a session ID for the current execution."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"SESSION_{timestamp}"


class CheckpointManager:
    """Manages reading, updating, and atomically saving checkpoint data."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.current_data: CheckpointData = CheckpointData()

    def create_new_job(
        self,
        file_name: str,
        total_source_segments: int | None = None,
        job_id: str | None = None,
        session_id: str | None = None,
        status: str = CheckpointStatus.NOT_STARTED,
    ) -> CheckpointData:
        """Create a new job state for a given audio file."""
        now_iso = datetime.now(timezone.utc).isoformat()
        job_id = job_id or generate_job_id(file_name)
        session_id = session_id or generate_session_id()
        self.current_data = CheckpointData(
            job_id=job_id,
            session_id=session_id,
            current_block_id=None,
            last_confirmed_source_index=None,
            next_source_index=1,
            status=status,
            file_name=file_name,
            file_id=None,
            total_source_segments=total_source_segments,
            transcript_file=None,
            created_at=now_iso,
            updated_at=now_iso,
            error_message=None,
            confirmed_segments=[],
        )
        self.save()
        return self.current_data

    def load(self) -> CheckpointData:
        """Load checkpoint from disk if it exists, otherwise return default state."""
        if not self.checkpoint_path.exists():
            self.current_data = CheckpointData()
            return self.current_data

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.current_data = CheckpointData.from_dict(data)
            return self.current_data
        except Exception as exc:
            raise CheckpointError(f"Failed to read checkpoint from {self.checkpoint_path}: {exc}") from exc

    def commit_block(
        self,
        block_id: str,
        last_source_index: int,
        new_segments: list[dict[str, Any]],
        file_id: str | None = None,
    ) -> CheckpointData:
        """Atomic Checkpoint Commit: Only called after validation and merge PASS."""
        now_iso = datetime.now(timezone.utc).isoformat()
        self.current_data.current_block_id = block_id
        self.current_data.last_confirmed_source_index = last_source_index
        self.current_data.next_source_index = last_source_index + 1
        self.current_data.status = CheckpointStatus.RUNNING
        self.current_data.updated_at = now_iso
        if file_id is not None:
            self.current_data.file_id = str(file_id)

        # Idempotently merge confirmed segments
        existing_indices = {s["source_index"] for s in self.current_data.confirmed_segments}
        for seg in new_segments:
            if seg["source_index"] in existing_indices:
                # Update existing
                for i, existing in enumerate(self.current_data.confirmed_segments):
                    if existing["source_index"] == seg["source_index"]:
                        self.current_data.confirmed_segments[i] = seg
                        break
            else:
                self.current_data.confirmed_segments.append(seg)
                existing_indices.add(seg["source_index"])

        # Sort segments strictly by source_index
        self.current_data.confirmed_segments.sort(key=lambda s: s["source_index"])

        self.save()
        return self.current_data

    def update_status(
        self,
        status: str,
        file_id: str | None = None,
        transcript_file: str | None = None,
        error_message: str | None = None,
        last_confirmed_source_index: int | None = None,
        total_source_segments: int | None = None,
    ) -> CheckpointData:
        """Update job status and metadata, then atomically persist to disk."""
        if status not in CheckpointStatus.ALL:
            raise ValueError(f"Invalid status '{status}'. Must be one of {CheckpointStatus.ALL}")

        self.current_data.status = status
        self.current_data.updated_at = datetime.now(timezone.utc).isoformat()

        if file_id is not None:
            self.current_data.file_id = file_id

        if transcript_file is not None:
            self.current_data.transcript_file = transcript_file

        if error_message is not None:
            self.current_data.error_message = error_message

        if last_confirmed_source_index is not None:
            self.current_data.last_confirmed_source_index = last_confirmed_source_index
            self.current_data.next_source_index = last_confirmed_source_index + 1

        if total_source_segments is not None:
            self.current_data.total_source_segments = total_source_segments

        self.save()
        return self.current_data

    def save(self) -> None:
        """Atomically save checkpoint to disk using a temporary file."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.checkpoint_path.with_suffix(".tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.current_data.to_dict(), f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            # Atomic replace
            temp_path.replace(self.checkpoint_path)
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise CheckpointError(f"Failed to atomically save checkpoint: {exc}") from exc

    def is_completed_for(self, file_name: str, transcript_path: Path) -> bool:
        """Check idempotency: return True if a job for this file already completed and transcript exists."""
        if (
            self.current_data.file_name == file_name
            and self.current_data.status == CheckpointStatus.COMPLETED
            and transcript_path.exists()
            and transcript_path.stat().st_size > 0
        ):
            return True
        return False
