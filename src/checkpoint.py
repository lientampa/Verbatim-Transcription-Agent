"""Checkpoint state management with atomic disk writes for Milestone 3."""

import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dataclasses import dataclass, asdict, field
from src.source_identity import SourceIdentity, DURATION_TOLERANCE_US, to_us


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
    schema_version: int = 1  # Legacy records remain readable, never trusted for resume.
    source_identity: dict[str, Any] | None = None
    next_audio_start_us: int = 0
    adaptive_state: dict[str, Any] = field(default_factory=dict)

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
            "schema_version", "source_identity", "next_audio_start_us", "adaptive_state",
        }
        filtered = {k: v for k, v in data.items() if k in allowed_fields}
        if data.get("schema_version") == 2:
            required = {"source_identity", "next_audio_start_us", "block_metrics", "confirmed_segments",
                        "next_source_index", "last_confirmed_source_index", "current_block_id"}
            if set(data) - allowed_fields or required - set(data):
                raise CheckpointError("CHECKPOINT_INVALID: unexpected or missing checkpoint fields")
        # Compute next_source_index if missing
        if filtered.get("schema_version", 1) == 1 and filtered.get("last_confirmed_source_index") is not None and filtered.get("next_source_index") is None:
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
        source_identity: SourceIdentity | None = None,
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
            schema_version=2 if source_identity else 1,
            source_identity=source_identity.to_dict() if source_identity else None,
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
            if self.current_data.schema_version == 2:
                self.validate_resume()
            return self.current_data
        except Exception as exc:
            raise CheckpointError(f"Failed to read checkpoint from {self.checkpoint_path}: {exc}") from exc

    def commit_block(
        self,
        block_id: str,
        last_source_index: int,
        new_segments: list[dict[str, Any]],
        file_id: str | None = None,
        metric: dict[str, Any] | None = None,
        adaptive_state: dict[str, Any] | None = None,
    ) -> CheckpointData:
        """Atomic Checkpoint Commit: Only called after validation and merge PASS."""
        previous = self.current_data
        self.current_data = deepcopy(previous)
        if metric is not None:
            self.current_data.block_metrics.append(dict(metric, last_source_index=last_source_index))
            self.current_data.next_audio_start_us = to_us(metric["actual_end_offset"])
        if adaptive_state is not None:
            self.current_data.adaptive_state = adaptive_state
        now_iso = datetime.now(timezone.utc).isoformat()
        self.current_data.current_block_id = block_id
        # AUDIO counters retain response metadata for compatibility only.
        # Resume progress is next_audio_start_us, never these model ordinals.
        self.current_data.last_confirmed_source_index = last_source_index
        self.current_data.next_source_index = last_source_index + 1
        self.current_data.status = CheckpointStatus.RUNNING
        self.current_data.updated_at = now_iso
        if file_id is not None:
            self.current_data.file_id = str(file_id)

        if metric and metric.get("source_mode") == "AUDIO":
            # Backend block identity scopes untouched model ordinals.
            self.current_data.confirmed_segments.extend(dict(seg, block_id=block_id) for seg in new_segments)
        else:
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

        try:
            if self.current_data.schema_version == 2:
                self.validate_resume()
            self.save()
        except Exception:
            self.current_data = previous
            raise
        return self.current_data

    def validate_resume(self, source: SourceIdentity | None = None) -> None:
        """Reject contradictions without repairing or rewriting checkpoint evidence."""
        data = self.current_data
        if data.schema_version != 2 or not data.source_identity:
            raise CheckpointError("CHECKPOINT_LEGACY_UNTRUSTED: use explicit --force for a new job")
        try:
            identity = data.source_identity
            if type(identity["size_bytes"]) is not int or identity["size_bytes"] < 0:
                raise ValueError("invalid source size")
            if not re.fullmatch(r"[0-9a-f]{64}", identity["fingerprint"]):
                raise ValueError("invalid fingerprint")
            duration = identity["duration_us"]
            if type(duration) is not int or duration <= 0 or type(data.next_audio_start_us) is not int:
                raise ValueError("invalid offset type")
            if source:
                if identity["fingerprint"] != source.fingerprint or identity["size_bytes"] != source.size_bytes:
                    raise CheckpointError("CHECKPOINT_SOURCE_MISMATCH")
                if abs(duration - source.duration_us) > DURATION_TOLERANCE_US:
                    raise ValueError("source duration mismatch")
            if data.status not in CheckpointStatus.ALL or not data.job_id or not data.session_id:
                raise ValueError("invalid job identity/status")
            if not 0 <= data.next_audio_start_us <= duration:
                raise ValueError("next audio start outside source")
            end, last_index, segment_offset = 0, 0, 0
            for ordinal, metric in enumerate(data.block_metrics, 1):
                start, current_end = to_us(metric["actual_start_offset"]), to_us(metric["actual_end_offset"])
                if start != end or not 0 <= start < current_end <= duration:
                    raise ValueError("noncontiguous or invalid confirmed boundaries")
                if "final_duration" in metric and to_us(metric["final_duration"]) != current_end - start:
                    raise ValueError("metric duration contradicts boundaries")
                if metric["block_id"] != f"BLOCK_{ordinal:03d}":
                    raise ValueError("confirmed block order mismatch")
                if metric["fidelity_decision"] not in ("ACCEPT", "ACCEPT_WITH_WARNING"):
                    raise ValueError("unaccepted metric")
                current_index = metric["last_source_index"]
                if type(current_index) is not int or current_index <= 0:
                    raise ValueError("invalid final ordinal")
                audio_mode = metric.get("source_mode") == "AUDIO"
                count = metric["segment_count"] if audio_mode else current_index - last_index
                if type(count) is not int or count <= 0:
                    raise ValueError("invalid segment count")
                segments = data.confirmed_segments[segment_offset:segment_offset + count]
                indices = [s["source_index"] for s in segments]
                if len(segments) != count or any(type(i) is not int or i <= 0 for i in indices):
                    raise ValueError("invalid segment ordinals")
                if audio_mode:
                    if any(s.get("block_id") != metric["block_id"] for s in segments):
                        raise ValueError("segment block identity mismatch")
                    if any(a >= b for a, b in zip(indices, indices[1:])) or indices[-1] != current_index:
                        raise ValueError("invalid local ordinal order/end")
                elif any(index != last_index + offset for offset, index in enumerate(indices, 1)):
                    raise ValueError("confirmed segment progress mismatch")
                segment_offset += count
                end, last_index = current_end, current_index
            if data.next_audio_start_us != end:
                raise ValueError("next start disagrees with confirmed end")
            if len(data.confirmed_segments) != segment_offset:
                raise ValueError("confirmed segment count mismatch")
            if data.last_confirmed_source_index != (last_index or None) or data.next_source_index != last_index + 1:
                raise ValueError("segment counters disagree")
            if data.current_block_id != (data.block_metrics[-1]["block_id"] if data.block_metrics else None):
                raise ValueError("current confirmed block mismatch")
            if data.status == CheckpointStatus.COMPLETED and end != duration:
                raise ValueError("completed checkpoint does not reach EOF")
        except CheckpointError:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise CheckpointError(f"CHECKPOINT_INVALID: {exc}") from exc

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

        previous = self.current_data
        self.current_data = deepcopy(previous)
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

        try:
            if self.current_data.schema_version == 2:
                self.validate_resume()
            self.save()
        except Exception:
            self.current_data = previous
            raise
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

    def is_completed_for(self, file_name: str, transcript_path: Path, source_identity: SourceIdentity | None = None) -> bool:
        """Require verified source + consistent completed EOF; filename is diagnostic."""
        if source_identity is None:
            return False
        self.validate_resume(source_identity)
        if (
            self.current_data.status == CheckpointStatus.COMPLETED
            and transcript_path.exists()
            and transcript_path.stat().st_size > 0
        ):
            return True
        return False
