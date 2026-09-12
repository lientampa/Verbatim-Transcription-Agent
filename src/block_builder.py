"""Block Builder Implementation for Milestone 2 & 3.

Segments long audio files (e.g. 500MB+ or multi-hour recordings) into
deterministic blocks with sequential integer SOURCE_INDEX (001, 002, 003...).
SOURCE_INDEX is generated deterministically by the backend, NEVER by the LLM.

Milestone 3: DabbAudioOrchestrator wires DynamicAdaptiveBlockBuilder (DABB)
into the audio pipeline, making block duration token-budget-driven instead of
fixed-duration. DABB grow/shrink logic feeds back after each block result.
"""

import logging
import math
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from uuid import uuid4
import json
import hashlib
from src.source_identity import SourceIdentity, to_us, UNITS_PER_SECOND, sha256_file
from pathlib import Path
from typing import Iterator

from src.adaptive_block_builder import (
    SourceSegment,
    AdaptiveBlock,
    DynamicAdaptiveBlockBuilder,
)
from src.failure_classifier import FailureType

logger = logging.getLogger(__name__)


class BlockBuilderError(Exception):
    """Base exception for block builder errors."""
    pass



@dataclass(frozen=True)
class SourceBlock:
    """Represents a segmented audio block."""
    source_index: int
    source_index_str: str
    start_time_seconds: float
    end_time_seconds: float
    file_path: Path
    total_blocks: int


class BlockBuilder(ABC):
    """Abstract interface for segmenting source audio into indexable blocks."""

    @abstractmethod
    def __init__(self, audio_path: Path, block_size_seconds: int = 300, cache_dir: Path | None = None) -> None:
        pass

    @abstractmethod
    def get_duration(self) -> float:
        """Return total duration of the audio in seconds."""
        pass

    @abstractmethod
    def total_blocks(self) -> int:
        """Return the total number of deterministic source blocks."""
        pass

    @abstractmethod
    def build_blocks(self) -> list[SourceBlock]:
        """Build and return all sequential source blocks."""
        pass


def _find_binary(name: str) -> str:
    """Find the path to ffmpeg or ffprobe executable."""
    found = shutil.which(name)
    if found:
        return found

    # Check common Windows WinGet installation paths
    local_appdata = os.getenv("LOCALAPPDATA", "")
    if local_appdata:
        winget_path = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        matches = list(winget_path.glob(f"**/{name}.exe"))
        if matches:
            return str(matches[0])

    raise BlockBuilderError(
        f"Required binary '{name}' was not found. Please ensure FFmpeg is installed."
    )


class AudioBlockBuilder(BlockBuilder):
    """Concrete BlockBuilder utilizing FFmpeg for streaming audio slicing."""

    def __init__(
        self,
        audio_path: Path,
        block_size_seconds: int = 300,
        cache_dir: Path | None = None,
        job_id: str | None = None,
        source_identity: SourceIdentity | None = None,
    ) -> None:
        self.audio_path = Path(audio_path)
        if not self.audio_path.exists():
            raise BlockBuilderError(f"Audio file not found: {self.audio_path}")

        self.block_size_seconds = max(10, block_size_seconds)
        self.job_id = job_id or "default_job"
        self.source_identity = source_identity

        if cache_dir is None:
            cache_dir = self.audio_path.parent.parent / ".cache" / "blocks"
        self.cache_dir = Path(cache_dir) / self.job_id
        self.shared_cache_dir = Path(cache_dir) / "slices"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._ffmpeg_bin = _find_binary("ffmpeg")
        self._ffprobe_bin = _find_binary("ffprobe")
        self._duration: float | None = None

    def get_duration(self) -> float:
        """Probe audio file duration in seconds."""
        if self._duration is not None:
            return self._duration

        cmd = [
            self._ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(self.audio_path),
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            duration_str = result.stdout.strip()
            self._duration = float(duration_str)
            return self._duration
        except (subprocess.CalledProcessError, ValueError) as exc:
            raise BlockBuilderError(f"Failed to probe audio duration via ffprobe: {exc}") from exc

    def total_blocks(self) -> int:
        """Calculate total number of blocks."""
        duration = self.get_duration()
        if duration <= 0:
            return 1
        return max(1, math.ceil(duration / self.block_size_seconds))

    def slice_block(self, source_index: int, start_sec: float, end_sec: float) -> SourceBlock:
        """Slice a single audio block into cache folder.

        Uses stream copy (-c copy) for high-speed slicing without re-encoding,
        falling back to aac encoding if container constraints require it.
        """
        source_index_str = f"{source_index:03d}"
        ext = self.audio_path.suffix.lower()
        block_file = self.cache_dir / f"block_{source_index_str}{ext}"

        if block_file.exists() and block_file.stat().st_size > 0:
            return SourceBlock(
                source_index=source_index,
                source_index_str=source_index_str,
                start_time_seconds=start_sec,
                end_time_seconds=end_sec,
                file_path=block_file,
                total_blocks=self.total_blocks(),
            )

        duration = end_sec - start_sec

        # Try fast copy first
        copy_cmd = [
            self._ffmpeg_bin,
            "-y",
            "-ss", f"{start_sec:.2f}",
            "-t", f"{duration:.2f}",
            "-i", str(self.audio_path),
            "-c", "copy",
            str(block_file),
        ]

        proc = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            # Fallback to audio re-encode
            encode_cmd = [
                self._ffmpeg_bin,
                "-y",
                "-ss", f"{start_sec:.2f}",
                "-t", f"{duration:.2f}",
                "-i", str(self.audio_path),
                "-c:a", "aac",
                "-b:a", "64k",
                str(block_file),
            ]
            re_proc = subprocess.run(encode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if re_proc.returncode != 0:
                raise BlockBuilderError(
                    f"FFmpeg failed to slice block {source_index_str}: {re_proc.stderr.decode('utf-8', errors='ignore')}"
                )

        return SourceBlock(
            source_index=source_index,
            source_index_str=source_index_str,
            start_time_seconds=start_sec,
            end_time_seconds=end_sec,
            file_path=block_file,
            total_blocks=self.total_blocks(),
        )

    def iter_blocks(self) -> Iterator[SourceBlock]:
        """Yield each SourceBlock sequentially."""
        total_duration = self.get_duration()
        n_blocks = self.total_blocks()

        for i in range(1, n_blocks + 1):
            start_sec = (i - 1) * self.block_size_seconds
            end_sec = min(total_duration, i * self.block_size_seconds)
            yield self.slice_block(source_index=i, start_sec=start_sec, end_sec=end_sec)

    def build_blocks(self) -> list[SourceBlock]:
        """Build all blocks and return list."""
        return list(self.iter_blocks())

    def slice_time_range(self, block_id: str, start_sec: float, end_sec: float) -> Path:
        """Reuse only verified source/boundary/profile/artifact cache records."""
        if self.source_identity is None:
            self.source_identity = SourceIdentity.capture(self.audio_path, self.get_duration())
        self.source_identity.assert_unchanged()
        start_us, end_us = to_us(start_sec), to_us(end_sec)
        if not 0 <= start_us < end_us <= self.source_identity.duration_us:
            raise BlockBuilderError("Invalid audio slice boundaries for verified source")
        start_sec, end_sec = start_us / UNITS_PER_SECOND, end_us / UNITS_PER_SECOND
        ext = self.audio_path.suffix.lower()
        identity = dict(source_fingerprint=self.source_identity.fingerprint, start_us=start_us,
                        end_us=end_us, profile=f"ffmpeg-copy-aac64-v1:{ext}")
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.shared_cache_dir.mkdir(parents=True, exist_ok=True)
        final_file = self.shared_cache_dir / f"{key}{ext}"
        sidecar = self.shared_cache_dir / f"{key}.json"
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            if record["identity"] == identity and final_file.stat().st_size > 0 and record["artifact_hash"] == sha256_file(final_file):
                logger.info("CACHE_HIT source=%s start_us=%s end_us=%s", self.source_identity.fingerprint[:12], start_us, end_us)
                return final_file
        except (OSError, ValueError, KeyError, TypeError):
            pass
        logger.info("CACHE_MISS identity/metadata/artifact mismatch source=%s start_us=%s end_us=%s", self.source_identity.fingerprint[:12], start_us, end_us)
        block_file = self.cache_dir / f"{block_id}_{uuid4().hex}{ext}"

        duration = end_sec - start_sec
        if start_sec < 0 or duration <= 0:
            raise BlockBuilderError("Invalid audio slice boundaries")
        copy_cmd = [
            self._ffmpeg_bin,
            "-y",
            "-ss", f"{start_sec:.9f}",
            "-t", f"{duration:.9f}",
            "-i", str(self.audio_path),
            "-c", "copy",
            str(block_file),
        ]
        proc = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            encode_cmd = [
                self._ffmpeg_bin,
                "-y",
                "-ss", f"{start_sec:.9f}",
                "-t", f"{duration:.9f}",
                "-i", str(self.audio_path),
                "-c:a", "aac",
                "-b:a", "64k",
                str(block_file),
            ]
            re_proc = subprocess.run(encode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if re_proc.returncode != 0:
                raise BlockBuilderError(
                    f"FFmpeg failed to slice {block_id}: {re_proc.stderr.decode('utf-8', errors='ignore')}"
                )

        self.source_identity.assert_unchanged()
        record = dict(identity=identity, artifact_hash=sha256_file(block_file))
        temp_metadata = self.cache_dir / f"{uuid4().hex}.json"
        temp_metadata.write_text(json.dumps(record), encoding="utf-8")
        block_file.replace(final_file)
        temp_metadata.replace(sidecar)
        return final_file


# ---------------------------------------------------------------------------
# DABB Audio Orchestrator — Milestone 3
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveBlockSlice:
    """Metadata for a DABB-driven audio block slice."""
    source_block: "SourceBlock"
    block_num: int
    current_duration_seconds: float
    estimated_input_tokens: int
    context_budget: int
    output_budget: int
    generation: int = 1


class DabbAudioOrchestrator:
    """Orchestrates AudioBlockBuilder with DynamicAdaptiveBlockBuilder (DABB).

    Converts DABB's token target into an audio slice duration, then feeds
    actual transcription results back to DABB for adaptive grow/shrink.

    Token ↔ Duration mapping:
        - Vietnamese audio averages ~120 words/minute (moderate density)
        - Tokens per word ≈ 1.25 (Vietnamese BPE)
        - tokens_per_second ≈ 120 * 1.25 / 60 = 2.5 tokens/second
    """

    # Audio density constants (tunable)
    WORDS_PER_MINUTE: float = 120.0
    TOKENS_PER_WORD: float = 1.25

    def __init__(
        self,
        audio_path: Path,
        config: "AppConfig",  # type: ignore[name-defined]  # imported at runtime
        system_prompt: str = "",
        cache_dir: Path | None = None,
        job_id: str | None = None,
    ) -> None:
        from src.config import AppConfig  # local import to avoid circular
        self.config = config
        self.audio_path = Path(audio_path)
        self.cache_dir = cache_dir or config.cache_dir
        self.job_id = job_id or f"audio_{uuid4().hex}"
        if not 0 < config.block_shrink_factor < 1 or config.min_duration_seconds <= 0:
            raise BlockBuilderError("Shrink factor must be between 0 and 1; minimum duration must be positive")
        self.system_prompt = system_prompt

        # DABB instance — no source segments (audio mode: duration is the proxy)
        self.dabb = DynamicAdaptiveBlockBuilder(
            source_segments=None,
            config=config,
            system_prompt=system_prompt,
        )

        # Audio block builder — block_size will be driven by DABB dynamically
        self._audio_builder: AudioBlockBuilder | None = None
        self._total_duration: float | None = None
        self._slice_paths: set[Path] = set()
        self.source_identity: SourceIdentity | None = None
        self.next_block_number = 1
        self.active_model = config.gemini_model
        self.model_profiles = {self.active_model: dict(target=config.max_duration_seconds,
            safe_duration=None, streak=0, coverage_failures=0, size_failures=0, dabb=self.dabb)}
        if (config.coverage_max_generations < 1 or config.coverage_growth_passes < 1
                or not 1 < config.coverage_growth_factor <= 1.25
                or not 0 <= config.coverage_growth_headroom_sec <= config.coverage_tail_gap_threshold_sec
                or not 0 < config.unknown_model_duration_sec <= config.max_duration_seconds):
            raise BlockBuilderError("Invalid coverage adaptation configuration")

        # Compute tokens/second from density constants
        self._tokens_per_second: float = (
            self.WORDS_PER_MINUTE * self.TOKENS_PER_WORD / 60.0
        )

    def _get_audio_builder(self, block_size_seconds: int) -> "AudioBlockBuilder":
        """Return (or recreate) AudioBlockBuilder with given block size."""
        return AudioBlockBuilder(
            audio_path=self.audio_path,
            block_size_seconds=block_size_seconds,
            cache_dir=self.cache_dir,
            job_id=self.job_id,
            source_identity=self.source_identity,
        )

    def get_duration(self) -> float:
        """Return total audio duration (seconds), cached after first probe."""
        if self._total_duration is None:
            # Need a builder just to probe duration
            tmp = AudioBlockBuilder(
                audio_path=self.audio_path,
                block_size_seconds=300,  # doesn't matter for probe
                cache_dir=self.cache_dir,
                job_id=self.job_id,
            )
            self._total_duration = tmp.get_duration()
        return self._total_duration

    def _target_tokens_to_seconds(self, target_tokens: int) -> float:
        """Convert DABB's current token target to audio duration in seconds."""
        raw = target_tokens / max(self._tokens_per_second, 0.1)
        # Clamp to [min_duration, max_duration] from config
        return max(
            self.config.min_duration_seconds,
            min(self.config.max_duration_seconds, raw),
        )

    def iter_adaptive_blocks(
        self, start_from_seconds: float = 0.0
    ) -> Iterator[AdaptiveBlockSlice]:
        """Yield AdaptiveBlockSlice instances driven by DABB token budget.

        Each iteration:
        1. Ask DABB for current token target
        2. Convert to duration in seconds
        3. Slice audio at that duration
        4. Yield block — caller must call on_block_success() or on_block_failure()
        """
        total_duration = self.get_duration()
        current_start = to_us(start_from_seconds) / UNITS_PER_SECOND
        total_duration = to_us(total_duration) / UNITS_PER_SECOND
        block_num = self.next_block_number - 1

        while current_start < total_duration:
            block_num += 1

            # Ask DABB for current token target → convert to duration
            target_tokens = self.dabb.current_target_tokens
            duration_sec = self.current_block_duration_seconds
            block_size = duration_sec

            current_end = min(total_duration, to_us(current_start + block_size) / UNITS_PER_SECOND)
            actual_duration = current_end - current_start
            est_input_tokens = int(actual_duration * self._tokens_per_second)

            safe_context = self.config.model_context_limit - self.config.context_safety_margin
            safe_output = self.config.model_max_output_tokens - self.config.output_safety_margin

            block_id = f"BLOCK_{block_num:03d}"

            logger.info(
                f"[DABB] {block_id}: target={target_tokens}tok → "
                f"duration={actual_duration:.0f}s "
                f"[{int(current_start)//60:02d}:{int(current_start)%60:02d} – "
                f"{int(current_end)//60:02d}:{int(current_end)%60:02d}], "
                f"est_input≈{est_input_tokens}tok"
            )

            # Slice audio
            builder = self._get_audio_builder(block_size)
            source_block = builder.slice_time_range(
                block_id=block_id,
                start_sec=current_start,
                end_sec=current_end,
            )
            self._slice_paths.add(source_block)

            # Build a minimal SourceBlock wrapper
            src_block = SourceBlock(
                source_index=block_num,
                source_index_str=f"{block_num:03d}",
                start_time_seconds=current_start,
                end_time_seconds=current_end,
                file_path=source_block,
                total_blocks=0,  # unknown until all blocks processed
            )

            attempt_slice = AdaptiveBlockSlice(
                source_block=src_block,
                block_num=block_num,
                current_duration_seconds=actual_duration,
                estimated_input_tokens=est_input_tokens,
                context_budget=safe_context,
                output_budget=safe_output,
            )
            yield attempt_slice

            # A rebuilt retry updates the yielded slice. The caller resumes this
            # iterator only after successful validation/commit (or legacy skip).
            current_start = attempt_slice.source_block.end_time_seconds

    def rebuild_audio_block(self, attempt_slice: AdaptiveBlockSlice, reduction_factor: float | None = None) -> AdaptiveBlockSlice:
        """Rebuild the same logical block at its unchanged unconfirmed start.

        Called after SIZE_FAILURE feedback reduces the token target. Cap duration
        against the actual failed interval, even when that interval was a short tail.
        """
        old = attempt_slice.source_block
        old_duration = old.end_time_seconds - old.start_time_seconds
        factor = self.config.block_shrink_factor if reduction_factor is None else reduction_factor
        duration = max(self.config.min_duration_seconds, min(
            self.current_block_duration_seconds, old_duration * factor))
        if duration >= old_duration - 1e-9:
            raise BlockBuilderError("BLOCK_SIZE_EXHAUSTED: cannot reduce current audio interval further")
        end = to_us(old.start_time_seconds + duration) / UNITS_PER_SECOND
        duration = end - old.start_time_seconds
        path = self._get_audio_builder(max(10, int(duration))).slice_time_range(
            block_id=f"BLOCK_{old.source_index_str}", start_sec=old.start_time_seconds, end_sec=end)
        self._slice_paths.add(path)
        attempt_slice.source_block = replace(old, end_time_seconds=end, file_path=path)
        attempt_slice.current_duration_seconds = duration
        attempt_slice.estimated_input_tokens = int(duration * self._tokens_per_second)
        attempt_slice.generation += 1
        return attempt_slice

    def cleanup_slices(self) -> None:
        """Remove only files created by this orchestrator, including failed retries."""
        owned_root = (Path(self.cache_dir) / self.job_id).resolve()
        for path in self._slice_paths:
            if path.resolve().is_relative_to(owned_root):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Unable to remove owned slice %s", path)

    def on_block_success(
        self,
        actual_input_tokens: int = 0,
        actual_output_tokens: int = 0,
    ) -> None:
        """Call after a block is validated & checkpointed successfully.

        Passes actual token usage to DABB so it can adaptively grow block size.
        """
        # Create a minimal AdaptiveBlock proxy for DABB tracking
        proxy = AdaptiveBlock(
            block_id="__proxy__",
            first_source_index=0,
            last_source_index=0,
            segments=[],
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
        )
        self.dabb.on_block_confirmed(
            block=proxy,
            actual_input_tokens=actual_input_tokens,
            actual_output_tokens=actual_output_tokens,
        )

    def on_block_failure(self, failure_type: FailureType) -> bool:
        """Call when a block fails. Returns True if block should be retried smaller."""
        proxy = AdaptiveBlock(
            block_id="__proxy__",
            first_source_index=0,
            last_source_index=0,
            segments=[],
        )
        if failure_type == FailureType.SIZE_FAILURE:
            self.model_profiles[self.active_model]["size_failures"] += 1
            self.model_profiles[self.active_model]["streak"] = 0
        return self.dabb.on_block_failure(block=proxy, failure_type=failure_type)

    @property
    def current_block_duration_seconds(self) -> float:
        """Current DABB-recommended block duration in seconds."""
        return min(self._target_tokens_to_seconds(self.dabb.current_target_tokens),
                   self.model_profiles[self.active_model]["target"])

    def select_model(self, model: str, reason=None) -> float:
        if model != self.active_model:
            previous, target = self.active_model, self.current_block_duration_seconds
            if model not in self.model_profiles:
                self.model_profiles[model] = dict(target=max(self.config.min_duration_seconds,
                    min(target, self.config.unknown_model_duration_sec)), safe_duration=None,
                    streak=0, coverage_failures=0, size_failures=0,
                    dabb=DynamicAdaptiveBlockBuilder(source_segments=None, config=self.config,
                                                     system_prompt=self.system_prompt))
            self.active_model = model
            self.dabb = self.model_profiles[model]["dabb"]
            print(f" [MODEL_SWITCH from={previous} to={model} reason={reason} profile_target={self.current_block_duration_seconds}]", flush=True)
        return self.current_block_duration_seconds

    def on_coverage_failure(self, duration: float) -> None:
        profile = self.model_profiles[self.active_model]
        profile["target"] = max(self.config.min_duration_seconds,
            min(profile["target"], duration * self.config.coverage_shrink_factor))
        profile["streak"] = 0
        profile["coverage_failures"] += 1
        print(f" [COVERAGE_FAIL actual_model={self.active_model} duration={duration} next_target={self.current_block_duration_seconds}]", flush=True)

    def on_coverage_success(self, duration: float, coverage) -> None:
        profile = self.model_profiles[self.active_model]
        profile["safe_duration"] = duration
        profile["target"] = min(profile["target"], max(self.config.min_duration_seconds, duration))
        strong = coverage.decision.value == "COVERAGE_PASS" and coverage.tail_gap_seconds <= self.config.coverage_growth_headroom_sec
        profile["streak"] = profile["streak"] + 1 if strong else 0
        reason = "accepted_physical_duration"
        if profile["streak"] >= self.config.coverage_growth_passes:
            profile["target"] = min(self.config.max_duration_seconds, profile["target"] * self.config.coverage_growth_factor)
            profile["streak"] = 0
            reason = "coverage_headroom_growth"
        print(f" [COVERAGE_ACCEPT actual_model={self.active_model} safe_duration={duration} next_target={self.current_block_duration_seconds} reason={reason}]", flush=True)
