"""Dynamic Adaptive Block Builder (DABB) for Token-Aware and Context-Aware transcription."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.config import AppConfig, load_config
from src.token_estimator import TokenEstimator, OutputRatioTracker
from src.failure_classifier import FailureType, classify_failure

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceSegment:
    """Immutable source segment with deterministic sequential index."""
    source_index: int
    timestamp: str | None = None
    speaker: str | None = None
    source_text: str = ""
    start_time_seconds: float | None = None
    end_time_seconds: float | None = None


@dataclass
class AdaptiveBlock:
    """Dynamic Adaptive Block with explicit identity, boundaries, and token metrics."""
    block_id: str
    first_source_index: int
    last_source_index: int
    segments: list[SourceSegment]
    start_time_seconds: float | None = None
    end_time_seconds: float | None = None
    file_path: Path | None = None

    # Token & Budget metrics for logging and decision tracking
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None
    context_budget: int = 0
    output_budget: int = 0
    decision: str = "SAFE"  # SAFE / SHRINK / RETRY / CONFIRMED
    retry_count: int = 0

    def to_log_dict(self) -> dict[str, Any]:
        """Format metrics for structured logging conforming to Section 14."""
        start_ts = self.segments[0].timestamp if self.segments and self.segments[0].timestamp else (
            f"{int(self.start_time_seconds // 60):02d}:{int(self.start_time_seconds % 60):02d}"
            if self.start_time_seconds is not None else None
        )
        end_ts = self.segments[-1].timestamp if self.segments and self.segments[-1].timestamp else (
            f"{int(self.end_time_seconds // 60):02d}:{int(self.end_time_seconds % 60):02d}"
            if self.end_time_seconds is not None else None
        )

        return {
            "block_id": self.block_id,
            "first_source_index": self.first_source_index,
            "last_source_index": self.last_source_index,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "actual_input_tokens": self.actual_input_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "context_budget": self.context_budget,
            "output_budget": self.output_budget,
            "decision": self.decision,
            "retry_count": self.retry_count,
        }


class DynamicAdaptiveBlockBuilder:
    """Token-aware, context-aware, output-aware adaptive block builder (DABB).

    Replaces fixed-duration slicing with dynamic budget optimization:
    INPUT TOKENS + OUTPUT TOKENS + CONTEXT LIMIT + OUTPUT LIMIT + SAFETY MARGINS.
    """

    def __init__(
        self,
        source_segments: Sequence[SourceSegment] | None = None,
        config: AppConfig | None = None,
        token_estimator: TokenEstimator | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.token_estimator = token_estimator or TokenEstimator()
        self.output_tracker = OutputRatioTracker(default_ratio=self.config.default_output_ratio)
        self.system_prompt = system_prompt or ""

        # Ensure segments have immutable sequential indices
        raw_segments = list(source_segments or [])
        self.segments: list[SourceSegment] = []
        for i, seg in enumerate(raw_segments):
            # If segment lacks source_index or starts at 0, assign deterministic 1-based index
            idx = seg.source_index if (seg.source_index is not None and seg.source_index > 0) else (i + 1)
            self.segments.append(
                SourceSegment(
                    source_index=idx,
                    timestamp=seg.timestamp,
                    speaker=seg.speaker,
                    source_text=seg.source_text,
                    start_time_seconds=seg.start_time_seconds,
                    end_time_seconds=seg.end_time_seconds,
                )
            )

        # Dynamic target tracker
        self.current_target_tokens: int = self.config.initial_target_input_tokens
        self._block_counter: int = 0
        self.logs: list[dict[str, Any]] = []

    def set_segments(self, segments: Sequence[SourceSegment]) -> None:
        """Update source segments while maintaining sequential identity."""
        self.segments = list(segments)

    def build_next_block(
        self,
        start_source_index: int = 1,
        block_id: str | None = None,
    ) -> AdaptiveBlock | None:
        """Construct the next safe adaptive block starting at start_source_index.

        Enforces:
        1. Context safety margin
        2. Output safety margin
        3. Segment boundary guarantee
        4. Token target constraint
        5. Duration heuristic (secondary soft limit)
        """
        if not self.segments:
            return None

        # Locate first segment >= start_source_index
        start_idx = -1
        for i, seg in enumerate(self.segments):
            if seg.source_index >= start_source_index:
                start_idx = i
                break

        if start_idx == -1:
            return None  # All segments already consumed

        self._block_counter += 1
        b_id = block_id or f"BLOCK_{self._block_counter:03d}"

        sys_tokens = self.token_estimator.estimate_system_prompt(self.system_prompt)
        ctrl_tokens = self.token_estimator.estimate_control_tokens(b_id)
        ratio = self.output_tracker.get_ratio()

        safe_context_limit = self.config.model_context_limit - self.config.context_safety_margin
        safe_output_limit = self.config.model_max_output_tokens - self.config.output_safety_margin

        block_segments: list[SourceSegment] = []
        current_input_tokens = 0
        earliest_start: float | None = None
        latest_end: float | None = None

        for seg in self.segments[start_idx:]:
            seg_tokens = self.token_estimator.estimate_segment(seg)
            prospective_input_tokens = current_input_tokens + seg_tokens
            prospective_output_tokens = self.token_estimator.estimate_output_tokens(prospective_input_tokens, ratio)
            prospective_total_context = sys_tokens + ctrl_tokens + prospective_input_tokens + prospective_output_tokens

            # Evaluate duration heuristic if timestamps exist
            prospective_duration = 0.0
            cur_start = seg.start_time_seconds if seg.start_time_seconds is not None else earliest_start
            cur_end = seg.end_time_seconds if seg.end_time_seconds is not None else latest_end
            if earliest_start is not None and cur_end is not None:
                prospective_duration = max(0.0, cur_end - earliest_start)

            # If we already have at least 1 segment, check boundaries:
            if block_segments:
                # 1. Hard Check: Context limit
                if prospective_total_context > safe_context_limit:
                    break

                # 2. Hard Check: Output limit
                if prospective_output_tokens > safe_output_limit:
                    break

                # 3. Target Input tokens constraint
                if prospective_input_tokens > self.current_target_tokens:
                    break

                # 4. Duration heuristic (secondary soft bound)
                if (
                    self.config.max_duration_seconds > 0
                    and prospective_duration > self.config.max_duration_seconds
                ):
                    break

            # Add segment to block
            block_segments.append(seg)
            current_input_tokens = prospective_input_tokens

            if earliest_start is None and seg.start_time_seconds is not None:
                earliest_start = seg.start_time_seconds
            if seg.end_time_seconds is not None:
                latest_end = seg.end_time_seconds

        if not block_segments:
            return None

        first_index = block_segments[0].source_index
        last_index = block_segments[-1].source_index
        est_output = self.token_estimator.estimate_output_tokens(current_input_tokens, ratio)

        block = AdaptiveBlock(
            block_id=b_id,
            first_source_index=first_index,
            last_source_index=last_index,
            segments=block_segments,
            start_time_seconds=earliest_start,
            end_time_seconds=latest_end,
            estimated_input_tokens=current_input_tokens,
            estimated_output_tokens=est_output,
            context_budget=safe_context_limit,
            output_budget=safe_output_limit,
            decision="SAFE",
            retry_count=0,
        )

        self.log_block(block, decision="SAFE")
        return block

    def on_block_confirmed(
        self,
        block: AdaptiveBlock,
        actual_input_tokens: int,
        actual_output_tokens: int,
    ) -> None:
        """Invoked upon successful validation & checkpoint commit.

        Updates historical output/input ratio and performs Adaptive Growth if safe.
        """
        block.actual_input_tokens = actual_input_tokens
        block.actual_output_tokens = actual_output_tokens
        block.decision = "CONFIRMED"

        self.output_tracker.record_block(actual_input_tokens, actual_output_tokens)

        # Adaptive Growth Rule:
        # If output was well below budget (< 80%), increase next block target
        safe_output_limit = self.config.model_max_output_tokens - self.config.output_safety_margin
        if actual_output_tokens < safe_output_limit * 0.8:
            old_target = self.current_target_tokens
            self.current_target_tokens = min(
                self.current_target_tokens + self.config.block_growth_step,
                self.config.max_block_tokens,
            )
            logger.info(
                f"[DABB Growth] Block {block.block_id} confirmed. Output {actual_output_tokens} << {safe_output_limit}. "
                f"Growing target input tokens: {old_target} -> {self.current_target_tokens}"
            )

        self.log_block(block, decision="CONFIRMED")

    def on_block_failure(
        self,
        block: AdaptiveBlock,
        failure_type: FailureType,
    ) -> bool:
        """Invoked when a block fails execution or validation.

        Only SIZE_FAILURE triggers Adaptive Shrink.
        STRUCTURAL_FAILURE or CONTENT_FAILURE triggers retry without shrink.

        Returns:
            True if block was shrunk, False otherwise.
        """
        block.retry_count += 1

        if failure_type == FailureType.SIZE_FAILURE:
            old_target = self.current_target_tokens
            self.current_target_tokens = max(
                int(self.current_target_tokens * self.config.block_shrink_factor),
                self.config.min_block_tokens,
            )
            block.decision = "SHRINK"
            logger.warning(
                f"[DABB Shrink] Block {block.block_id} encountered SIZE_FAILURE. "
                f"Shrinking target input tokens: {old_target} -> {self.current_target_tokens}"
            )
            self.log_block(block, decision="SHRINK")
            return True
        else:
            block.decision = "RETRY"
            logger.info(
                f"[DABB Retry] Block {block.block_id} encountered {failure_type.value}. "
                f"Retrying without shrinking target (current target: {self.current_target_tokens})."
            )
            self.log_block(block, decision="RETRY")
            return False

    def log_block(self, block: AdaptiveBlock, decision: str | None = None) -> None:
        """Record and output structured log for debugging and observability."""
        if decision:
            block.decision = decision
        log_entry = block.to_log_dict()
        self.logs.append(log_entry)

        logger.info(
            f"[DABB Log] Block: {log_entry['block_id']} | Range: [{log_entry['first_source_index']}..{log_entry['last_source_index']}] | "
            f"Est In/Out: {log_entry['estimated_input_tokens']}/{log_entry['estimated_output_tokens']} tokens | "
            f"Decision: {log_entry['decision']} | Retries: {log_entry['retry_count']}"
        )
