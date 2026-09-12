"""Transcript Validator module for Milestone 3 quality assurance on structured results."""

import re
from dataclasses import dataclass, field
from typing import Any
from src.response_parser import TranscriptionBlockResult, TranscriptSegment


@dataclass
class ValidationResult:
    """Represents the outcome of transcript validation."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation_tier: str = "STRUCTURAL"  # STRUCTURAL | FIDELITY | COMBINED

    def summary(self) -> str:
        if self.is_valid:
            warn_str = f" ({len(self.warnings)} warning(s))" if self.warnings else ""
            return f"PASS{warn_str}"
        return f"FAIL: {'; '.join(self.errors)}"


class TranscriptValidator:
    """Rule-based quality assurance validator for structured Vietnamese verbatim transcripts."""

    FORBIDDEN_META_PHRASES: list[str] = [
        "dưới đây là bản phiên âm",
        "đây là nội dung phiên âm",
        "đây là bản phiên âm",
        "hy vọng bản phiên âm",
        "hy vọng nội dung này",
        "nội dung phiên âm như sau",
        "theo yêu cầu của bạn",
        "tôi đã phiên âm",
        "tóm tắt nội dung:",
        "bản ghi âm kết thúc tại đây",
    ]

    TIMESTAMP_PATTERN = re.compile(r"^(?:\d{1,2}:)?\d{1,2}:\d{2}$")

    def __init__(self, max_consecutive_repetitions: int = 4) -> None:
        self.max_consecutive_repetitions = max_consecutive_repetitions

    def validate_block_result(
        self,
        block_result: TranscriptionBlockResult,
        expected_first_index: int | None = None,
        expected_last_index: int | None = None,
    ) -> ValidationResult:
        """Validate a structured TranscriptionBlockResult against all Milestone 3 rules.

        Checks:
        1. Coverage (all indices from first to last must be present)
        2. Duplicate (no duplicated indices)
        3. Order (strictly increasing indices)
        4. Range (no indices outside [first, last])
        5. Empty text (no segment has empty text)
        6. Timestamp formatting (if present, valid HH:MM:SS or MM:SS)
        7. Speaker consistency
        8. Verbatim integrity (no AI meta-commentary, no hallucination loops)
        """
        errors: list[str] = []
        warnings: list[str] = []

        first_idx = expected_first_index if expected_first_index is not None else block_result.first_source_index

        if expected_last_index is not None:
            last_idx = expected_last_index
            if block_result.last_source_index != last_idx:
                errors.append(f"Block last_source_index {block_result.last_source_index} does not match expected {last_idx}.")
        else:
            last_idx = block_result.segments[-1].source_index if block_result.segments else block_result.last_source_index
            block_result.last_source_index = last_idx

        # 4. Check Range consistency with block metadata
        if block_result.first_source_index != first_idx:
            errors.append(f"Block first_source_index {block_result.first_source_index} does not match expected {first_idx}.")

        if not block_result.segments:
            errors.append("Segments list is empty.")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        expected_indices = list(range(first_idx, last_idx + 1))
        actual_indices = [seg.source_index for seg in block_result.segments]

        # 2. Duplicate Check
        seen_indices = set()
        duplicates = []
        for idx in actual_indices:
            if idx in seen_indices:
                duplicates.append(idx)
            seen_indices.add(idx)
        if duplicates:
            errors.append(f"Duplicate source_index detected: {duplicates}")

        # 3. Order Check
        for i in range(len(actual_indices) - 1):
            if actual_indices[i] >= actual_indices[i + 1]:
                errors.append(f"Wrong order: index {actual_indices[i]} appears before {actual_indices[i+1]}.")
                break

        # 4. Range Check on individual segments
        out_of_range = [idx for idx in actual_indices if idx < first_idx or idx > last_idx]
        if out_of_range:
            errors.append(f"Source indices out of range [{first_idx}, {last_idx}]: {out_of_range}")

        # 1. Coverage Check
        missing_indices = [idx for idx in expected_indices if idx not in seen_indices]
        if missing_indices:
            errors.append(f"Missing source indices: {missing_indices}")

        # Validate each segment content
        for seg in block_result.segments:
            # 5. Empty text check
            if not seg.text or not seg.text.strip():
                errors.append(f"Empty text in segment index {seg.source_index}.")

            # 6. Timestamp check (if present)
            if seg.timestamp is not None:
                ts = seg.timestamp.strip()
                if ts and not self.TIMESTAMP_PATTERN.match(ts):
                    warnings.append(f"Timestamp '{ts}' in segment {seg.source_index} does not match [HH:]MM:SS pattern.")

            # 7. Speaker check
            if seg.speaker is not None and not seg.speaker.strip():
                warnings.append(f"Speaker in segment {seg.source_index} is empty or whitespace.")

            # 8. Verbatim integrity: check forbidden AI meta-commentary
            text_lower = seg.text.lower()
            for phrase in self.FORBIDDEN_META_PHRASES:
                if phrase in text_lower:
                    errors.append(f"Forbidden AI meta-commentary in segment {seg.source_index}: '{phrase}'.")

            # Check phrase loops
            phrase_loop_match = re.search(r"(\b[\w\s]{4,30}\b)(?:\s+\1){4,}", seg.text, re.IGNORECASE)
            if phrase_loop_match:
                errors.append(f"Repetitive phrase loop in segment {seg.source_index}: '{phrase_loop_match.group(1).strip()}'.")

        # Timestamp estimate warning: if timestamps exist but are all Gemini estimates
        # (no forced alignment in current pipeline), add informational warning
        segs_with_ts = [s for s in block_result.segments if s.timestamp]
        if segs_with_ts and not errors:
            warnings.append(
                "TIMESTAMP_ESTIMATE: Timestamps in this block are Gemini estimates (no forced alignment). "
                "Do not treat as exact timing values."
            )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, validation_tier="STRUCTURAL")

    def validate(
        self,
        data: Any,
        expected_first_index: int | None = None,
        expected_last_index: int | None = None,
        expected_start_seconds: float | None = None,
    ) -> ValidationResult:
        """Universal validation entrypoint supporting structured results and strings."""
        if isinstance(data, TranscriptionBlockResult):
            return self.validate_block_result(
                data,
                expected_first_index=expected_first_index,
                expected_last_index=expected_last_index,
            )

        if isinstance(data, dict):
            try:
                block_res = TranscriptionBlockResult.from_dict(data)
                return self.validate_block_result(
                    block_res,
                    expected_first_index=expected_first_index,
                    expected_last_index=expected_last_index,
                )
            except Exception as exc:
                return ValidationResult(is_valid=False, errors=[f"Invalid block result structure: {exc}"])

        # String validation (e.g. legacy text or raw response string)
        if isinstance(data, str):
            cleaned = data.strip()
            if not cleaned:
                return ValidationResult(is_valid=False, errors=["Empty transcript text."])

            # Check if JSON string
            if cleaned.startswith("{") or "```json" in cleaned:
                try:
                    from src.response_parser import ResponseParser
                    parser = ResponseParser()
                    block_res = parser.parse(cleaned)
                    return self.validate_block_result(
                        block_res,
                        expected_first_index=expected_first_index,
                        expected_last_index=expected_last_index,
                    )
                except Exception as exc:
                    return ValidationResult(is_valid=False, errors=[f"JSON parsing/schema error: {exc}"])

            # Legacy plain text validation
            errors: list[str] = []
            warnings: list[str] = []
            text_lower = cleaned.lower()

            for phrase in self.FORBIDDEN_META_PHRASES:
                if phrase in text_lower:
                    errors.append(f"Forbidden AI meta-commentary: '{phrase}'.")

            phrase_loop_match = re.search(r"(\b[\w\s]{4,30}\b)(?:\s+\1){4,}", cleaned, re.IGNORECASE)
            if phrase_loop_match:
                errors.append(f"Repetitive phrase loop: '{phrase_loop_match.group(1).strip()}'.")

            # Timestamp check
            if not re.search(r"\[?\d{1,2}:\d{2}\]?", cleaned):
                warnings.append("No timestamp markers detected in transcript text.")

            return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

        return ValidationResult(is_valid=False, errors=[f"Unsupported data type for validation: {type(data).__name__}"])

