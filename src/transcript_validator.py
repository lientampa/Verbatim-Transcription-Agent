"""Transcript Validator module for Milestone 3 quality assurance on structured results."""

import re
from dataclasses import dataclass, field
from typing import Any
from src.response_parser import TranscriptionBlockResult, TranscriptSegment


@dataclass(frozen=True)
class ExpectedBlockContext:
    """Application request identity; timestamps are absolute seconds.

    One second of boundary tolerance covers whole-second timestamps and the
    prompt's floored slice offset. Comparison never changes the returned value.

    AUDIO ordinals are response-local; TEXT_TIMESTAMP indices are canonical.
    last_source_index is optional; AUDIO final endpoint discrepancies are advisory.
    """
    job_id: str
    session_id: str
    block_id: str
    first_source_index: int | None
    last_source_index: int | None = None
    audio_start: float = 0.0
    audio_end: float | None = None
    timestamp_tolerance: float = 1.0
    source_mode: str = "TEXT_TIMESTAMP"
    native_wordinfo_order: bool = False


@dataclass
class ValidationResult:
    """Represents the outcome of transcript validation."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation_tier: str = "STRUCTURAL"  # STRUCTURAL | FIDELITY | COMBINED
    reason_codes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.is_valid:
            warn_str = f" ({len(self.warnings)} warning(s))" if self.warnings else ""
            return f"PASS{warn_str}"
        if len(self.errors) > 5:
            from collections import Counter
            counts = Counter(self.reason_codes)
            return f"FAIL: {dict(counts)}; first={self.errors[0]}; last={self.errors[-1]}; full diagnostics retained"
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
        expected_context: ExpectedBlockContext | None = None,
    ) -> ValidationResult:
        """Validate a structured TranscriptionBlockResult against all Milestone 3 rules.

        Checks:
        1. TEXT_TIMESTAMP canonical coverage; AUDIO ordinals need not be contiguous
        2. Duplicate (no duplicated indices)
        3. Order (strictly increasing indices)
        4. Range (no indices outside [first, last])
        5. Empty text (no segment has empty text)
        6. Timestamp formatting (if present, valid HH:MM:SS or MM:SS)
        7. Speaker consistency
        8. Verbatim integrity (no AI meta-commentary; loops reviewed by fidelity)
        """
        errors: list[str] = []
        warnings: list[str] = []
        codes: list[str] = []

        def reject(code: str, detail: str) -> None:
            codes.append(code)
            errors.append(f"{code}: {detail}")

        # Domain objects may be supplied directly, bypassing the JSON parser.
        from src.response_parser import ResponseParser
        import jsonschema
        shape_errors = list(jsonschema.Draft7Validator(ResponseParser()._load_schema()).iter_errors(block_result.to_dict()))
        if shape_errors:
            for error in shape_errors:
                reject("SCHEMA_VALIDATION_ERROR", f"{list(error.path)}: {error.message}")
            return ValidationResult(False, errors, warnings, reason_codes=codes)

        if expected_context is not None:
            expected_first_index = expected_context.first_source_index
            expected_last_index = expected_context.last_source_index
            for name in ("job_id", "session_id", "block_id"):
                expected = getattr(expected_context, name)
                received = getattr(block_result, name)
                if expected != received:
                    reject(f"{name.upper()}_MISMATCH", f"{name}: expected={expected!r}, received={received!r}")

        if block_result.status != "CONFIRMED":
            reject("FAILED_RESPONSE_STATUS" if block_result.status == "FAILED" else "INVALID_RESPONSE_STATUS",
                   f"status: expected='CONFIRMED', received={block_result.status!r}; model status cannot authorize confirmation")

        audio_mode = expected_context is not None and expected_context.source_mode == "AUDIO"
        if audio_mode:
            expected_first_index = expected_last_index = None

        first_idx = expected_first_index if expected_first_index is not None else block_result.first_source_index

        if expected_last_index is not None:
            last_idx = expected_last_index
            if block_result.last_source_index != last_idx:
                reject("LAST_SOURCE_INDEX_MISMATCH", f"last_source_index: expected={last_idx}, received={block_result.last_source_index}")
        else:
            last_idx = block_result.last_source_index

        if audio_mode and block_result.segments:
            # Declared AUDIO end is redundant metadata, not a bound on local ordinals.
            last_idx = block_result.segments[-1].source_index

        # 4. Check Range consistency with block metadata
        if block_result.first_source_index != first_idx:
            reject("FIRST_SOURCE_INDEX_MISMATCH", f"first_source_index: expected={first_idx}, received={block_result.first_source_index}")

        if not block_result.segments:
            errors.append("Segments list is empty.")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings, reason_codes=codes + ["EMPTY_SEGMENTS"])

        if last_idx < first_idx:
            reject("INVALID_SOURCE_RANGE", f"first={first_idx}, last={last_idx}")
        actual_indices = [seg.source_index for seg in block_result.segments]
        for field_name, received, actual in [
            ("FIRST_SOURCE_INDEX", block_result.first_source_index, actual_indices[0]),
            ("LAST_SOURCE_INDEX", block_result.last_source_index, actual_indices[-1]),
        ]:
            if received != actual:
                if audio_mode and field_name == "LAST_SOURCE_INDEX":
                    warnings.append(f"AUDIO_LAST_SEGMENT_ORDINAL_MISMATCH: received_last_source_index={received}, final_segment_ordinal={actual}, block_id={block_result.block_id}")
                    continue
                reject(field_name + "_MISMATCH", f"{field_name.lower()}: segment endpoint={actual}, received={received}")

        # 2. Duplicate Check
        seen_indices = set()
        duplicates = []
        for idx in actual_indices:
            if idx in seen_indices:
                duplicates.append(idx)
            seen_indices.add(idx)
        if duplicates:
            reject("DUPLICATE_SOURCE_INDEX", f"Duplicate source_index detected: {duplicates}")

        # 3. Order Check
        for i in range(len(actual_indices) - 1):
            if actual_indices[i] >= actual_indices[i + 1]:
                reject("OUT_OF_ORDER_SOURCE_INDEX", f"Wrong order: index {actual_indices[i]} appears before {actual_indices[i+1]}.")
                break

        # 4. Range Check on individual segments
        out_of_range = [idx for idx in actual_indices if idx < first_idx or idx > last_idx]
        if out_of_range:
            reject("OUT_OF_RANGE_SOURCE_INDEX", f"Source indices out of range [{first_idx}, {last_idx}]: {out_of_range}")

        # 1. Coverage Check
        # Diagnose gaps without allocating a model-controlled range of arbitrary size.
        if not audio_mode:
            cursor = first_idx
            for idx in sorted(i for i in seen_indices if first_idx <= i <= last_idx):
                if idx > cursor:
                    reject("MISSING_SOURCE_INDEX", f"Missing source indices: {cursor}..{idx - 1}")
                cursor = idx + 1
            if cursor <= last_idx:
                reject("MISSING_SOURCE_INDEX", f"Missing source indices: {cursor}..{last_idx}")

        # Validate each segment content
        previous_seconds = None
        for seg in block_result.segments:
            # 5. Empty text check
            if not seg.text or not seg.text.strip():
                errors.append(f"Empty text in segment index {seg.source_index}.")

            # 6. Timestamp check (if present)
            if seg.timestamp is not None:
                ts = seg.timestamp
                parts = ts.split(":")
                valid = bool(re.fullmatch(r"(?:[0-9]{1,2}:)?[0-9]{1,2}:[0-9]{2}", ts))
                if valid:
                    values = list(map(int, parts))
                    valid = values[-1] < 60 and (len(values) == 2 or values[-2] < 60)
                if not valid:
                    reject("INVALID_TIMESTAMP", f"segment={seg.source_index}, received={ts!r}, expected=[HH:]MM:SS with valid clock components")
                else:
                    seconds = sum(value * multiplier for value, multiplier in zip(reversed(values), (1, 60, 3600)))
                    if previous_seconds is not None and seconds < previous_seconds and not (expected_context and expected_context.source_mode == "AUDIO" and expected_context.native_wordinfo_order):
                        reject("TIMESTAMP_ORDER_ERROR", f"segment={seg.source_index}, received={ts!r}, previous_seconds={previous_seconds}")
                    previous_seconds = seconds
                    if expected_context and (seconds < expected_context.audio_start - expected_context.timestamp_tolerance or
                            (expected_context.audio_end is not None and seconds > expected_context.audio_end + expected_context.timestamp_tolerance)):
                        reject("TIMESTAMP_OUT_OF_RANGE", f"segment={seg.source_index}, received={ts!r}, expected absolute seconds [{expected_context.audio_start}, {expected_context.audio_end}], tolerance={expected_context.timestamp_tolerance}s")

            # 7. Speaker check
            if seg.speaker is not None and not seg.speaker.strip():
                warnings.append(f"Speaker in segment {seg.source_index} is empty or whitespace.")

            # 8. Verbatim integrity: check forbidden AI meta-commentary
            text_lower = seg.text.lower()
            for phrase in self.FORBIDDEN_META_PHRASES:
                if phrase in text_lower:
                    errors.append(f"Forbidden AI meta-commentary in segment {seg.source_index}: '{phrase}'.")

            # Repetition is spoken content; heuristic loop review belongs to fidelity.

        # Timestamp estimate warning: if timestamps exist but are all Gemini estimates
        # (no forced alignment in current pipeline), add informational warning
        segs_with_ts = [s for s in block_result.segments if s.timestamp]
        if segs_with_ts and not errors:
            warnings.append(
                "TIMESTAMP_ESTIMATE: Timestamps in this block are Gemini estimates (no forced alignment). "
                "Do not treat as exact timing values."
            )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, validation_tier="STRUCTURAL", reason_codes=codes)

    def validate(
        self,
        data: Any,
        expected_first_index: int | None = None,
        expected_last_index: int | None = None,
        expected_start_seconds: float | None = None,
        expected_context: ExpectedBlockContext | None = None,
    ) -> ValidationResult:
        """Universal validation entrypoint supporting structured results and strings."""
        if isinstance(data, TranscriptionBlockResult):
            return self.validate_block_result(
                data,
                expected_first_index=expected_first_index,
                expected_last_index=expected_last_index,
                expected_context=expected_context,
            )

        if isinstance(data, dict):
            try:
                import json
                from src.response_parser import ResponseParser
                block_res = ResponseParser().parse(json.dumps(data))
                return self.validate_block_result(
                    block_res,
                    expected_first_index=expected_first_index,
                    expected_last_index=expected_last_index,
                    expected_context=expected_context,
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
                        expected_context=expected_context,
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

            # Legacy text callers receive an advisory; repetition cannot prove failure.
            from src.fidelity_validator import ContentFidelityValidator
            if ContentFidelityValidator().check_repetition([TranscriptSegment(1, cleaned)]):
                warnings.append("POSSIBLE_GENERATION_LOOP: audio review needed.")

            # Timestamp check
            if not re.search(r"\[?\d{1,2}:\d{2}\]?", cleaned):
                warnings.append("No timestamp markers detected in transcript text.")

            return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

        return ValidationResult(is_valid=False, errors=[f"Unsupported data type for validation: {type(data).__name__}"])
