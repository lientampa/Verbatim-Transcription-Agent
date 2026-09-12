"""Failure classification module for adaptive block sizing decisions."""

from enum import Enum
from typing import Any


class FailureType(str, Enum):
    """Categorization of pipeline errors to determine adaptive reaction.

    TYPE A — STRUCTURAL_FAILURE: Invalid JSON, missing field, duplicate index.
             → retry same block (no shrink).
    TYPE B — SIZE_FAILURE: Output truncation, context overflow.
             → DABB shrink block + retry.
    TYPE C — FIDELITY_FAILURE: Content hallucination, semantic substitution.
             → audio recheck + [không rõ] substitution (no DABB shrink).
    """
    STRUCTURAL_FAILURE = "STRUCTURAL_FAILURE"
    CONTENT_FAILURE = "CONTENT_FAILURE"
    SIZE_FAILURE = "SIZE_FAILURE"
    FIDELITY_FAILURE = "FIDELITY_FAILURE"  # TYPE C — Milestone 3.1
    NETWORK_FAILURE = "NETWORK_FAILURE"
    COVERAGE_FAILURE = "COVERAGE_FAILURE"


def classify_failure(
    error: Exception | str,
    validation_result: Any | None = None,
) -> FailureType:
    """Classify a pipeline or validation error.

    Only SIZE_FAILURE triggers an adaptive block shrink.
    Other failures (structural schema bugs, duplicate indices, content hallucination)
    must trigger retry without shrinking block size.
    """
    cause = error
    while isinstance(cause, Exception):
        if getattr(cause, "failure_type", None) == FailureType.SIZE_FAILURE:
            return FailureType.SIZE_FAILURE
        cause = cause.__cause__
    error_str = str(error).lower()
    if getattr(validation_result, "reason_codes", None):
        return FailureType.STRUCTURAL_FAILURE
    if any(word in error_str for word in ("schema validation", "response parsing", "invalid json", "failed to parse json")):
        return FailureType.STRUCTURAL_FAILURE
    if any(word in error_str for word in ("timeout", "timed out", "connection", "429", "rate limit", "503", "temporarily unavailable")):
        return FailureType.NETWORK_FAILURE
    val_errors = [e.lower() for e in (getattr(validation_result, "errors", []) or [])]
    all_text = f"{error_str} {' '.join(val_errors)}"

    # 1. Check for SIZE_FAILURE indicators
    size_keywords = [
        "output_truncated",
        "context_limit",
        "input too large",
        "context_overflow",
        "max_tokens",
        "output limit",
        "context limit",
        "payload too large",
    ]
    for kw in size_keywords:
        if kw in all_text:
            return FailureType.SIZE_FAILURE

    # 2. Check for FIDELITY_FAILURE indicators (TYPE C — Milestone 3.1)
    fidelity_keywords = [
        "fidelity_fail",
        "fidelity fail",
        "audio review fail",
        "known_bad_substitution",
        "semantic_formalization",
        "high risk",
        "semantic substitution",
    ]
    for kw in fidelity_keywords:
        if kw in all_text:
            return FailureType.FIDELITY_FAILURE

    # 3. Check for CONTENT_FAILURE indicators
    content_keywords = [
        "forbidden ai meta",
        "repetitive phrase loop",
        "hallucination",
        "empty text",
    ]
    for kw in content_keywords:
        if kw in all_text:
            return FailureType.CONTENT_FAILURE

    # 4. Check for STRUCTURAL_FAILURE indicators
    structural_keywords = [
        "duplicate source_index",
        "wrong order",
        "missing source_index",
        "missing source indices",
        "out of range",
        "schema validation failed",
    ]
    for kw in structural_keywords:
        if kw in all_text:
            return FailureType.STRUCTURAL_FAILURE

    # Default fallback to STRUCTURAL_FAILURE to avoid false shrinks
    return FailureType.STRUCTURAL_FAILURE
