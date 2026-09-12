"""Acceptance must distinguish uncertainty from suspicious or proven corruption."""
import pytest

from src.fidelity_validator import (
    ContentFidelityValidator, FidelityDecision, FidelityIssue, FidelityRisk,
    FidelityStatus, FidelityValidationResult,
)
from src.response_parser import TranscriptionBlockResult, TranscriptSegment


def candidate(text):
    return TranscriptionBlockResult("1.0", "job", "session", "block", 1, 1,
                                    "CONFIRMED", [TranscriptSegment(1, text)])


@pytest.mark.parametrize("reason,decision", [
    ("UNCERTAIN_SPEECH", FidelityDecision.ACCEPT_WITH_WARNING),
    ("FULLY_UNCERTAIN_SEGMENT", FidelityDecision.ACCEPT_WITH_WARNING),
    ("HIGH_UNCERTAINTY_RATE", FidelityDecision.ACCEPT_WITH_WARNING),
    ("POSSIBLE_GENERATION_LOOP", FidelityDecision.RETRY_REVIEW),
    ("SEMANTIC_FORMALIZATION", FidelityDecision.RETRY_REVIEW),
    ("OVER_NORMALIZED", FidelityDecision.RETRY_REVIEW),
    ("UNKNOWN_FUTURE_REASON", FidelityDecision.RETRY_REVIEW),
    ("KNOWN_BAD_SUBSTITUTION", FidelityDecision.FAIL),
    ("UNSUPPORTED_REPLACEMENT", FidelityDecision.FAIL),
])
def test_reason_action_mapping(reason, decision):
    result = FidelityValidationResult(FidelityStatus.REVIEW, FidelityRisk.MEDIUM,
                                     [FidelityIssue(1, reason, "test")])
    assert result.decision == decision


def test_unexplained_review_cannot_be_accepted():
    assert FidelityValidationResult(FidelityStatus.REVIEW, FidelityRisk.LOW).decision == FidelityDecision.RETRY_REVIEW


def test_uncertainty_never_masks_loop_or_deterministic_failure():
    validator = ContentFidelityValidator()
    result = validator.validate(candidate("[không rõ] " + "chúng ta cần xử lý phần này " * 5))
    assert result.decision == FidelityDecision.RETRY_REVIEW
    assert not result.allows_confirmation
    result = validator.validate(candidate("[không rõ] dữ liệu"), {1: "[không rõ] nguy cơ"})
    assert result.decision == FidelityDecision.FAIL
    assert not result.allows_confirmation


def test_uncertainty_is_accepted_while_evidenced_guess_is_blocked():
    validator = ContentFidelityValidator()
    uncertainty = validator.validate(candidate("[không rõ]"))
    # Trusted evidence is necessary: a text-only validator cannot identify every guess.
    guess = validator.validate(candidate("dữ liệu"), {1: "nguy cơ"})
    assert uncertainty.decision == FidelityDecision.ACCEPT_WITH_WARNING
    assert uncertainty.allows_confirmation
    assert guess.decision == FidelityDecision.FAIL
    assert not guess.allows_confirmation
