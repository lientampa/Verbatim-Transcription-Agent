"""Vietnamese golden fixtures: heuristic risk is not acoustic verification."""
import json
from copy import deepcopy

import pytest

from src.audio_review import AudioReviewResult, ReviewIssue, ReviewStatus, _parse_review_response
from src.fidelity_validator import ContentFidelityValidator, FidelityStatus, apply_unknown_substitutions
from src.response_parser import TranscriptSegment, TranscriptionBlockResult
from src.transcript_validator import TranscriptValidator


def block(text):
    return TranscriptionBlockResult("1.0", "job", "session", "BLOCK_001", 1, 1,
                                    "CONFIRMED", [TranscriptSegment(1, text, "00:01", "Người A")])


@pytest.mark.parametrize("text", [
    "dữ liệu", "Chúng ta đang xem dữ liệu hôm nay.",
    "À, ừ, thì là tôi nghĩ vậy nhé.", "tôi tôi nghĩ là...", "cái cái này...",
    "ừ ừ đúng rồi", "không không phải thế", "vâng vâng tôi hiểu",
    "anh anh cho tôi hỏi...", "Tôi định... à thôi, để mai.",
    "Tiếp tục", "Resume", "Dừng", "Bắt đầu",
])
def test_golden_spoken_content_passes_unchanged(text):
    candidate = block(text)
    before = deepcopy(candidate)
    result = ContentFidelityValidator().validate(candidate)
    assert result.status == FidelityStatus.PASS
    assert result.issues == []
    assert candidate == before
    assert TranscriptValidator().validate_block_result(candidate).is_valid


def test_substitution_requires_trusted_context():
    candidate = block("Có dữ liệu ở đây.")
    validator = ContentFidelityValidator()
    assert validator.validate(candidate).status == FidelityStatus.PASS
    assert validator.validate(candidate, {1: "Có dữ liệu ở đây."}).status == FidelityStatus.PASS
    assert validator.validate(candidate, {2: "Có nguy cơ ở đây."}).status == FidelityStatus.PASS
    assert validator.validate(candidate, {1: "Nguy cơ khác."}).status == FidelityStatus.PASS
    before = deepcopy(candidate)
    result = validator.validate(candidate, {1: "Có nguy cơ ở đây."})
    assert result.status == FidelityStatus.FAIL
    assert result.issues[0].indicator == "KNOWN_BAD_SUBSTITUTION"
    assert candidate == before


@pytest.mark.parametrize("text,fully,count,total", [
    ("[không rõ]", True, 1, 1),
    ("[không rõ] [không rõ]", True, 2, 2),
    ("tôi [không rõ] cái này", False, 1, 4),
    ("... [KHÔNG RÕ] !", True, 1, 1),
])
def test_semantic_uncertainty(text, fully, count, total):
    candidate = block(text)
    before = deepcopy(candidate)
    result = ContentFidelityValidator().validate(candidate)
    assert any(i.indicator == "FULLY_UNCERTAIN_SEGMENT" for i in result.issues) == fully
    assert result.unknown_token_count == count
    assert result.total_word_count == total
    assert result.unknown_token_rate == count / total
    # Stage 2.1 keeps all uncertainty visible, separately from acceptance.
    assert result.status == FidelityStatus.REVIEW
    assert result.allows_confirmation
    assert candidate == before


@pytest.mark.parametrize("phrase,copies,status", [
    ("chúng ta cần xử lý phần này", 4, FidelityStatus.PASS),
    ("chúng ta cần xử lý phần này", 5, FidelityStatus.REVIEW),
    ("chúng ta cần làm việc này", 5, FidelityStatus.REVIEW),
    ("ừ", 2, FidelityStatus.PASS),
    ("ừ", 12, FidelityStatus.REVIEW),
])
def test_loop_threshold_is_review(phrase, copies, status):
    candidate = block(".\n".join([phrase] * copies))
    before = deepcopy(candidate)
    result = ContentFidelityValidator().validate(candidate)
    assert result.status == status
    assert candidate == before
    assert TranscriptValidator().validate_block_result(candidate).is_valid
    assert any(i.indicator == "POSSIBLE_GENERATION_LOOP" for i in result.issues) == (status == FidelityStatus.REVIEW)


def test_formal_speech_is_review_not_deterministic_failure():
    result = ContentFidelityValidator().validate(block("nhằm mục đích thực hiện công tác"))
    assert result.status == FidelityStatus.REVIEW
    assert result.needs_audio_review


@pytest.mark.parametrize("replacement", ["nội dung hoàn toàn khác", "quảng cáo", "abc [không rõ]", "", None])
def test_arbitrary_replacement_rejected_at_all_boundaries(replacement):
    candidate = block("abc")
    before = deepcopy(candidate)
    with pytest.raises(ValueError):
        apply_unknown_substitutions(candidate, [{"source_index": 1, "replacement": replacement}])
    parsed = _parse_review_response(json.dumps({"status": "PASS", "issues": [
        {"source_index": 1, "candidate_text": "abc", "replacement": replacement}
    ]}))
    assert parsed.status == ReviewStatus.UNCERTAIN
    assert parsed.error
    assert parsed.get_substitutions() == []
    review = AudioReviewResult(ReviewStatus.FAIL, [ReviewIssue(1, "abc", "unverified", replacement)])
    with pytest.raises(ValueError):
        review.get_substitutions()
    assert candidate == before


def test_safe_annotation_preserves_source_and_identity():
    candidate = block("marketing")
    before = deepcopy(candidate)
    parsed = _parse_review_response(json.dumps({"status": "FAIL", "issues": [
        {"source_index": 1, "candidate_text": "marketing", "replacement": "[không rõ]"}
    ]}))
    result = apply_unknown_substitutions(candidate, parsed.get_substitutions())
    expected = deepcopy(candidate)
    expected.segments[0].text = "[không rõ]"
    assert result == expected
    assert candidate == before
    assert ContentFidelityValidator().validate(result).status == FidelityStatus.REVIEW


@pytest.mark.parametrize("substitutions", [
    [{"source_index": 2}], [{"source_index": True}],
    [{"source_index": 1}, {"source_index": 1}],
])
def test_invalid_annotation_identity_rejected(substitutions):
    candidate = block("abc")
    before = deepcopy(candidate)
    with pytest.raises(ValueError):
        apply_unknown_substitutions(candidate, substitutions)
    assert candidate == before
