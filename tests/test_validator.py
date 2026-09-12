"""Unit tests for TranscriptValidator (Milestone 3)."""

import pytest
from src.validator import TranscriptValidator, ValidationResult


@pytest.fixture
def validator():
    return TranscriptValidator()


def test_validator_valid_transcript(validator):
    good_text = (
        "[00:00] Người nói 1: Xin chào quý vị và các bạn, hôm nay chúng ta tiếp tục.\n"
        "[00:05] Người nói 2: Dạ vâng, tôi nghe rất rõ rồi ạ."
    )
    result = validator.validate(good_text)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert "PASS" in result.summary()


def test_validator_empty_transcript(validator):
    result = validator.validate("   \n\t  ")
    assert result.is_valid is False
    assert any("empty" in e.lower() for e in result.errors)
    assert "FAIL" in result.summary()


def test_validator_forbidden_meta_phrases(validator):
    bad_text = (
        "Dưới đây là bản phiên âm của cuộc hội thoại:\n"
        "[00:00] Người nói 1: Xin chào anh."
    )
    result = validator.validate(bad_text)
    assert result.is_valid is False
    assert any("dưới đây là bản phiên âm" in e.lower() for e in result.errors)


def test_validator_hallucination_loop(validator):
    looping_text = (
        "[00:00] Người nói 1: " + "cảm ơn các bạn rất nhiều " * 6
    )
    result = validator.validate(looping_text)
    assert result.is_valid is False
    assert any("loop" in e.lower() or "repetitive" in e.lower() for e in result.errors)


def test_validator_silence_tag(validator):
    silence_text = "[05:00] [Không có tiếng người nói / im lặng]"
    result = validator.validate(silence_text)
    assert result.is_valid is True
    assert len(result.errors) == 0


def test_validator_warnings_on_missing_timestamps(validator):
    no_timestamp_text = "Người nói 1: Đây là câu nói không có mốc thời gian nào cả."
    result = validator.validate(no_timestamp_text)
    assert result.is_valid is True  # Still valid but has warning
    assert len(result.warnings) > 0
    assert any("timestamp" in w.lower() for w in result.warnings)
