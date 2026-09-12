"""Exercise the CLI's actual parser, transcriber, decision, retry and commit path."""
import json
import re

import pytest

from src.checkpoint import CheckpointManager
from src.transcript_validator import TranscriptValidator
from src.fidelity_validator import (ContentFidelityValidator, FidelityValidationResult, FidelityIssue,
                                    FidelityStatus, FidelityRisk, FidelityDecision)
from src.response_parser import TranscriptionBlockResult, TranscriptSegment
from tests.test_stage4_audio_retry import run_audio_case


@pytest.mark.parametrize("mixed", [False, True])
def test_audio_over_normalized_consumes_only_generation_a(tmp_path, monkeypatch, capsys, mixed):
    provider_calls, commits = [], []
    original_validate = TranscriptValidator.validate_block_result
    original_commit = CheckpointManager.commit_block
    def commit(self, **kwargs):
        result = original_commit(self, **kwargs)
        commits.append(kwargs)
        return result
    monkeypatch.setattr(CheckpointManager, "commit_block", commit)

    def generate(self, user_prompt, **kwargs):
        index = [78, 74, 91][len(provider_calls)]
        provider_calls.append(index)
        identity = {field: re.search(r'- ' + field + r': "([^"]+)"', user_prompt).group(1)
                    for field in ("job_id", "session_id", "block_id")}
        segments = [dict(source_index=i, text="Vâng.", timestamp="00:01") for i in range(1, 101)]
        segments[index - 1]["text"] = "đảm bảo an toàn"
        if mixed:
            for i in (81, 83):
                segments[i - 1]["text"] = "Tôi nghĩ [không rõ] phần này."
        return json.dumps(dict(schema_version="1.0", **identity, first_source_index=1,
                               last_source_index=100, status="CONFIRMED", segments=segments))

    def setup(config):
        monkeypatch.setattr("src.main.GeminiClient.generate_transcription", generate)
        monkeypatch.setattr(TranscriptValidator, "validate_block_result", original_validate)
    result, _, _, _, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup,
                                                total_duration=60, attempts=3)
    output = capsys.readouterr()
    if result != 0:
        assert provider_calls == [78, 74, 91]
        assert "BLOCK_REVIEW_REQUIRED" in output.err
        assert "Retrying validation attempt 2" in output.out
        assert commits == []
    assert result == 0, output.out + output.err
    assert provider_calls == [78]
    assert len(commits) == 1
    assert checkpoint["last_confirmed_source_index"] == 100
    metric = checkpoint["block_metrics"][0]
    assert metric["fidelity_status"] == "REVIEW"
    assert metric["fidelity_decision"] == "ACCEPT_WITH_WARNING"
    assert metric["requires_quality_review"] is True
    assert metric["issues"][0]["evidence"] == "HEURISTIC"
    assert [(i["reason"], i["source_index"]) for i in metric["issues"]] == (
        [("OVER_NORMALIZED", 78)] + ([("UNCERTAIN_SPEECH", 81), ("UNCERTAIN_SPEECH", 83)] if mixed else []))
    assert "Retrying validation attempt" not in output.out
    assert "BLOCK_REVIEW_REQUIRED" not in output.err


@pytest.mark.parametrize("reason,attempts", [
    ("POSSIBLE_GENERATION_LOOP", 3), ("UNKNOWN_REVIEW_REASON", 3),
    ("UNSUPPORTED_REPLACEMENT", 1), ("TRUSTED_REFERENCE_MISMATCH", 1),
])
def test_blocking_reasons_still_block_real_orchestration(tmp_path, monkeypatch, reason, attempts):
    def validate(self, block_result):
        return FidelityValidationResult(FidelityStatus.REVIEW, FidelityRisk.MEDIUM,
            [FidelityIssue(1, "OVER_NORMALIZED", "style heuristic"), FidelityIssue(1, reason, "blocking reason")],
            source_mode="AUDIO")
    def setup(config):
        monkeypatch.setattr(ContentFidelityValidator, "validate", validate)
    result, _, _, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup,
        total_duration=60, attempts=3, fail_count=0)
    assert result == 1
    assert len(calls) == attempts
    assert not checkpoint["confirmed_segments"]
    assert not checkpoint["block_metrics"]
    assert checkpoint["next_audio_start_us"] == 0


@pytest.mark.parametrize("mode,trusted,acoustic,evidence,decision", [
    ("AUDIO", False, False, "HEURISTIC", FidelityDecision.ACCEPT_WITH_WARNING),
    ("TEXT_TIMESTAMP", False, False, "HEURISTIC", FidelityDecision.RETRY_REVIEW),
    ("UNKNOWN", False, False, "HEURISTIC", FidelityDecision.RETRY_REVIEW),
    ("AUDIO", True, False, "HEURISTIC", FidelityDecision.RETRY_REVIEW),
    ("AUDIO", False, True, "HEURISTIC", FidelityDecision.RETRY_REVIEW),
    ("AUDIO", False, False, "TRUSTED_REFERENCE", FidelityDecision.RETRY_REVIEW),
    ("AUDIO", False, False, "AUDIO_REVIEW", FidelityDecision.RETRY_REVIEW),
])
def test_only_explicit_unverified_audio_heuristic_is_downgraded(mode, trusted, acoustic, evidence, decision):
    result = FidelityValidationResult(FidelityStatus.REVIEW, FidelityRisk.MEDIUM,
        [FidelityIssue(1, "OVER_NORMALIZED", "test", evidence=evidence)], source_mode=mode,
        has_trusted_reference=trusted, has_acoustic_review=acoustic)
    assert result.decision == decision


def test_trusted_filler_repetition_removal_remains_actionable():
    block = TranscriptionBlockResult("1.0", "job", "session", "block", 1, 1, "CONFIRMED",
                                    [TranscriptSegment(1, "tôi không biết")])
    result = ContentFidelityValidator(source_mode="TEXT_TIMESTAMP").validate(
        block, trusted_source_texts={1: "ờ tôi tôi không không biết"})
    assert result.decision == FidelityDecision.RETRY_REVIEW
    assert not result.allows_confirmation
    assert result.issues[0].evidence == "TRUSTED_REFERENCE"
    assert block.segments[0].text == "tôi không biết"
