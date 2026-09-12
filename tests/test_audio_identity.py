"""Mode-aware ordinals through parser, validation, coverage, merge and resume."""
from copy import deepcopy
import json

import pytest

from src.gemini_client import GeminiClient
from src.coverage_validator import TailActivityAnalyzer
from src.transcript_validator import ExpectedBlockContext, TranscriptValidator
from src.response_parser import ResponseParser
from tests.test_stage4_audio_retry import run_audio_case


def payload(indices):
    return dict(schema_version="1.0", job_id="j", session_id="s", block_id="BLOCK_001",
                first_source_index=indices[0], last_source_index=indices[-1], status="CONFIRMED",
                segments=[dict(source_index=i, text="Vâng.", timestamp="09:52") for i in indices])


@pytest.mark.parametrize("indices,reason", [
    ([1, 2, 10, 50], None), ([1, 2, 2, 3], "DUPLICATE_SOURCE_INDEX"),
    ([1, 3, 2, 4], "OUT_OF_ORDER_SOURCE_INDEX"),
])
def test_audio_local_structure_without_mutation(indices, reason):
    data = payload(indices)
    parsed = ResponseParser().parse(json.dumps(data))
    before = deepcopy(parsed)
    result = TranscriptValidator().validate_block_result(parsed,
        expected_context=ExpectedBlockContext("j", "s", "BLOCK_001", 999, 1000,
                                              audio_end=600, source_mode="AUDIO"))
    assert parsed == before
    assert "MISSING_SOURCE_INDEX" not in result.reason_codes
    assert result.is_valid == (reason is None)
    if reason:
        assert reason in result.reason_codes


def test_text_timestamp_canonical_gap_still_fails():
    result = TranscriptValidator().validate_block_result(ResponseParser().parse(json.dumps(payload([15, 16, 19]))),
        expected_context=ExpectedBlockContext("j", "s", "BLOCK_001", 15, 19,
                                              audio_end=600, source_mode="TEXT_TIMESTAMP"))
    assert not result.is_valid
    assert any("MISSING_SOURCE_INDEX: Missing source indices: 17..18" in e for e in result.errors)


def install_response(monkeypatch, indices, active=False):
    original = GeminiClient.generate_transcription
    def generate(self, gemini_file, **kwargs):
        data = json.loads(original(self, gemini_file=gemini_file, **kwargs))
        end = 180 if active else int(gemini_file.bounds[1]) - 8
        data.update(first_source_index=indices[0], last_source_index=indices[-1],
                    segments=[dict(source_index=i, text="Vâng.", timestamp=f"{end // 60:02d}:{end % 60:02d}") for i in indices])
        return json.dumps(data)
    monkeypatch.setattr(GeminiClient, "generate_transcription", generate)
    if active:
        monkeypatch.setattr(TailActivityAnalyzer, "analyze", lambda self, path, offset, duration, threshold: duration)


@pytest.mark.parametrize("indices", [list(range(1, 17)) + list(range(20, 41)), [1, 2, 4, 7, 8]])
def test_production_audio_gap_commits_once_with_coverage(tmp_path, monkeypatch, indices):
    def setup(config):
        install_response(monkeypatch, indices)
    result, _, _, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup,
                                                   fail_count=0, total_duration=600)
    assert result == 0
    assert len(calls) == 1
    assert checkpoint["next_audio_start_us"] == 600_000000
    assert checkpoint["block_metrics"][0]["coverage"]["decision"] == "COVERAGE_PASS"
    assert [s["source_index"] for s in checkpoint["confirmed_segments"]] == indices


def test_gaps_do_not_bypass_active_tail_gate(tmp_path, monkeypatch):
    def setup(config):
        install_response(monkeypatch, [1, 2, 10, 50], active=True)
    result, _, _, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup,
                                                   fail_count=0, total_duration=600, attempts=1)
    assert result == 1
    assert len(calls) == 1
    assert checkpoint["next_audio_start_us"] == 0
    assert not checkpoint["confirmed_segments"]
    assert "COVERAGE_RETRY" in checkpoint["error_message"]
    assert "MISSING_SOURCE_INDEX" not in checkpoint["error_message"]


def test_coverage_rebuild_then_gapped_response_succeeds(tmp_path, monkeypatch):
    attempts = []
    def setup(config):
        install_response(monkeypatch, list(range(1, 17)) + list(range(20, 41)))
        original = GeminiClient.generate_transcription
        def generate(self, **kwargs):
            data = json.loads(original(self, **kwargs))
            attempts.append(data)
            if len(attempts) == 1:
                for segment in data["segments"]:
                    segment["timestamp"] = "03:00"
            return json.dumps(data)
        monkeypatch.setattr(GeminiClient, "generate_transcription", generate)
        monkeypatch.setattr(TailActivityAnalyzer, "analyze", lambda self, path, offset, duration, threshold: duration)
    result, _, uploads, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup,
        fail_count=0, total_duration=600, attempts=2)
    assert result == 0
    assert [c[0].bounds for c in calls] == [[0, 600], [0, 300], [300, 600]]
    assert uploads[0].path != uploads[1].path
    assert checkpoint["block_metrics"][0]["attempt_count"] == 2
    assert checkpoint["next_audio_start_us"] == 600_000000


@pytest.mark.parametrize("restart", [False, True])
def test_local_ordinals_restart_across_blocks_and_resume(tmp_path, monkeypatch, restart):
    indices = [1, 2, 10, 50]
    def setup(config):
        install_response(monkeypatch, indices)
    result, _, _, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup,
                                                   fail_count=0, total_duration=900, restart=restart)
    assert result == 0
    assert len(calls) == (1 if restart else 2)
    assert [s["source_index"] for s in checkpoint["confirmed_segments"]] == indices * 2
    assert [s["block_id"] for s in checkpoint["confirmed_segments"]] == ["BLOCK_001"] * 4 + ["BLOCK_002"] * 4
    assert checkpoint["next_audio_start_us"] == 900_000000
    # Both blocks survive output merging; no global ordinal deduplication.
    output = (tmp_path / "output" / "transcript.txt").read_text(encoding="utf-8")
    assert output.count("Vâng.") == 8
