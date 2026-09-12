"""Stage 1 integration: real transcriber/parser/validators with a mocked provider."""
import json
from pathlib import Path

import pytest

from src.main import run_pipeline
from src.block_builder import SourceBlock, AdaptiveBlockSlice
from src.failure_classifier import FailureType
from src.transcriber import GeminiTranscriber, TranscriptionOutcome
from src.checkpoint import CheckpointManager
from src.fidelity_validator import ContentFidelityValidator, FidelityValidationResult, FidelityStatus, FidelityRisk


@pytest.mark.parametrize("structural_ok,fidelity,expected_success", [
    (True, "PASS", True),
    (False, "PASS", False),
    (True, "FAIL", False),
    (False, "FAIL", False),
    (True, "REVIEW", False),
    (True, "LOOP", False),
])
def test_validation_gates_preserve_confirmed_progress(
    tmp_path, monkeypatch, structural_ok, fidelity, expected_success
):
    (tmp_path / "audio").mkdir()
    audio = tmp_path / "audio" / "sample.wav"
    audio.write_bytes(b"mock audio")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system_prompt.txt").write_text("Verbatim", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=mock-key\nVALIDATOR_MAX_RETRIES=2\n", encoding="utf-8"
    )
    # Real schema, transcriber, validators, merger, checkpoint, and renderer.
    (tmp_path / "schemas").mkdir()
    schema = Path(__file__).resolve().parent.parent / "schemas" / "transcription_result.schema.json"
    (tmp_path / "schemas" / schema.name).write_bytes(schema.read_bytes())
    slices = [
        AdaptiveBlockSlice(
            SourceBlock(i, f"{i:03d}", (i-1)*60.0, i*60.0, audio, 2),
            i, 60.0, 150, 998000, 7192,
        ) for i in (1, 2)
    ]
    monkeypatch.setattr("src.main.DabbAudioOrchestrator.get_duration", lambda self: 120.0)
    monkeypatch.setattr("src.main.DabbAudioOrchestrator.iter_adaptive_blocks",
                        lambda self, start_from_seconds=0.0: iter(slices))
    monkeypatch.setattr("src.main.GeminiClient.__init__", lambda self, **kwargs: None)
    monkeypatch.setattr("src.main.GeminiClient.upload_audio", lambda self, **kwargs: "mock-file")
    monkeypatch.setattr("src.main.time.sleep", lambda delay: None)
    failures = []
    monkeypatch.setattr("src.main.DabbAudioOrchestrator.on_block_failure",
                        lambda self, failure: failures.append(failure))
    requests = []
    events = []
    original_transcribe = GeminiTranscriber.transcribe_block
    original_commit = CheckpointManager.commit_block
    original_validate = ContentFidelityValidator.validate

    def validate_with_deterministic_failure(self, block_result):
        # Stage 2 removed the invalid lexical FAIL for ordinary "dữ liệu".
        # Inject FAIL to keep testing the orchestrator's independent status gate.
        if block_result.block_id == "BLOCK_002" and fidelity == "FAIL":
            return FidelityValidationResult(FidelityStatus.FAIL, FidelityRisk.HIGH)
        return original_validate(self, block_result)

    monkeypatch.setattr(ContentFidelityValidator, "validate", validate_with_deterministic_failure)

    def trace_transcribe(self, **kwargs):
        outcome = original_transcribe(self, **kwargs)
        assert isinstance(outcome, TranscriptionOutcome)
        events.append((
            "validated", outcome.block_result.block_id,
            outcome.structural_validation.is_valid,
            outcome.fidelity_validation.is_valid,
        ))
        return outcome

    def trace_commit(self, **kwargs):
        # A model-provided CONFIRMED status alone must never authorize a commit.
        assert events[-1] == ("validated", kwargs["block_id"], True, True)
        result = original_commit(self, **kwargs)
        events.append(("checkpoint", kwargs["block_id"]))
        return result

    monkeypatch.setattr(GeminiTranscriber, "transcribe_block", trace_transcribe)
    monkeypatch.setattr(CheckpointManager, "commit_block", trace_commit)

    def generate(self, **kwargs):
        second = "BLOCK_002" in kwargs["user_prompt"]
        requests.append(second)
        index = 2 if second else 1
        text = "Anh... anh cho tôi hỏi cái này."
        if second:
            text = {
                "PASS": "Dạ vâng, anh cứ hỏi.", "FAIL": "dữ liệu",
                "REVIEW": "[không rõ]", "LOOP": "chúng ta cần xử lý phần này " * 5,
            }[fidelity]
        segments = [{"source_index": index, "text": text, "timestamp": "01:00" if second else "00:00"}]
        if second and not structural_ok:
            segments.append(dict(segments[0]))  # Deterministic duplicate failure.
        return json.dumps(dict(
            schema_version="1.0", job_id="mock-job", session_id="mock-session",
            block_id="BLOCK_002" if second else "BLOCK_001",
            first_source_index=index, last_source_index=index,
            status="CONFIRMED", segments=segments,
        ), ensure_ascii=False)
    monkeypatch.setattr("src.main.GeminiClient.generate_transcription", generate)
    result = run_pipeline(base_dir=tmp_path)
    checkpoint = json.loads((tmp_path / "state" / "checkpoint.json").read_text(encoding="utf-8"))
    assert result == (0 if expected_success else 1)
    assert checkpoint["last_confirmed_source_index"] == (2 if expected_success else 1)
    assert checkpoint["next_source_index"] == (3 if expected_success else 2)
    assert checkpoint["current_block_id"] == ("BLOCK_002" if expected_success else "BLOCK_001")
    assert len(checkpoint["confirmed_segments"]) == (2 if expected_success else 1)
    assert checkpoint["confirmed_segments"][0]["text"] == "Anh... anh cho tôi hỏi cái này."
    assert requests == ([False, True] if expected_success else [False, True, True])
    assert events[:2] == [
        ("validated", "BLOCK_001", True, True),
        ("checkpoint", "BLOCK_001"),
    ]
    second_validation = ("validated", "BLOCK_002", structural_ok, fidelity == "PASS")
    assert events[2:] == (
        [second_validation, ("checkpoint", "BLOCK_002")]
        if expected_success else [second_validation, second_validation]
    )
    if expected_success:
        assert checkpoint["status"] == "COMPLETED"
        output = (tmp_path / "output" / "transcript.txt").read_text(encoding="utf-8")
        assert "Dạ vâng, anh cứ hỏi." in output
        assert not failures
    else:
        assert checkpoint["status"] == "FAILED"
        assert "Structural:" in checkpoint["error_message"]
        assert "Fidelity:" in checkpoint["error_message"]
        assert not (tmp_path / "output" / "transcript.txt").exists()
        if fidelity != "PASS":
            assert failures == [FailureType.FIDELITY_FAILURE] * 2
