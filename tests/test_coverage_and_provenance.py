from copy import deepcopy
from dataclasses import replace
import json
import re
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from src.coverage_validator import CoverageValidator, CoverageDecision, TailActivityAnalyzer
from src.block_builder import SourceBlock
from src.response_parser import TranscriptionBlockResult, TranscriptSegment
from src.gemini_client import GeminiClient
from src.validator import TranscriptValidator
from tests.test_stage4_audio_retry import run_audio_case


def coverage(start, end, timestamp, active):
    config = SimpleNamespace(coverage_tail_gap_threshold_sec=120, coverage_active_tail_min_sec=30,
                             coverage_silence_threshold_db=-40, coverage_shrink_factor=0.5)
    observed = []
    def analyze(path, offset, duration, threshold):
        observed.append((offset, duration))
        return active
    transcript = TranscriptionBlockResult("1.0", "j", "s", "b", 1, 1, "CONFIRMED",
                                         [TranscriptSegment(1, "Vâng.", timestamp)])
    before = deepcopy(transcript)
    result = CoverageValidator(config, SimpleNamespace(analyze=analyze)).validate(
        SourceBlock(1, "001", start, end, Path("fixture.wav"), 1), transcript)
    assert transcript == before
    return result, observed


@pytest.mark.parametrize("start,end,timestamp,active,decision", [
    (0, 1200, "05:00", 850, CoverageDecision.RETRY),
    (0, 1200, "15:00", 0, CoverageDecision.WARNING),
    (0, 1200, "19:47", 0, CoverageDecision.PASS),
    (2400, 3590.635, "44:56", 800, CoverageDecision.RETRY),
    (0, 1200, None, 1100, CoverageDecision.RETRY),
    (0, 1200, None, 0, CoverageDecision.WARNING),
])
def test_coverage_policy_preserves_absolute_timestamp(start, end, timestamp, active, decision):
    result, observed = coverage(start, end, timestamp, active)
    assert result.decision == decision
    if start == 2400:
        assert result.tail_gap_seconds == pytest.approx(894.635)
        assert observed[0][0] == 296  # Relative seek into the physical slice only.


@pytest.mark.parametrize("persistent", [False, True])
def test_stop_with_active_tail_cannot_commit_and_rebuilds(tmp_path, monkeypatch, persistent):
    analyzed = []
    def setup(config):
        def activity(self, path, offset, duration, threshold):
            analyzed.append((path, offset, duration))
            return duration if persistent or len(analyzed) == 1 else 0
        monkeypatch.setattr(TailActivityAnalyzer, "analyze", activity)
        original = GeminiClient.generate_transcription
        def generate(self, **kwargs):
            self.last_response_metadata = {"finish_reason": "STOP", "output_tokens": 100}
            return original(self, **kwargs)
        monkeypatch.setattr(GeminiClient, "generate_transcription", generate)
    result, slices, uploads, calls, checkpoint = run_audio_case(tmp_path, monkeypatch,
        setup=setup, fail_count=0, attempts=2)
    assert calls[0][0].bounds == [0, 600]
    assert calls[1][0].bounds == [0, 300]
    assert calls[0][0].name != calls[1][0].name
    assert calls[0][0].path != calls[1][0].path
    if persistent:
        assert result == 1
        assert checkpoint["next_audio_start_us"] == 0
        assert checkpoint["block_metrics"] == []
        assert checkpoint["confirmed_segments"] == []
        assert "COVERAGE_FAILURE" in checkpoint["error_message"]
    else:
        assert result == 0
        assert calls[2][0].bounds[0] == 300
        assert checkpoint["block_metrics"][0]["coverage"]["decision"] == "COVERAGE_WARNING"


@pytest.mark.parametrize("active", [False, True])
def test_real_ffmpeg_tail_activity(tmp_path, active):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        # Alternating samples form non-silent audio; this test does not claim speech.
        audio.writeframes((b"\x00\x20\x00\xe0" if active else b"\x00\x00\x00\x00") * 8000 * 2)
    duration = TailActivityAnalyzer().analyze(path, 1, 3, -40)
    assert duration == pytest.approx(3 if active else 0, abs=0.1)


def test_analysis_error_blocks_instead_of_becoming_silence(tmp_path):
    config = SimpleNamespace(coverage_tail_gap_threshold_sec=120, coverage_active_tail_min_sec=30,
                             coverage_silence_threshold_db=-40, coverage_shrink_factor=0.5)
    result = CoverageValidator(config).validate(SourceBlock(1, "001", 0, 600, tmp_path / "missing.wav", 1),
        TranscriptionBlockResult("1.0", "j", "s", "b", 1, 1, "CONFIRMED", [TranscriptSegment(1, "Vâng.", "00:00")]))
    assert result.decision == CoverageDecision.FAIL


@pytest.mark.parametrize("fallback", [True, False])
def test_actual_non_lite_fallback_or_model_lock_through_pipeline(tmp_path, monkeypatch, fallback):
    sdk_calls = []
    real_generate = GeminiClient.generate_transcription
    real_validate = TranscriptValidator.validate_block_result
    def setup(config):
        monkeypatch.setattr(TranscriptValidator, "validate_block_result", real_validate)
        monkeypatch.setattr("src.main.load_config", lambda **kwargs: replace(config, model_fallback_enabled=fallback))
        def init(self, model_name, fallback_enabled, **kwargs):
            self.model_name = self.requested_model = model_name
            self.fallback_enabled = fallback_enabled
            self.fallback_models = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.8-flash")
            self.max_retries = 1
            self.initial_delay_seconds = 0
            def generate_content(model, contents, **kwargs):
                sdk_calls.append(model)
                if model != "gemini-3.5-flash":
                    raise ConnectionError("503 unavailable")
                prompt = contents[1]
                identity = {k: re.search(r'- ' + k + r': "([^"]+)"', prompt).group(1)
                            for k in ("job_id", "session_id", "block_id")}
                text = json.dumps(dict(schema_version="1.0", **identity, first_source_index=1,
                    last_source_index=1, status="CONFIRMED", segments=[dict(source_index=1, text="Vâng.", timestamp="00:01")]))
                return SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP",
                    content=SimpleNamespace(parts=[SimpleNamespace(text=text, thought=False)]))], usage_metadata=None)
            self.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        monkeypatch.setattr(GeminiClient, "__init__", init)
        monkeypatch.setattr(GeminiClient, "generate_transcription", real_generate)
    result, _, _, _, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup, total_duration=60, attempts=1)
    if fallback:
        assert result == 0
        metadata = checkpoint["block_metrics"][0]["provider_metadata"]
        assert metadata["actual_model"] == "gemini-3.5-flash"
        assert metadata["requested_model"] == sdk_calls[0]
        assert metadata["fallback_used"] is True
        assert "503" in metadata["fallback_reason"]
        assert metadata["attempt"] == len(sdk_calls)
        assert metadata["block_id"] == "BLOCK_001"
    else:
        assert result == 1
        assert len(sdk_calls) == 1
        assert "lite" not in sdk_calls[0]
        assert not checkpoint["confirmed_segments"]


@pytest.mark.parametrize("completed", [False, True])
def test_unverified_historical_coverage_cannot_bypass_gate(tmp_path, monkeypatch, completed):
    from tests.test_stage5_resume_identity import seed
    from src.checkpoint import CheckpointStatus
    original = []
    def setup(config):
        manager, _ = seed(config.checkpoint_file_path, config.audio_dir / "source.wav",
                          duration=900, ends=(900,) if completed else (360,))
        manager.current_data.block_metrics[0].pop("coverage")
        if completed:
            manager.current_data.status = CheckpointStatus.COMPLETED
        manager.save()
        original.append(config.checkpoint_file_path.read_bytes())
    result, slices, uploads, calls, _ = run_audio_case(tmp_path, monkeypatch, setup=setup)
    assert result == 1
    assert not slices and not uploads and not calls
    assert (tmp_path / "state" / "checkpoint.json").read_bytes() == original[0]
