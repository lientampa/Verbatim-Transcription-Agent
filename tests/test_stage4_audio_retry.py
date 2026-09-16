"""Real orchestration, slicing command construction, and validators; fake I/O."""
import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest

from src.config import load_config
from src.main import run_pipeline


def run_audio_case(tmp_path, monkeypatch, failure="output limit", attempts=3, minimum=60,
                   fail_count=1, fail_block="BLOCK_001", restart=False, setup=None, total_duration=900.0):
    # Legacy/native retry contract harness; capability selection has independent tests.
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "source.wav").write_bytes(b"source")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "system_prompt.txt").write_text("Verbatim", encoding="utf-8")
    (tmp_path / "schemas").mkdir()
    schema = Path(__file__).resolve().parents[1] / "schemas" / "transcription_result.schema.json"
    (tmp_path / "schemas" / schema.name).write_bytes(schema.read_bytes())
    (tmp_path / ".env").write_text("GEMINI_API_KEY=mock-key\n", encoding="utf-8")
    config = replace(load_config(tmp_path), strict_speaker_format=False, next_target_policy="LEARNED", model_health_strike_limit=10, initial_target_input_tokens=1500,
                     min_block_tokens=10, min_duration_seconds=minimum,
                     max_duration_seconds=900, validator_max_retries=attempts, coverage_max_generations=attempts,
                     block_shrink_factor=0.6, retry_initial_delay_seconds=0)
    monkeypatch.setattr("src.main.load_config", lambda **kwargs: config)
    monkeypatch.setattr("src.block_builder._find_binary", lambda name: name)
    monkeypatch.setattr("src.block_builder.AudioBlockBuilder.get_duration", lambda self: total_duration)
    slices, uploads, calls = [], [], []
    from src.transcript_validator import TranscriptValidator
    original_validate = TranscriptValidator.validate_block_result
    def validate(self, block_result, **kwargs):
        expected = kwargs["expected_context"]
        assert [expected.audio_start, expected.audio_end] == calls[-1][0].bounds
        return original_validate(self, block_result, **kwargs)
    monkeypatch.setattr(TranscriptValidator, "validate_block_result", validate)

    def ffmpeg(cmd, **kwargs):
        start = float(cmd[cmd.index("-ss") + 1])
        duration = float(cmd[cmd.index("-t") + 1])
        path = Path(cmd[-1])
        path.write_text(json.dumps([start, start + duration]))
        slices.append((start, start + duration, path))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("src.block_builder.subprocess.run", ffmpeg)
    # This harness writes boundary JSON instead of audio; model its tail as silent.
    # Coverage-specific tests override this analyzer with explicit activity fixtures.
    monkeypatch.setattr("src.coverage_validator.TailActivityAnalyzer.analyze", lambda self, *args: 0.0)
    monkeypatch.setattr("src.main.GeminiClient.__init__", lambda self, **kwargs: None)
    monkeypatch.setattr("src.main.GeminiClient.refresh_audio_upload", lambda self, file, path: file)

    def upload(self, audio_path):
        item = SimpleNamespace(name=f"upload-{len(uploads)}", bounds=json.loads(audio_path.read_text()), path=audio_path)
        uploads.append(item)
        return item

    def generate(self, gemini_file, user_prompt, **kwargs):
        calls.append((gemini_file, user_prompt))
        first = int(re.search(r"- first_source_index: (\d+)", user_prompt).group(1))
        identity = {name: re.search(r'- ' + name + r': "([^"]+)"', user_prompt).group(1)
                    for name in ("job_id", "session_id", "block_id")}
        failed_attempts = sum(f'- block_id: "{fail_block}"' in p for _, p in calls)
        should_fail = identity["block_id"] == fail_block and failed_attempts <= fail_count
        if should_fail:
            state = json.loads(config.checkpoint_file_path.read_text(encoding="utf-8"))
            assert len(state["confirmed_segments"]) == (0 if fail_block == "BLOCK_001" else 1)
            if failure not in ("structural", "fidelity"):
                raise RuntimeError(failure)
            if failure == "structural":
                identity["block_id"] = "BLOCK_999"
        sec = int(gemini_file.bounds[0])
        return json.dumps(dict(schema_version="1.0", **identity, first_source_index=first,
                               last_source_index=first, status="CONFIRMED",
                               segments=[dict(source_index=first,
                                              text="chúng ta cần xử lý phần này " * 5 if should_fail and failure == "fidelity" else "Vâng.",
                                              timestamp=f"{sec // 60:02d}:{sec % 60:02d}")]))

    monkeypatch.setattr("src.main.GeminiClient.upload_audio", upload)
    monkeypatch.setattr("src.main.GeminiClient.generate_transcription", generate)
    if setup:
        setup(config)
    if restart:
        from src.checkpoint import CheckpointManager
        original_commit = CheckpointManager.commit_block
        def interrupt(self, **kwargs):
            value = original_commit(self, **kwargs)
            raise KeyboardInterrupt("simulated process interruption after atomic commit")
        monkeypatch.setattr(CheckpointManager, "commit_block", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_pipeline(base_dir=tmp_path, prompt_controlled_audio=False)
        monkeypatch.setattr(CheckpointManager, "commit_block", original_commit)
        calls.clear()
        slices.clear()
        uploads.clear()
        fail_count = 0
    result = run_pipeline(base_dir=tmp_path, prompt_controlled_audio=False)
    checkpoint = json.loads(config.checkpoint_file_path.read_text(encoding="utf-8"))
    return result, slices, uploads, calls, checkpoint


def test_restart_after_shrink_begins_at_confirmed_boundary(tmp_path, monkeypatch):
    result, slices, uploads, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, restart=True)
    assert result == 0
    assert slices[0][0] == 360
    assert calls[0][0].bounds[0] == 360
    assert '- block_id: "BLOCK_002"' in calls[0][1]


def test_size_failure_rebuilds_and_continues_from_smaller_end(tmp_path, monkeypatch):
    result, slices, uploads, calls, checkpoint = run_audio_case(tmp_path, monkeypatch)
    assert result == 0
    assert calls[0][0].bounds == [0, 600]
    assert calls[1][0].bounds == [0, 360]
    assert calls[0][0].name != calls[1][0].name
    assert calls[0][0].path != calls[1][0].path
    assert calls[2][0].bounds[0] == 360
    assert slices[1][:2] == (0, 360)
    assert checkpoint["status"] == "COMPLETED"
    metrics = checkpoint["block_metrics"]
    assert metrics[0]["actual_end_offset"] == 360
    assert metrics[0]["attempt_count"] == 2
    assert all(a["actual_end_offset"] == b["actual_start_offset"] for a, b in zip(metrics, metrics[1:]))
    assert metrics[-1]["actual_end_offset"] == 900
    assert "end=360" in calls[1][1]


@pytest.mark.parametrize("failure", ["network timeout", "503 temporarily unavailable", "structural", "fidelity", "invalid json"])
def test_non_size_failure_reuses_same_boundaries_and_upload(tmp_path, monkeypatch, failure):
    result, slices, uploads, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, failure=failure)
    assert result == 0
    assert calls[0][0] is calls[1][0]
    assert calls[1][0].bounds == [0, 600]
    assert len([s for s in slices if s[0] == 0]) == 1
    assert checkpoint["block_metrics"][0]["generation"] == 1


@pytest.mark.parametrize("failure", ["CONTEXT_LIMIT", "input too large", "OUTPUT_TRUNCATED"])
def test_all_explicit_size_failures_rebuild(tmp_path, monkeypatch, failure):
    result, _, _, calls, _ = run_audio_case(tmp_path, monkeypatch, failure=failure)
    assert result == 0
    assert calls[1][0].bounds == [0, 360]


def test_later_slice_keeps_absolute_offset_and_tail_continuity(tmp_path, monkeypatch):
    result, _, _, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, fail_block="BLOCK_002")
    assert result == 0
    assert calls[1][0].bounds == [600, 900]
    assert calls[2][0].bounds == [600, 780]
    assert calls[3][0].bounds[0] == 780
    assert "start=600" in calls[2][1] and "end=780" in calls[2][1]


@pytest.mark.parametrize("minimum,attempts,expected_calls,reason", [
    (360, 10, 6, "EMERGENCY_SIZE_FLOOR_EXHAUSTED"),
    (60, 2, 2, "SIZE_RETRIES_EXHAUSTED"),
])
def test_exhaustion_is_bounded_and_uncommitted(tmp_path, monkeypatch, minimum, attempts, expected_calls, reason):
    result, slices, _, calls, checkpoint = run_audio_case(tmp_path, monkeypatch,
        minimum=minimum, attempts=attempts, fail_count=100)
    assert result == 1
    assert len(calls) == expected_calls
    assert reason in checkpoint["error_message"]
    assert checkpoint["last_confirmed_source_index"] is None
    assert checkpoint["next_source_index"] == 1
    assert checkpoint["confirmed_segments"] == []
    assert checkpoint["block_metrics"] == []
    assert all(not path.exists() for _, _, path in slices)


def test_real_ffmpeg_creates_smaller_file_and_bypasses_stale_cache(tmp_path):
    import wave
    from src.block_builder import AudioBlockBuilder
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 8000 * 6)
    builder = AudioBlockBuilder(source, cache_dir=tmp_path / "cache", job_id="owned")
    stale = builder.cache_dir / "BLOCK_001.wav"
    stale.write_bytes(source.read_bytes())
    original = builder.slice_time_range("BLOCK_001", 0, 6)
    rebuilt = builder.slice_time_range("BLOCK_001", 0, 3.6)
    assert original != rebuilt and rebuilt != stale
    with wave.open(str(original), "rb") as audio:
        old_duration = audio.getnframes() / audio.getframerate()
    with wave.open(str(rebuilt), "rb") as audio:
        new_duration = audio.getnframes() / audio.getframerate()
    assert old_duration == pytest.approx(6, abs=0.1)
    assert new_duration == pytest.approx(3.6, abs=0.1)
    assert rebuilt.stat().st_size < original.stat().st_size


@pytest.mark.parametrize("finish_reason", ["MAX_TOKENS", "STOP"])
def test_provider_finish_reason_and_usage_preserved_without_fallback(finish_reason):
    from google.genai import types
    from src.gemini_client import GeminiClient, GeminiSizeError
    from src.failure_classifier import classify_failure, FailureType
    client = GeminiClient.__new__(GeminiClient)
    client.model_name = "test-model"
    client.max_retries = 3
    responses = []
    def generate(**kwargs):
        responses.append(kwargs)
        return types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason=finish_reason,
                content=types.Content(parts=[types.Part(text='{"segments": []}')]))],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=120, candidates_token_count=45, total_token_count=165))
    client.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    if finish_reason == "MAX_TOKENS":
        with pytest.raises(GeminiSizeError) as failure:
            client.generate_transcription("asset", "verbatim")
        assert classify_failure(failure.value) == FailureType.SIZE_FAILURE
        assert failure.value.metadata["output_tokens"] == 45
    else:
        assert client.generate_transcription("asset", "verbatim") == '{"segments": []}'
    assert len(responses) == 1
    assert {k: client.last_response_metadata[k] for k in ("input_tokens", "output_tokens", "total_tokens", "finish_reason")} == dict(input_tokens=120, output_tokens=45,
                                                 total_tokens=165, finish_reason=finish_reason)
    assert client.last_response_metadata["actual_model"] == "test-model"
    assert client.last_response_metadata["fallback_used"] is False


def test_provider_absent_usage_is_not_fabricated():
    from google.genai import types
    from src.gemini_client import GeminiClient
    client = GeminiClient.__new__(GeminiClient)
    response = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(text="{}")]), finish_reason="STOP")])
    client.client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response))
    assert client._call_model("model", "asset", "prompt", None) == "{}"
    assert client.last_response_metadata["input_tokens"] is None
    assert client.last_response_metadata["output_tokens"] is None
