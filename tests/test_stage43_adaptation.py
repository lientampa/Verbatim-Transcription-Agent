import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.config import load_config
from src.block_builder import DabbAudioOrchestrator
from src.coverage_validator import CoverageDecision, CoverageResult, TailActivityAnalyzer
from src.gemini_client import GeminiClient, GeminiTranscribeError, structured_retry_after
from tests.test_stage4_audio_retry import run_audio_case


@pytest.mark.parametrize("safe", [600, 300])
def test_real_cross_block_learning_with_separate_budgets(tmp_path, monkeypatch, safe):
    def setup(config):
        config = replace(config, initial_target_input_tokens=3000, max_duration_seconds=1200,
                         min_duration_seconds=300, validator_max_retries=1, coverage_max_generations=3)
        monkeypatch.setattr("src.main.load_config", lambda **kwargs: config)
        original = GeminiClient.generate_transcription
        def generate(self, gemini_file, **kwargs):
            data = json.loads(original(self, gemini_file=gemini_file, **kwargs))
            start, end = gemini_file.bounds
            timestamp = int(end - 8 if end - start <= safe else start + 1)
            data["segments"][0]["timestamp"] = f"{timestamp // 60:02d}:{timestamp % 60:02d}"
            self.last_response_metadata = {"actual_model": config.gemini_model, "finish_reason": "STOP"}
            return json.dumps(data)
        monkeypatch.setattr(GeminiClient, "generate_transcription", generate)
        monkeypatch.setattr(TailActivityAnalyzer, "analyze", lambda self, path, offset, duration, threshold: duration)
    result, _, _, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup,
        fail_count=0, total_duration=1200)
    assert result == 0
    durations = [c[0].bounds[1] - c[0].bounds[0] for c in calls]
    expected = [1200, 600] if safe == 600 else [1200, 600, 300]
    assert durations[:len(expected)] == expected
    assert durations[len(expected)] == safe
    assert checkpoint["block_metrics"][0]["validation_attempt"] == 1
    assert checkpoint["block_metrics"][0]["coverage_generation"] == len(expected)
    assert checkpoint["next_audio_start_us"] == 1200_000000


@pytest.fixture
def planner(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock-key")
    cfg = replace(load_config(tmp_path), gemini_model="Flash", next_target_policy="LEARNED", initial_target_input_tokens=3000,
                  max_duration_seconds=1200, min_duration_seconds=120)
    return DabbAudioOrchestrator(tmp_path / "audio.wav", cfg)


def passed():
    return CoverageResult(CoverageDecision.PASS, "09:59", 1, "NOT_ANALYZED_SMALL_GAP")


def test_growth_is_slow_and_failure_overrides_it(planner):
    planner.on_coverage_failure(1200)
    for _ in range(3):
        planner.on_coverage_success(600, passed())
    assert planner.current_block_duration_seconds == pytest.approx(660)
    planner.on_coverage_failure(660)
    assert planner.current_block_duration_seconds == pytest.approx(330)
    assert planner.model_profiles["Flash"]["streak"] == 0


def test_profiles_are_isolated_and_unknown_model_is_conservative(planner):
    planner.on_coverage_success(600, passed())
    assert planner.select_model("Lite", "quota") == 300
    planner.on_coverage_failure(600)
    planner.on_coverage_failure(300)
    assert planner.current_block_duration_seconds == 150
    assert planner.model_profiles["Flash"]["coverage_failures"] == 0
    assert planner.select_model("Flash") == 600
    from src.failure_classifier import FailureType
    flash_tokens = planner.dabb.current_target_tokens
    planner.select_model("Lite")
    planner.on_block_failure(FailureType.SIZE_FAILURE)
    planner.select_model("Flash")
    assert planner.dabb.current_target_tokens == flash_tokens


def test_minimum_failure_never_commits(tmp_path, monkeypatch):
    def setup(config):
        monkeypatch.setattr("src.main.load_config", lambda **kwargs: replace(config,
            min_duration_seconds=300, coverage_max_generations=4, validator_max_retries=1))
        monkeypatch.setattr(TailActivityAnalyzer, "analyze", lambda self, path, offset, duration, threshold: duration)
    result, _, _, calls, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup,
        fail_count=0, total_duration=600)
    assert result == 1
    assert [c[0].bounds for c in calls] == [[0, 600], [0, 300]]
    assert checkpoint["next_audio_start_us"] == 0
    assert not checkpoint["block_metrics"]
    assert "COVERAGE_UNRESOLVED" in checkpoint["error_message"]


class ProviderError(ConnectionError):
    def __init__(self, code=429, delay="19s"):
        self.code = code
        self.details = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                                "retryDelay": delay}]}} if delay is not None else {}
        super().__init__(str(code))


def client(policy, effects, monkeypatch, maximum=30):
    obj = GeminiClient.__new__(GeminiClient)
    obj.model_name = obj.requested_model = "A"
    obj.fallback_enabled = True
    obj.fallback_policy = policy
    obj.fallback_models = ("B", "A", "C", "B")
    obj.max_backoff_seconds = maximum
    obj.max_retries = 2
    obj.initial_delay_seconds = 0.1
    calls, waits = [], []
    def generate_content(model, **kwargs):
        calls.append(model)
        effect = effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP",
            content=SimpleNamespace(parts=[SimpleNamespace(text="{}", thought=False)]))], usage_metadata=None)
    obj.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr("src.gemini_client.time.sleep", waits.append)
    return obj, calls, waits


@pytest.mark.parametrize("delay,expected", [("19s", 19), ({"seconds": "19", "nanos": 500000000}, 19.5)])
def test_retryinfo_structural(delay, expected):
    assert structured_retry_after(ProviderError(delay=delay)) == expected


def test_short_quota_wait_stays_on_primary(monkeypatch):
    obj, calls, waits = client("PREFER_WAIT", [ProviderError(), True], monkeypatch)
    assert obj.generate_transcription("asset", "verbatim") == "{}"
    assert calls == ["A", "A"] and waits == [19]
    assert obj.call_history[0]["retry_after_seconds"] == 19
    assert obj.last_response_metadata["provider_attempt"] == 2
    assert not obj.last_response_metadata["fallback_used"]


@pytest.mark.parametrize("policy,maximum", [("PREFER_FALLBACK", 30), ("PREFER_WAIT", 10)])
def test_fallback_without_wait_when_selected_or_delay_exceeds_limit(monkeypatch, policy, maximum):
    obj, calls, waits = client(policy, [ProviderError(), True], monkeypatch, maximum)
    obj.generate_transcription("asset", "verbatim")
    assert calls == ["A", "B"] and waits == []
    assert obj.last_response_metadata["actual_model"] == "B"
    assert obj.last_response_metadata["fallback_reason"] == "429"


def test_locked_model_exhausts_own_budget(monkeypatch):
    obj, calls, waits = client("LOCK_MODEL", [ProviderError(), ProviderError()], monkeypatch)
    with pytest.raises(GeminiTranscribeError):
        obj.generate_transcription("asset", "verbatim")
    assert calls == ["A", "A"] and waits == [19]


def test_503_then_deterministic_fallback_no_reinsertion(monkeypatch):
    obj, calls, waits = client("PREFER_WAIT", [ProviderError(503, None)] * 4 + [True, True], monkeypatch)
    obj.generate_transcription("asset", "verbatim")
    obj.generate_transcription("asset2", "verbatim")
    assert calls == ["A", "A", "B", "B", "C", "C"]
    assert waits == [0.1, 0.1]
    assert obj.last_response_metadata["requested_model"] == "A"


@pytest.mark.parametrize("available", [True, False])
def test_real_provider_path_replans_before_unknown_model_call(tmp_path, monkeypatch, available):
    sdk_calls = []
    real_generate = GeminiClient.generate_transcription
    def setup(config):
        cfg = replace(config, gemini_model="A", provider_fallback_policy="PREFER_FALLBACK",
                      validator_max_retries=3, unknown_model_duration_sec=300)
        monkeypatch.setattr("src.main.load_config", lambda **kwargs: cfg)
        harness_generate = GeminiClient.generate_transcription
        def init(self, **kwargs):
            self.model_name = self.requested_model = "A"
            self.fallback_enabled = True
            self.fallback_policy = "PREFER_FALLBACK"
            self.fallback_models = ("B",)
            self.max_retries = 2
            self.initial_delay_seconds = 0.1
            self.max_backoff_seconds = 30
            def generate_content(model, contents, **kwargs):
                asset, prompt = contents
                sdk_calls.append((model, asset.bounds[:]))
                if model == "A" or not available:
                    raise ProviderError(503, None)
                data = json.loads(harness_generate(self, gemini_file=asset, user_prompt=prompt))
                timestamp = int(asset.bounds[1]) - 8
                data["segments"][0]["timestamp"] = f"{timestamp // 60:02d}:{timestamp % 60:02d}"
                return SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(
                    parts=[SimpleNamespace(text=json.dumps(data), thought=False)]))], usage_metadata=None)
            self.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        monkeypatch.setattr(GeminiClient, "__init__", init)
        monkeypatch.setattr(GeminiClient, "generate_transcription", real_generate)
    result, _, uploads, _, checkpoint = run_audio_case(tmp_path, monkeypatch, setup=setup,
        fail_count=0, total_duration=600)
    assert sdk_calls[:2] == [("A", [0, 600]), ("B", [0, 300])]
    assert uploads[0].path != uploads[1].path
    if available:
        assert result == 0
        first = checkpoint["block_metrics"][0]
        assert first["provider_metadata"]["actual_model"] == "B"
        assert first["provider_metadata"]["provider_attempt"] == 2
        assert first["physical_generation"] == 2
        assert first["validation_attempt"] == first["coverage_generation"] == 1
    else:
        assert result == 1 and len(sdk_calls) == 2
        assert not checkpoint["block_metrics"]
        assert checkpoint["next_audio_start_us"] == 0
