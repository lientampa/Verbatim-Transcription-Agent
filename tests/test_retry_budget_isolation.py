import json
from dataclasses import replace
from types import SimpleNamespace

from google.genai.types import Model

from src.gemini_client import GeminiClient
from src.model_policy import PREFERRED_MODEL_CHAIN
# Legacy generateContent-only fake SDK: Transcribe endpoint is absent here.
PREFERRED_MODEL_CHAIN = tuple(m for m in PREFERRED_MODEL_CHAIN if m != "gemini-3.5-transcribe")
from tests.test_stage43_adaptation import ProviderError
from tests.test_stage4_audio_retry import run_audio_case


def test_production_mixed_failure_chain_keeps_two_size_rebuilds(tmp_path, monkeypatch, capsys, caplog):
    real_init = GeminiClient.__init__
    real_generate = GeminiClient.generate_transcription
    effects = [ProviderError(503, None), ProviderError(delay="59s"), "malformed", "empty",
               ProviderError(delay="59s"), "max", "max"]
    sdk_calls, waits = [], []
    instances = []
    def setup(config):
        config = replace(config, next_target_policy="MAX_FIRST", max_duration_seconds=1200,
                         initial_target_input_tokens=3000, validator_max_retries=3,
                         retry_initial_delay_seconds=1)
        monkeypatch.setattr("src.main.load_config", lambda **kw: config)
        harness_generate = GeminiClient.generate_transcription
        def generate_content(model, contents, **kw):
            asset, prompt = contents
            sdk_calls.append((model, list(asset.bounds)))
            effect = effects.pop(0) if effects else "success"
            if isinstance(effect, Exception):
                raise effect
            if effect == "max":
                return SimpleNamespace(candidates=[SimpleNamespace(finish_reason="MAX_TOKENS",
                    content=SimpleNamespace(parts=[SimpleNamespace(text="incomplete JSON", thought=False)]))],
                    usage_metadata=SimpleNamespace(prompt_token_count=9000, candidates_token_count=8192,
                        total_token_count=18000, thoughts_token_count=808))
            if effect in ("malformed", "empty"):
                return SimpleNamespace(text="{" if effect == "malformed" else "", candidates=None, usage_metadata=None)
            text = harness_generate(instances[-1], gemini_file=asset, user_prompt=prompt)
            data = json.loads(text)
            seconds = int(asset.bounds[1] - 2)
            data["segments"][0]["timestamp"] = f"{seconds//60:02d}:{seconds%60:02d}"
            return SimpleNamespace(text=json.dumps(data), candidates=None, usage_metadata=None)
        sdk = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content,
            list=lambda: [Model(name="models/"+m, supported_actions=["generateContent"]) for m in PREFERRED_MODEL_CHAIN]))
        monkeypatch.setattr("src.gemini_client.genai.Client", lambda **kw: sdk)
        def init(self, **kw):
            real_init(self, **kw)
            instances.append(self)
        monkeypatch.setattr(GeminiClient, "__init__", init)
        monkeypatch.setattr(GeminiClient, "generate_transcription", real_generate)
        monkeypatch.setattr("src.gemini_client.time.sleep", waits.append)
    result, slices, uploads, calls, checkpoint = run_audio_case(tmp_path, monkeypatch,
        fail_count=0, setup=setup, total_duration=1200)
    assert result == 0
    assert [m for m,_ in sdk_calls[:8]] == [PREFERRED_MODEL_CHAIN[0]]*2 + [PREFERRED_MODEL_CHAIN[1]]*2 + [PREFERRED_MODEL_CHAIN[2]] + [PREFERRED_MODEL_CHAIN[3]]*3
    assert [bounds for _,bounds in sdk_calls[5:8]] == [[0,1200],[0,720],[0,432]]
    first = checkpoint["block_metrics"][0]
    assert first["size_retry_count"] == 2
    assert first["structural_retry_count"] == 1
    assert first["validation_attempt_count"] == 2
    assert first["coverage_generation_count"] == 1
    assert first["physical_generation_count"] == 3
    assert first["actual_end_offset"] == 432
    assert first["provider_attempt_count"] == 8
    truncated = [m for m in instances[0].call_history if m.get("finish_reason") == "MAX_TOKENS"]
    assert len(truncated) == 2
    assert all(m["output_tokens"] == 8192 and m["thought_tokens"] == 808 for m in truncated)
    logs = capsys.readouterr().out
    assert "MAX_TOKENS_DIAGNOSTICS" in logs and "[OUTPUT_LIMIT]" in logs
    assert "__proxy__" not in logs and "Block BLOCK_001 encountered SIZE_FAILURE" in caplog.text
    assert "reason=EMPTY_RESPONSE" in logs and "None_RETRY_POLICY" not in logs
    assert all(m["physical_duration"] == 1200 for m in instances[0].call_history[:6] if "physical_duration" in m)
    assert truncated[1]["dabb_target_input_tokens"] == 1800
    assert truncated[1]["provider_usage_metadata"]["candidates_token_count"] == 8192


def test_size_retries_do_not_consume_validation_budget(tmp_path, monkeypatch):
    def setup(config):
        original = GeminiClient.generate_transcription
        count = 0
        def generate(self, **kw):
            nonlocal count
            count += 1
            if count <= 2:
                raise RuntimeError("OUTPUT_TRUNCATED")
            if count <= 4:
                return "{"
            return original(self, **kw)
        monkeypatch.setattr(GeminiClient, "generate_transcription", generate)
    result, _, _, _, cp = run_audio_case(tmp_path, monkeypatch, fail_count=0, setup=setup, attempts=3)
    assert result == 0
    metric = cp["block_metrics"][0]
    assert metric["size_retry_count"] == 2 and metric["validation_attempt_count"] == 3
    assert metric["structural_retry_count"] == 2


def test_coverage_budget_survives_structural_and_size_failures(tmp_path, monkeypatch):
    def setup(config):
        original = GeminiClient.generate_transcription
        count = 0
        def generate(self, **kw):
            nonlocal count
            count += 1
            if count == 1:
                return "{"
            if count == 2:
                raise RuntimeError("OUTPUT_TRUNCATED")
            data = json.loads(original(self, **kw))
            start, end = kw["gemini_file"].bounds
            sec = int(start + 1 if count == 3 else end - 2)
            data["segments"][0]["timestamp"] = f"{sec//60:02d}:{sec%60:02d}"
            return json.dumps(data)
        monkeypatch.setattr(GeminiClient, "generate_transcription", generate)
        monkeypatch.setattr("src.coverage_validator.TailActivityAnalyzer.analyze", lambda self,path,offset,duration,threshold: duration)
    result, _, _, _, cp = run_audio_case(tmp_path, monkeypatch, fail_count=0, setup=setup, attempts=3)
    assert result == 0
    metric = cp["block_metrics"][0]
    assert metric["size_retry_count"] == 1
    assert metric["validation_attempt_count"] == 2
    assert metric["coverage_generation_count"] == 2
    assert metric["physical_generation_count"] == 3
    assert metric["attempt_count"] == 4


def test_explicit_size_limit_means_two_rebuilds_not_two_total_calls(tmp_path, monkeypatch, capsys):
    def setup(config):
        config = replace(config, next_target_policy="MAX_FIRST", max_duration_seconds=1200,
                         initial_target_input_tokens=3000, size_retry_limit=2)
        monkeypatch.setattr("src.main.load_config", lambda **kw: config)
    result, _, uploads, calls, cp = run_audio_case(tmp_path, monkeypatch, setup=setup,
        attempts=1, fail_count=100, total_duration=1200)
    assert result == 1
    assert [c[0].bounds for c in calls] == [[0,1200],[0,720],[0,432]]
    assert len({u.name for u in uploads}) == 3
    assert cp["next_audio_start_us"] == 0 and cp["block_metrics"] == []
    logs = capsys.readouterr().out
    assert "[SIZE_RETRY_EXHAUSTED]" in logs and "size_retry_limit=2" in logs


def test_actual_model_output_cap_is_rebuilt_from_discovery(monkeypatch):
    seen = []
    def generate_content(model, config, **kw):
        seen.append((model, config.max_output_tokens))
        if len(seen) == 1:
            raise ProviderError(delay="59s")
        return SimpleNamespace(text="{}", candidates=None, usage_metadata=None)
    models = [Model(name="models/gemini-3.8-flash", supported_actions=["generateContent"], output_token_limit=8000),
              Model(name="models/gemini-3.5-flash", supported_actions=["generateContent"], output_token_limit=2000)]
    monkeypatch.setattr("src.gemini_client.genai.Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(list=lambda: models, generate_content=generate_content)))
    client = GeminiClient("mock-key", requested_max_output_tokens=4000)
    client.generate_transcription("asset", "verbatim")
    assert seen == [("gemini-3.5-flash",2000),("gemini-3.8-flash",4000)]
    assert client.last_response_metadata["configured_max_output_tokens"] == 4000
    assert client.last_response_metadata["effective_max_output_tokens"] == 4000
    assert client.last_response_metadata["provider_output_token_limit"] == 8000


def test_coverage_then_size_preserves_size_allowance(tmp_path, monkeypatch):
    def setup(config):
        config = replace(config, next_target_policy="MAX_FIRST", max_duration_seconds=600)
        monkeypatch.setattr("src.main.load_config", lambda **kw: config)
        original = GeminiClient.generate_transcription
        count = 0
        def generate(self, **kw):
            nonlocal count
            count += 1
            if count in (2,3):
                raise RuntimeError("MAX_TOKENS")
            data = json.loads(original(self, **kw))
            start, end = kw["gemini_file"].bounds
            sec = int(start+1 if count==1 else end-2)
            data["segments"][0]["timestamp"] = f"{sec//60:02d}:{sec%60:02d}"
            return json.dumps(data)
        monkeypatch.setattr(GeminiClient,"generate_transcription",generate)
        monkeypatch.setattr("src.coverage_validator.TailActivityAnalyzer.analyze", lambda self,path,offset,duration,threshold: duration)
    result, _, _, _, cp = run_audio_case(tmp_path,monkeypatch,setup=setup,fail_count=0)
    assert result == 0
    metric = cp["block_metrics"][0]
    assert metric["size_retry_count"] == 2
    assert metric["coverage_generation_count"] == 2
    assert metric["physical_generation_count"] == 4


def test_size_only_success_commits_third_generation(tmp_path, monkeypatch):
    def setup(config):
        config = replace(config, next_target_policy="MAX_FIRST", max_duration_seconds=1200,
                         initial_target_input_tokens=3000, size_retry_limit=2)
        monkeypatch.setattr("src.main.load_config", lambda **kw: config)
    result, _, uploads, calls, cp = run_audio_case(tmp_path, monkeypatch, setup=setup,
        attempts=1, fail_count=2, total_duration=1200)
    assert result == 0
    assert [c[0].bounds for c in calls[:3]] == [[0,1200],[0,720],[0,432]]
    metric = cp["block_metrics"][0]
    assert metric["actual_end_offset"] == 432
    assert metric["size_retry_count"] == 2 and metric["physical_generation_count"] == 3
