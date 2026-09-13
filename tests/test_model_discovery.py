from types import SimpleNamespace

import pytest
from google.genai.types import Model

from src.gemini_client import GeminiClient, GeminiClientError, GeminiTranscribeError
from src.model_policy import PREFERRED_MODEL_CHAIN, ModelDisallowedError, is_model_allowed
# Legacy generateContent-only fake SDK: Transcribe endpoint is absent here.
PREFERRED_MODEL_CHAIN = tuple(m for m in PREFERRED_MODEL_CHAIN if m != "gemini-3.5-transcribe")
from tests.test_stage43_adaptation import ProviderError


def make_client(monkeypatch, available=PREFERRED_MODEL_CHAIN, effects=(), **kwargs):
    calls, assets, waits = [], [], []
    remaining = iter(effects)
    def generate_content(model, contents, **config):
        calls.append(model)
        assets.append(contents[0])
        effect = next(remaining)
        if isinstance(effect, Exception):
            raise effect
        return SimpleNamespace(text="{}", candidates=None, usage_metadata=None)
    models = SimpleNamespace(
        list=lambda: iter(Model(name="models/" + m, supported_actions=["generateContent"]) for m in available),
        generate_content=generate_content)
    monkeypatch.setattr("src.gemini_client.genai.Client", lambda **kw: SimpleNamespace(models=models))
    monkeypatch.setattr("src.gemini_client.time.sleep", waits.append)
    return GeminiClient("mock-key", **kwargs), calls, assets, waits


@pytest.mark.parametrize("missing", [None, "gemini-3.7-flash", "gemini-3.8-flash"])
def test_discovery_order_and_skip(monkeypatch, missing, capsys):
    available = [m for m in reversed(PREFERRED_MODEL_CHAIN) if m != missing]
    obj, calls, _, _ = make_client(monkeypatch, available + ["gemini-3.5-flash-lite"])
    assert obj.effective_model_chain == tuple(m for m in PREFERRED_MODEL_CHAIN if m != missing)
    assert obj.model_name == obj.effective_model_chain[0]
    assert not calls
    assert "[MODEL_POLICY] lite_allowed=false" in capsys.readouterr().out


def test_real_selection_retry_and_same_asset(monkeypatch):
    obj, calls, assets, waits = make_client(monkeypatch, effects=[ProviderError(delay="59s")] + [ProviderError(503, None)] * 3 + [True, True])
    asset = object()
    obj.generate_transcription(asset, "verbatim")
    obj.generate_transcription(asset, "verbatim")
    assert calls == [PREFERRED_MODEL_CHAIN[0]] + [PREFERRED_MODEL_CHAIN[1]] * 3 + [PREFERRED_MODEL_CHAIN[2]] * 2
    assert all(a is asset for a in assets)
    assert waits == [1, 2]
    assert all(is_model_allowed(m) for m in calls)


def test_all_fail_never_lite(monkeypatch):
    obj, calls, _, waits = make_client(monkeypatch, effects=[ProviderError(delay="59s")] * 4)
    with pytest.raises(GeminiTranscribeError, match="NO_ELIGIBLE_NON_LITE_MODEL"):
        obj.generate_transcription("asset", "verbatim")
    assert calls == list(PREFERRED_MODEL_CHAIN)
    assert waits == []


@pytest.mark.parametrize("available", [[], ["gemini-3.5-flash-lite"], ["other-model"]])
def test_empty_chain_fails_before_generation(monkeypatch, available):
    with pytest.raises(GeminiClientError, match="NO_ELIGIBLE_MODEL"):
        make_client(monkeypatch, available)


def test_lite_opt_in_cannot_bypass_ban(monkeypatch):
    monkeypatch.setattr("src.gemini_client.genai.Client", lambda **kw: pytest.fail("SDK constructed for Lite"))
    with pytest.raises(ModelDisallowedError, match="LITE_MODEL_DISABLED"):
        GeminiClient("mock-key", model_name="gemini-FLASH-LITE", allow_lite_models=True)
    assert not is_model_allowed("gemini-flash-lite", True)


def test_unsupported_generation_excluded():
    from src.model_policy import effective_model_chain
    assert effective_model_chain([Model(name="models/gemini-3.8-flash", supported_actions=["countTokens"])]) == ()


def test_lock_uses_only_discovered_requested_model(monkeypatch):
    obj, calls, _, _ = make_client(monkeypatch, model_name="gemini-3.6-flash", fallback_policy="MODEL_LOCK", effects=[True])
    obj.generate_transcription("asset", "verbatim")
    assert calls == ["gemini-3.6-flash"]
    with pytest.raises(GeminiTranscribeError, match="not discovered"):
        obj._call_model("gemini-3.8-flash", "asset", "prompt", None)


def test_discovery_error_fails_closed(monkeypatch):
    def fail():
        raise ConnectionError("discovery unavailable")
    monkeypatch.setattr("src.gemini_client.genai.Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(list=fail)))
    with pytest.raises(GeminiClientError, match="MODEL_DISCOVERY_FAILED"):
        GeminiClient("mock-key")
