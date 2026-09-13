from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from src.config import load_config
from src.gemini_client import GeminiClient, GeminiTranscribeError, retry_decision
from tests.test_stage43_adaptation import client, ProviderError
from tests.test_stage4_audio_retry import run_audio_case


def fast_client(monkeypatch, effects, policy="SPEED_FIRST", wait=5, budget=2):
    obj,calls,waits=client(policy,effects,monkeypatch,wait)
    obj.initial_delay_seconds=1
    obj.max_transient_retries=budget
    obj.retry_max_delay_seconds=4
    obj.backoff_multiplier=2
    return obj,calls,waits


def test_initial_one_and_bounded_exponential(monkeypatch):
    obj,calls,waits=fast_client(monkeypatch,[ProviderError(503,None)]*4+[True],budget=4)
    obj.generate_transcription("same-asset","verbatim")
    assert waits==[1,2,4,4]
    assert calls==["A"]*5


def test_three_503_then_fallback(monkeypatch):
    obj,calls,waits=fast_client(monkeypatch,[ProviderError(503,None)]*3+[True])
    obj.generate_transcription("asset","verbatim")
    assert calls==["A","A","A","B"] and waits==[1,2]


@pytest.mark.parametrize("delay,expected_calls,expected_waits",[(59,["A","B"],[]),(1,["A","A"],[1])])
def test_speed_quota_delay_decisions(monkeypatch,delay,expected_calls,expected_waits):
    obj,calls,waits=fast_client(monkeypatch,[ProviderError(delay=f"{delay}s"),True])
    obj.generate_transcription("asset","verbatim")
    assert calls==expected_calls and waits==expected_waits
    failed=obj.call_history[0]
    assert failed["provider_retry_after_seconds"]==delay
    assert failed["actual_sleep_seconds"]==(1 if delay==1 else 0)
    assert failed["provider_error_details"]


def test_wait_first_honors_bounded_advice(monkeypatch):
    obj,calls,waits=fast_client(monkeypatch,[ProviderError(delay="10s"),True],"WAIT_FIRST",wait=10)
    obj.generate_transcription("asset","verbatim")
    assert calls==["A","A"] and waits==[10]


@pytest.mark.parametrize("policy",["MODEL_LOCK","SPEED_FIRST"])
def test_long_quota_without_alternative_fails_once(monkeypatch,policy):
    obj,calls,waits=fast_client(monkeypatch,[ProviderError(delay="59s")],policy)
    obj.fallback_models=()
    with pytest.raises(GeminiTranscribeError,match="PROVIDER_UNAVAILABLE"):
        obj.generate_transcription("asset","verbatim")
    assert calls==["A"] and not waits


def test_model_lock_never_calls_configured_fallback(monkeypatch):
    obj,calls,waits=fast_client(monkeypatch,[ProviderError(503,None)]*3,"MODEL_LOCK")
    with pytest.raises(GeminiTranscribeError):
        obj.generate_transcription("asset","verbatim")
    assert calls==["A"]*3 and waits==[1,2]


def test_config_one_and_alias_precedence(tmp_path,monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY","mock-key")
    monkeypatch.setenv("RETRY_INITIAL_DELAY_SECONDS","1")
    monkeypatch.setenv("PROVIDER_FALLBACK_POLICY","PREFER_WAIT")
    monkeypatch.setenv("PROVIDER_RETRY_POLICY","SPEED_FIRST")
    cfg=load_config(tmp_path)
    assert cfg.retry_initial_delay_seconds==1 and cfg.provider_fallback_policy=="SPEED_FIRST"


def test_upload_uses_same_bounded_decision_without_hidden_quota_minimum(tmp_path,monkeypatch):
    obj,_,waits=fast_client(monkeypatch,[])
    obj.max_retries=3
    obj.client.files=SimpleNamespace()
    attempts=[]
    def upload(file):
        attempts.append(file)
        if len(attempts)==1:
            raise ProviderError(503,None)
        return SimpleNamespace(name="uploaded",state=None)
    obj.client.files.upload=upload
    path=tmp_path/'audio.wav'
    path.write_bytes(b'fixture')
    assert obj.upload_audio(path).name=="uploaded"
    assert attempts==[str(path)]*2 and waits==[1]


@pytest.mark.parametrize("succeeds",[True,False])
def test_pipeline_provider_retries_preserve_asset_and_counters(tmp_path,monkeypatch,succeeds):
    real_generate=GeminiClient.generate_transcription
    attempts,waits=[],[]
    def setup(config):
        cfg=replace(config,next_target_policy="MAX_FIRST",provider_fallback_policy="SPEED_FIRST")
        monkeypatch.setattr("src.main.load_config",lambda **kwargs:cfg)
        harness_generate=GeminiClient.generate_transcription
        def init(self,**kwargs):
            self.model_name=self.requested_model="A"
            self.fallback_models=("B",)
            self.fallback_enabled=True
            self.fallback_policy="SPEED_FIRST"
            self.max_retries=3
            self.max_transient_retries=2
            self.initial_delay_seconds=1
            self.max_backoff_seconds=5
            def generate_content(model,contents,**kwargs):
                asset,prompt=contents
                attempts.append((model,asset.path,asset.bounds[:]))
                if len(attempts)==1: raise ProviderError(503,None)
                if model=="A" or not succeeds: raise ProviderError(delay="59s")
                text=harness_generate(self,gemini_file=asset,user_prompt=prompt)
                return SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP",content=SimpleNamespace(
                    parts=[SimpleNamespace(text=text,thought=False)]))],usage_metadata=None)
            self.client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        monkeypatch.setattr(GeminiClient,"__init__",init)
        monkeypatch.setattr(GeminiClient,"generate_transcription",real_generate)
        monkeypatch.setattr("src.gemini_client.time.sleep",waits.append)
    result,slices,uploads,_,cp=run_audio_case(tmp_path,monkeypatch,setup=setup,fail_count=0,total_duration=600)
    assert [a[0] for a in attempts]==["A","A","B"]
    assert all(a[1:]==attempts[0][1:] for a in attempts)
    assert len(slices)==len(uploads)==1 and waits==[1]
    if succeeds:
        assert result==0
        metric=cp["block_metrics"][0]
        assert metric["validation_attempt"]==metric["coverage_generation"]==metric["physical_generation"]==1
        assert metric["provider_metadata"]["provider_attempt"]==3
    else:
        assert result==1 and cp["next_audio_start_us"]==0 and not cp["block_metrics"]
