from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.model_policy import allowed_models, is_model_allowed, ModelDisallowedError
from src.config import load_config, ConfigurationError
from src.gemini_client import GeminiClient, GeminiTranscribeError
from src.block_builder import DabbAudioOrchestrator
from src.model_health import ModelHealth, ModelHealthProfile
from tests.test_fast_retry import fast_client
from tests.test_stage43_adaptation import ProviderError


@pytest.mark.parametrize('name',['gemini-x-flash-lite','GEMINI-X-LITE','custom-provider-lite-preview'])
def test_case_insensitive_substring_ban_at_sdk_boundary(name):
    assert not is_model_allowed(name)
    obj=GeminiClient.__new__(GeminiClient)
    obj.client=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs:pytest.fail('SDK called')))
    with pytest.raises(ModelDisallowedError,match='MODEL_DISALLOWED: LITE_MODEL_DISABLED'):
        obj._call_model(name,'asset','prompt',None)


def test_ordered_filter_does_not_reject_unrelated_letters():
    assert allowed_models(['gemini-3.6-flash','gemini-3.8-flash','gemini-3.5-flash-lite'])==['gemini-3.6-flash','gemini-3.8-flash']
    assert is_model_allowed('elite-model')


@pytest.mark.parametrize('locked',[False,True])
def test_direct_request_fails_before_client_creation(monkeypatch,locked):
    monkeypatch.setattr('src.gemini_client.genai.Client',lambda **kwargs:pytest.fail('SDK initialized'))
    with pytest.raises(ModelDisallowedError,match='MODEL_DISALLOWED'):
        GeminiClient('mock-key',model_name='gemini-flash-lite',fallback_policy='MODEL_LOCK' if locked else 'SPEED_FIRST')


def test_config_rejects_requested_lite_and_filters_fallbacks(tmp_path,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','mock-key')
    monkeypatch.setenv('ALLOW_LITE_MODELS','false')
    monkeypatch.setenv('GEMINI_MODEL','gemini-flash-lite')
    with pytest.raises(ConfigurationError,match='LITE_MODEL_DISABLED'):
        load_config(tmp_path)
    monkeypatch.setenv('GEMINI_MODEL','gemini-3.6-flash')
    monkeypatch.setenv('FALLBACK_MODELS','gemini-3.8-flash,gemini-3.5-flash-lite')
    assert load_config(tmp_path).fallback_models==('gemini-3.8-flash',)


def test_speed_first_exhausts_non_lite_candidates_without_calling_lite(monkeypatch):
    obj,calls,waits=fast_client(monkeypatch,[ProviderError(delay='26s')]+[ProviderError(503,None)]*3)
    obj.model_name=obj.requested_model='gemini-3.6-flash'
    obj.fallback_models=('gemini-3.8-flash','gemini-3.5-flash-lite')
    with pytest.raises(GeminiTranscribeError,match='NO_ELIGIBLE_FALLBACK_MODEL'):
        obj.generate_transcription('asset','verbatim')
    assert calls==['gemini-3.6-flash']+['gemini-3.8-flash']*3
    assert waits==[1,2]  # No 26-second quota sleep.


def test_profiles_and_health_switch_cannot_restore_lite(tmp_path,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','mock-key')
    monkeypatch.setenv('GEMINI_MODEL','gemini-3.6-flash')
    monkeypatch.setenv('ALLOW_LITE_MODELS','false')
    cfg=load_config(tmp_path)
    planner=DabbAudioOrchestrator(tmp_path/'audio.wav',cfg)
    before=planner.active_model
    with pytest.raises(ModelDisallowedError):
        planner.select_model('gemini-3.5-flash-lite')
    assert planner.active_model==before and 'gemini-3.5-flash-lite' not in planner.model_profiles
    health=ModelHealth(cfg)
    health.profiles['gemini-3.5-flash-lite']=ModelHealthProfile()
    assert not health.eligible('gemini-3.5-flash-lite')
    obj=GeminiClient.__new__(GeminiClient)
    obj.model_health=health
    obj.model_name=obj.requested_model=before
    obj.fallback_models=('gemini-3.5-flash-lite',)
    with pytest.raises(GeminiTranscribeError,match='NO_ELIGIBLE_FALLBACK_MODEL'):
        obj.advance_after_output_failure()
