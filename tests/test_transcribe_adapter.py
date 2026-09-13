import json
from types import SimpleNamespace as NS
from dataclasses import replace
import pytest
from google.genai.types import Model
from src.model_policy import PREFERRED_MODEL_CHAIN,PRIMARY_MODEL,effective_model_chain
from src.gemini_client import GeminiClient,GeminiTranscribeError,GeminiSizeError
from src.provider_adapters import TranscriptionModelAdapter,ProviderAdapterError,capabilities
from src.response_parser import ResponseParser
from src.config import load_config
from tests.test_stage43_adaptation import ProviderError

PROMPT='''- job_id: "JOB_001"
- session_id: "SESSION_001"
- block_id: "BLOCK_004"
- Absolute audio boundaries (seconds): start=614.208, end=741.312.
'''


def native(status="completed"):
    words=[dict(type="word_info",text="Dạ,",speaker="spk_1",start_offset="1.0s",end_offset="1.4s"),
           dict(type="word_info",text="em...",speaker="spk_1",start_offset="1.5s",end_offset="2.0s"),
           dict(type="word_info",text="Vâng.",speaker="spk_2",start_offset="110.0s",end_offset="111.0s")]
    return dict(status=status,steps=[dict(type="model_output",content=[dict(type="text",text="Dạ, em... Vâng.",annotations=words)])],
                usage=dict(total_input_tokens=500,total_output_tokens=20,total_tokens=520))


def sdk_client(monkeypatch,effects):
    calls=[];remaining=iter(effects)
    def call(model,asset,config):
        calls.append((model,asset,config))
        effect=next(remaining)
        if isinstance(effect,Exception): raise effect
        if model=="gemini-3.5-transcribe": return native()
        return NS(text="{}",candidates=None,usage_metadata=None)
    sdk=NS(models=NS(list=lambda:[Model(name="models/"+m,supported_actions=["createInteraction" if m.endswith("transcribe") else "generateContent"]) for m in PREFERRED_MODEL_CHAIN],
        generate_content=lambda model,contents,config:call(model,contents[0],config)),
        interactions=NS(create=lambda model,input,generation_config:call(model,input,generation_config)))
    monkeypatch.setattr("src.gemini_client.genai.Client",lambda **kw:sdk)
    monkeypatch.setattr("src.gemini_client.time.sleep",lambda _:None)
    return GeminiClient("mock-key"),calls


def test_default_primary_and_exact_order(tmp_path,monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY","mock-key");monkeypatch.delenv("GEMINI_MODEL",raising=False)
    assert load_config(tmp_path).gemini_model==PRIMARY_MODEL=="gemini-2.5-flash"
    assert PREFERRED_MODEL_CHAIN==("gemini-2.5-flash","gemini-3.5-transcribe","gemini-3.6-flash","gemini-3.5-flash","gemini-3.8-flash")


def test_primary_success_no_fallback(monkeypatch):
    c,calls=sdk_client(monkeypatch,[True]);c.generate_transcription("asset","verbatim")
    assert [m for m,_,_ in calls]==[PRIMARY_MODEL]
    assert calls[0][2].thinking_config.thinking_budget==0
    assert calls[0][2].thinking_config.thinking_level is None


def test_transcribe_fallback_and_canonical(monkeypatch):
    c,calls=sdk_client(monkeypatch,[ProviderError(delay="12s"),True])
    result=c.generate_transcription(NS(uri="files/test",mime_type="audio/wav"),"verbatim",PROMPT)
    assert [m for m,_,_ in calls]==list(PREFERRED_MODEL_CHAIN[:2])
    assert calls[1][2]=={"transcription_config":{"mode":{"type":"verbatim","diarization_mode":"speaker","timestamp_granularities":["word"]}}}
    data=json.loads(result)
    assert data["block_id"]=="BLOCK_004" and data["segments"][0]["timestamp"]=="00:10:15"
    assert data["segments"][0]["text"]=="Dạ, em..."
    assert [s["speaker"] for s in data["segments"]]==["Người nói 1","Người nói 2"]
    assert c.actual_capabilities.provider_adapter=="transcribe" and not c.actual_capabilities.supports_thinking
    assert c.last_response_metadata["output_tokens"]==20
    from pathlib import Path
    ResponseParser(schema_path=Path("schemas/transcription_result.schema.json")).parse(result)


def test_all_endpoints_exhaust_in_exact_order(monkeypatch):
    c,calls=sdk_client(monkeypatch,[ProviderError(delay="12s")]*5)
    asset=NS(uri="files/test",mime_type="audio/wav")
    with pytest.raises(GeminiTranscribeError,match="NO_ELIGIBLE"):
        c.generate_transcription(asset,"verbatim",PROMPT)
    assert [m for m,_,_ in calls]==list(PREFERRED_MODEL_CHAIN)
    assert all(a is asset for m,a,_ in calls if m!="gemini-3.5-transcribe")
    assert c.actual_capabilities==capabilities("gemini-3.8-flash")
    assert c.last_response_metadata["actual_model"]=="gemini-3.8-flash"


def test_transcribe_failure_advances_to_36(monkeypatch):
    c,calls=sdk_client(monkeypatch,[ProviderError(delay="12s"),ProviderError(503,None),ProviderError(503,None),ProviderError(503,None),True])
    c.generate_transcription(NS(uri="files/test",mime_type="audio/wav"),"verbatim",PROMPT)
    assert [m for m,_,_ in calls]==[PRIMARY_MODEL]+["gemini-3.5-transcribe"]*3+["gemini-3.6-flash"]
    assert c.actual_capabilities.provider_adapter=="generate_content"
    assert calls[-1][2].thinking_config.thinking_level.value=="MINIMAL"


def test_discovery_requires_native_endpoint_and_bans_lite():
    models=[Model(name="models/gemini-3.5-transcribe",supported_actions=["generateContent"]),Model(name="models/gemini-3.5-flash-lite",supported_actions=["generateContent"])]
    assert effective_model_chain(models,True)==("gemini-3.5-transcribe",)
    models[0].supported_actions=["createInteraction"]
    assert effective_model_chain(models,False)==()
    assert effective_model_chain(models,True)==("gemini-3.5-transcribe",)


@pytest.mark.parametrize("change",[
    lambda d:d["steps"][0]["content"][0].update(annotations=[]),
    lambda d:d["steps"][0]["content"][0].update(text="extra missing words"),
    lambda d:d["steps"][0]["content"][0]["annotations"][0].update(start_offset="999s"),
    lambda d:d["steps"][0]["content"][0]["annotations"][0].update(speaker=42)])
def test_invalid_annotations_fail_closed(change):
    data=native();change(data)
    with pytest.raises(ProviderAdapterError):TranscriptionModelAdapter().canonical(data,PROMPT)


def test_no_cross_block_speaker_guess():
    prompt=PROMPT+"- Nhãn người nói đã dùng trong các block được xác nhận: ['Người nói 1', 'Người nói 2']."
    data=json.loads(TranscriptionModelAdapter().canonical(native(),prompt))
    assert data["segments"][0]["speaker"]=="Người nói 3"
    assert "spk_" not in json.dumps(data)


def test_truncation_remains_size_failure(monkeypatch):
    c,_=sdk_client(monkeypatch,[])
    c.model_name="gemini-3.5-transcribe"
    c.client.interactions.create=lambda **kw:native("incomplete")
    with pytest.raises(GeminiSizeError):
        c.generate_transcription(NS(uri="files/test",mime_type="audio/wav"),"verbatim",PROMPT)
    assert c.model_name=="gemini-3.5-transcribe"


def test_native_http_retry_info_preserved():
    import httpx
    from google.genai._gaos.errors.genaierror import GenAiError
    from src.gemini_client import structured_retry_after
    exc=GenAiError("quota",httpx.Response(429,json={"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"12s"}]}}))
    def fail(**kw): raise exc
    with pytest.raises(ProviderAdapterError) as caught:
        TranscriptionModelAdapter().generate(NS(interactions=NS(create=fail)),"gemini-3.5-transcribe",NS(uri="file",mime_type="audio/wav"),PROMPT)
    assert caught.value.code==429 and structured_retry_after(caught.value)==12


def test_runtime_transcribe_coverage_same_model_and_checkpoint(tmp_path,monkeypatch):
    from tests.test_stage4_audio_retry import run_audio_case
    from src.transcript_validator import TranscriptValidator
    real_init=GeminiClient.__init__;real_generate=GeminiClient.generate_transcription
    real_validate=TranscriptValidator.validate_block_result
    calls=[];assets={};instances=[]
    def setup(config):
        config=replace(config,gemini_model=PRIMARY_MODEL,next_target_policy="MAX_FIRST",max_duration_seconds=120,unknown_model_duration_sec=120)
        monkeypatch.setattr("src.main.load_config",lambda **kw:config)
        monkeypatch.setattr(TranscriptValidator,"validate_block_result",real_validate)
        upload=GeminiClient.upload_audio
        def upload_file(self,audio_path):
            asset=upload(self,audio_path);asset.uri=asset.name;asset.mime_type="audio/wav";assets[asset.uri]=asset;return asset
        monkeypatch.setattr(GeminiClient,"upload_audio",upload_file)
        def flash(model,contents,config):
            calls.append((model,contents[0]));raise ProviderError(delay="12s")
        def transcribe(model,input,generation_config):
            asset=assets[input[0]["uri"]];calls.append((model,asset))
            duration=asset.bounds[1]-asset.bounds[0]
            relative=1 if len(calls)==2 else duration-2
            return dict(status="completed",steps=[dict(type="model_output",content=[dict(type="text",text="Vâng.",annotations=[dict(type="word_info",text="Vâng.",speaker="spk_1",start_offset=f"{relative}s",end_offset=f"{relative+1}s")])])])
        sdk=NS(models=NS(list=lambda:[Model(name="models/"+m,supported_actions=["createInteraction" if m.endswith("transcribe") else "generateContent"]) for m in PREFERRED_MODEL_CHAIN],generate_content=flash),interactions=NS(create=transcribe))
        monkeypatch.setattr("src.gemini_client.genai.Client",lambda **kw:sdk)
        def init(self,**kw):real_init(self,**kw);instances.append(self)
        monkeypatch.setattr(GeminiClient,"__init__",init)
        monkeypatch.setattr(GeminiClient,"generate_transcription",real_generate)
        monkeypatch.setattr("src.coverage_validator.TailActivityAnalyzer.analyze",lambda self,path,offset,duration,threshold:duration)
    result,_,_,_,cp=run_audio_case(tmp_path,monkeypatch,total_duration=240,fail_count=0,setup=setup)
    assert result==0
    assert calls[0][1] is calls[1][1]  # Provider fallback keeps physical asset.
    assert [m for m,_ in calls]==[PRIMARY_MODEL]+["gemini-3.5-transcribe"]*(len(calls)-1)
    metric=cp["block_metrics"][0]
    assert metric["physical_generation_count"]==2 and metric["coverage_generation_count"]==2
    assert metric["size_retry_count"]==0 and metric["provider_attempt_count"]==3
    assert cp["next_audio_start_us"]==240000000
    assert "spk_" not in (tmp_path/"output"/"transcript.txt").read_text(encoding="utf-8")


def test_all_five_adaptive_profiles_are_independent(tmp_path,monkeypatch):
    from src.block_builder import DabbAudioOrchestrator
    monkeypatch.setenv("GEMINI_API_KEY","mock-key")
    config=replace(load_config(tmp_path),gemini_model=PRIMARY_MODEL)
    planner=DabbAudioOrchestrator(tmp_path/"audio.wav",config)
    for model in PREFERRED_MODEL_CHAIN:
        planner.select_model(model)
    profiles=[planner.model_profiles[m] for m in PREFERRED_MODEL_CHAIN]
    assert len({id(p["dabb"]) for p in profiles})==5
    assert len({id(p["dabb"].output_tracker) for p in profiles})==5
    profiles[0]["dabb"].output_tracker.record_block(100,1000)
    profiles[0]["safe_duration"]=50;profiles[0]["coverage_failures"]=2
    assert all(p["safe_duration"] is None and p["coverage_failures"]==0 for p in profiles[1:])
    assert all(p["dabb"].output_tracker.get_ratio()==config.default_output_ratio for p in profiles[1:])
