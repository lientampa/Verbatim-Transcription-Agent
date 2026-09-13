import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from google.genai.types import Model
from src.gemini_client import GeminiClient,GeminiTranscribeError
from src.provider_adapters import TranscriptionModelAdapter,ProviderAdapterError,absolute_timestamp
from src.model_policy import PREFERRED_MODEL_CHAIN
from src.model_health import ModelHealth
from src.config import load_config
from tests.test_stage43_adaptation import ProviderError
from tests.test_transcribe_adapter import sdk_client,native,PROMPT


def test_404_quarantine_and_fresh_job(monkeypatch):
    c,calls=sdk_client(monkeypatch,[ProviderError(404,None),True,True])
    asset=NS(uri="files/test",mime_type="audio/wav")
    c.generate_transcription(asset,"verbatim",PROMPT)
    state=c._registry().states["gemini-2.5-flash"]
    assert state.state=="UNAVAILABLE" and state.quarantined and state.reason=="MODEL_UNAVAILABLE"
    c.generate_transcription(asset,"verbatim",PROMPT.replace("BLOCK_004","BLOCK_005"))
    assert [m for m,_,_ in calls]==["gemini-2.5-flash","gemini-3.5-transcribe","gemini-3.5-transcribe"]
    assert "gemini-2.5-flash" not in c.runtime_eligible_chain
    fresh,fresh_calls=sdk_client(monkeypatch,[True])
    fresh.generate_transcription(asset,"verbatim",PROMPT)
    assert fresh_calls[0][0]=="gemini-2.5-flash"


def test_adapter_failure_is_not_transient(monkeypatch,capsys):
    c,calls=sdk_client(monkeypatch,[ProviderError(404,None),ProviderAdapterError("TRANSCRIBE_ANNOTATION_INVALID",detail="SPEAKER_TYPE"),True])
    c.generate_transcription(NS(uri="files/test",mime_type="audio/wav"),"verbatim",PROMPT)
    state=c._registry().states["gemini-3.5-transcribe"]
    assert state.state=="ADAPTER_UNAVAILABLE" and state.quarantined
    assert state.reason=="MODEL_ADAPTER_FAILURE"
    assert [m for m,_,_ in calls]==list(PREFERRED_MODEL_CHAIN[:3])
    assert c.call_history[1]["actual_sleep_seconds"]==0
    assert "stage=ANNOTATION_PARSE" in capsys.readouterr().out


def test_preflight_cheap_404_and_adapter_contract(monkeypatch):
    lookups=[]
    def lookup(model):
        lookups.append(model)
        if model=="gemini-2.5-flash":raise ProviderError(404,None)
        return Model(name=model)
    sdk=NS(models=NS(list=lambda:[Model(name=m,supported_actions=["generateContent"]) for m in PREFERRED_MODEL_CHAIN],
        get=lookup,generate_content=lambda **kw:pytest.fail("preflight generation")),interactions=NS(create=lambda **kw:pytest.fail("preflight transcription")))
    monkeypatch.setattr("src.gemini_client.genai.Client",lambda **kw:sdk)
    monkeypatch.setattr(TranscriptionModelAdapter,"preflight",lambda sdk:(False,"SDK_NATIVE_CONTRACT_MISMATCH"))
    c=GeminiClient("mock-key")
    assert c.preferred_chain==c.discovered_chain==PREFERRED_MODEL_CHAIN
    assert c.runtime_eligible_chain==PREFERRED_MODEL_CHAIN[2:]
    assert c.model_name=="gemini-3.6-flash"
    assert "gemini-3.5-transcribe" not in lookups
    assert c._registry().states["gemini-2.5-flash"].quarantined
    assert c._registry().states["gemini-3.5-transcribe"].reason=="MODEL_ADAPTER_UNAVAILABLE"


def test_optional_native_fields_and_utf8_span():
    d=native();words=d["steps"][0]["content"][0]["annotations"]
    words[0].pop("speaker");words[0].pop("text");words[0].pop("end_offset")
    words[0].update(start_index=0,end_index=len("Dạ,".encode("utf-8")))
    result=json.loads(TranscriptionModelAdapter().canonical(d,PROMPT))
    assert result["segments"][0]["text"]=="Dạ,"
    assert result["segments"][0]["speaker"].startswith("Người nói ")
    assert result["segments"][0]["speaker"]!=result["segments"][1]["speaker"]


@pytest.mark.parametrize("start",[0,120,600,4000])
def test_documented_relative_offsets(start):
    prompt=PROMPT.replace("start=614.208, end=741.312",f"start={start}, end={start+200}")
    d=native();word=d["steps"][0]["content"][0]["annotations"][0]
    word.update(start_offset="90s",end_offset="90.4s")
    d["steps"][0]["content"][0]["annotations"][1].update(start_offset="91s",end_offset="91.4s")
    result=json.loads(TranscriptionModelAdapter().canonical(d,prompt))
    second=start+90
    assert result["segments"][0]["timestamp"]==f"{second//3600:02d}:{second//60%60:02d}:{second%60:02d}"
    if start==120: assert result["segments"][0]["timestamp"]=="00:03:30"


def test_explicit_absolute_contract_no_double_offset():
    assert absolute_timestamp(210,120,300,basis="SOURCE_ABSOLUTE")==210
    assert absolute_timestamp(90,120,300,basis="BLOCK_RELATIVE")==210
    with pytest.raises(ProviderAdapterError):absolute_timestamp(90,120,300,basis="AUTO")
    assert TranscriptionModelAdapter.TIMESTAMP_BASIS=="BLOCK_RELATIVE"


def test_diagnostic_has_stage_condition_without_private_text(capsys):
    d=native();d["steps"][0]["content"][0]["annotations"][0]["speaker"]=42
    with pytest.raises(ProviderAdapterError) as caught:TranscriptionModelAdapter().canonical(d,PROMPT)
    assert caught.value.stage=="ANNOTATION_PARSE" and "SPEAKER_TYPE=int" in caught.value.detail
    logs=capsys.readouterr().out
    assert "Dạ" not in logs and "Vâng" not in logs


def test_five_cause_exhaustion(monkeypatch,tmp_path,capsys):
    effects=[ProviderError(404,None),ProviderAdapterError("TRANSCRIBE_ANNOTATION_INVALID"),ProviderError(delay="12s")]+[ProviderError(503,None)]*3
    c,calls=sdk_client(monkeypatch,effects)
    monkeypatch.setenv("GEMINI_API_KEY","mock-key")
    c.model_health=ModelHealth(load_config(tmp_path))
    c.model_health.profile("gemini-3.5-flash").degraded=True
    with pytest.raises(GeminiTranscribeError) as caught:
        c.generate_transcription(NS(uri="files/test",mime_type="audio/wav"),"verbatim",PROMPT)
    for reason in ("MODEL_UNAVAILABLE","MODEL_ADAPTER_FAILURE","PROVIDER_429","MODEL_HEALTH_CIRCUIT_OPEN","PROVIDER_503_RETRY_EXHAUSTED"):
        assert reason in str(caught.value)
    assert "gemini-3.5-flash" not in [m for m,_,_ in calls]
    assert c.runtime_eligible_chain==()
    assert not c._registry().states["gemini-3.8-flash"].quarantined
    c.log_model_summary()
    assert "models_quarantined=" in capsys.readouterr().out


@pytest.mark.parametrize("duration,gap",[(374.208,318.208),(187.104,144.104)])
def test_severe_active_coverage_remains_retry(tmp_path,monkeypatch,duration,gap):
    from src.coverage_validator import CoverageValidator,CoverageDecision
    from src.block_builder import SourceBlock
    from src.response_parser import TranscriptSegment
    monkeypatch.setenv("GEMINI_API_KEY","mock-key")
    validator=CoverageValidator(load_config(tmp_path))
    monkeypatch.setattr(validator.analyzer,"analyze",lambda *args:gap)
    last=int(duration-gap)
    block=SourceBlock(1,"001",0,duration,tmp_path/"fake.wav",1)
    result=validator.validate(block,NS(segments=[TranscriptSegment(1,"speech",f"00:{last:02d}")]))
    assert result.decision==CoverageDecision.RETRY and result.tail_activity=="ACTIVE"


def test_full_exhaustion_preserves_120_checkpoint_and_same_slice(tmp_path,monkeypatch,capsys):
    import re
    from tests.test_stage4_audio_retry import run_audio_case
    from src.transcript_validator import TranscriptValidator
    real_init=GeminiClient.__init__;real_generate=GeminiClient.generate_transcription
    real_validate=TranscriptValidator.validate_block_result
    calls=[];assets={};instances=[];waits=[]
    def setup(config):
        config=replace(config,gemini_model="gemini-2.5-flash",next_target_policy="MAX_FIRST",
            max_duration_seconds=120,unknown_model_duration_sec=120,model_health_strike_limit=2,retry_initial_delay_seconds=1)
        monkeypatch.setattr("src.main.load_config",lambda **kw:config)
        monkeypatch.setattr(TranscriptValidator,"validate_block_result",real_validate)
        upload=GeminiClient.upload_audio
        def upload_file(self,audio_path):
            asset=upload(self,audio_path);asset.uri=asset.name;asset.mime_type="audio/wav";assets[asset.uri]=asset;return asset
        monkeypatch.setattr(GeminiClient,"upload_audio",upload_file)
        def flash(model,contents,config):
            asset,prompt=contents;calls.append((model,asset))
            if len(calls)>1:
                cp=json.loads((tmp_path/"state"/"checkpoint.json").read_text(encoding="utf-8"))
                assert cp["next_audio_start_us"]==120000000
            if model=="gemini-2.5-flash" and len(calls)>1:raise ProviderError(404,None)
            if model=="gemini-3.6-flash":raise ProviderError(delay="12s")
            if model=="gemini-3.8-flash":raise ProviderError(503,None)
            identity={key:re.search(r'- '+key+r': "([^"\n]+)"',prompt).group(1) for key in ("job_id","session_id","block_id")}
            sec=int(asset.bounds[1]-2 if len(calls)==1 else asset.bounds[0]+1)
            data=dict(schema_version="1.0",**identity,first_source_index=1,last_source_index=1,status="CONFIRMED",
                segments=[dict(source_index=1,text="Vâng.",timestamp=f"{sec//60:02d}:{sec%60:02d}")])
            return NS(text=json.dumps(data),candidates=None,usage_metadata=None)
        def transcribe(model,input,generation_config):
            calls.append((model,assets[input[0]["uri"]]))
            raise ProviderAdapterError("TRANSCRIBE_ANNOTATION_INVALID",detail="SPEAKER_TYPE")
        sdk=NS(models=NS(list=lambda:[Model(name=m,supported_actions=["generateContent"]) for m in PREFERRED_MODEL_CHAIN],generate_content=flash),
            interactions=NS(create=transcribe))
        monkeypatch.setattr("src.gemini_client.genai.Client",lambda **kw:sdk)
        def init(self,**kw):real_init(self,**kw);instances.append(self)
        monkeypatch.setattr(GeminiClient,"__init__",init)
        monkeypatch.setattr(GeminiClient,"generate_transcription",real_generate)
        monkeypatch.setattr("src.gemini_client.time.sleep",waits.append)
        monkeypatch.setattr("src.coverage_validator.TailActivityAnalyzer.analyze",lambda self,path,offset,duration,threshold:duration)
    result,_,_,_,cp=run_audio_case(tmp_path,monkeypatch,total_duration=600,fail_count=0,setup=setup)
    assert result==1 and cp["next_audio_start_us"]==120000000
    assert len(cp["block_metrics"])==1 and len(cp["confirmed_segments"])==1
    for reason in ("MODEL_UNAVAILABLE","MODEL_ADAPTER_FAILURE","PROVIDER_429","MODEL_HEALTH_CIRCUIT_OPEN","PROVIDER_503_RETRY_EXHAUSTED"):
        assert reason in cp["error_message"]
    assert [m for m,_ in calls]==["gemini-2.5-flash"]*2+["gemini-3.5-transcribe","gemini-3.6-flash"]+["gemini-3.5-flash"]*2+["gemini-3.8-flash"]*3
    assert all(asset is calls[1][1] for _,asset in calls[1:5])
    assert all(asset is calls[5][1] for _,asset in calls[5:])
    assert waits[-2:]==[1,2]
    logs=capsys.readouterr().out
    assert "checkpoint_end=120.0" in logs
    assert "generation=2" in logs and "coverage_generation=2 physical_generation=2" in logs
    assert "[JOB_MODEL_SUMMARY]" in logs


def test_sdk_contract_preflight_is_local(monkeypatch):
    from google.genai._gaos.types.interactions.generationconfig import GenerationConfig
    sdk=NS(interactions=NS(create=lambda **kw:pytest.fail("must not invoke provider")))
    assert TranscriptionModelAdapter.preflight(sdk)==(True,"SDK_NATIVE_CONTRACT_AVAILABLE")
    def broken(*args,**kwargs):raise ValueError("old SDK")
    monkeypatch.setattr(GenerationConfig,"model_validate",broken)
    supported,reason=TranscriptionModelAdapter.preflight(sdk)
    assert not supported and reason.startswith("SDK_NATIVE_CONTRACT_UNAVAILABLE")


def test_unavailable_interactions_property_keeps_flash(monkeypatch):
    class SDK:
        models=NS(list=lambda:[Model(name=m,supported_actions=["generateContent"]) for m in PREFERRED_MODEL_CHAIN],generate_content=lambda **kw:None)
        @property
        def interactions(self):raise NotImplementedError("endpoint missing")
    monkeypatch.setattr("src.gemini_client.genai.Client",lambda **kw:SDK())
    c=GeminiClient("mock-key")
    assert c.model_name=="gemini-2.5-flash"
    assert "gemini-3.5-transcribe" not in c.runtime_eligible_chain


def test_unknown_timing_is_not_invented():
    d=native();d["steps"][0]["content"][0]["annotations"][0].pop("start_offset")
    with pytest.raises(ProviderAdapterError,match="TRANSCRIBE_TIMESTAMP_UNSUPPORTED"):
        TranscriptionModelAdapter().canonical(d,PROMPT)
