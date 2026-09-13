"""Production generation control with fake SDK/I/O; no paid calls."""
import json
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from google.genai import types
from src.gemini_client import GeminiClient, GeminiSizeError
from src.model_policy import PREFERRED_MODEL_CHAIN
# Legacy generateContent-only fake SDK: Transcribe endpoint is absent here.
PREFERRED_MODEL_CHAIN = tuple(m for m in PREFERRED_MODEL_CHAIN if m != "gemini-3.5-transcribe")
from tests.test_stage43_adaptation import ProviderError
from tests.test_stage4_audio_retry import run_audio_case


def response(truncated=False):
    return NS(text="{}", candidates=[NS(finish_reason="MAX_TOKENS" if truncated else "STOP",
        content=NS(parts=[NS(text="{}",thought=False)]))],
        usage_metadata=NS(prompt_token_count=6000,candidates_token_count=1000,
                          thoughts_token_count=60000,total_token_count=67000))


def client(monkeypatch, effects, **kwargs):
    seen=[]
    def generate_content(model, contents, config):
        seen.append((model,contents[0],config))
        effect=effects.pop(0) if effects else response()
        if isinstance(effect,Exception): raise effect
        return effect
    models=[types.Model(name="models/"+m,supported_actions=["generateContent"],output_token_limit=8000-i*1000)
            for i,m in enumerate(PREFERRED_MODEL_CHAIN)]
    monkeypatch.setattr("src.gemini_client.genai.Client",lambda **kw:NS(models=NS(list=lambda:models,generate_content=generate_content)))
    return GeminiClient("mock-key",**kwargs),seen


def test_fallback_rebuilds_thinking_and_cap(monkeypatch):
    c,seen=client(monkeypatch,[ProviderError(delay="12s")]*3+[response()])
    c.generate_transcription("asset","verbatim")
    assert [m for m,_,_ in seen]==list(PREFERRED_MODEL_CHAIN)
    assert [cfg.thinking_config.thinking_level.value.lower() if cfg.thinking_config.thinking_level else None for _,_,cfg in seen]==[None,"minimal","minimal","low"]
    assert [cfg.max_output_tokens for _,_,cfg in seen]==[8000,7000,6000,5000]
    assert len({id(cfg) for _,_,cfg in seen})==4
    assert seen[0][2].thinking_config.thinking_budget == 0
    assert all(cfg.automatic_function_calling.disable and not cfg.tools for _,_,cfg in seen)


@pytest.mark.parametrize("model,level",[("gemini-3.5-flash","minimal"),("gemini-3.6-flash","minimal"),
    ("gemini-3.7-flash","low"),("gemini-3.8-flash","low"),("gemini-3.1-pro-preview","low"),("gemini-2.5-flash",None)])
def test_model_capabilities(model,level):
    c=GeminiClient.__new__(GeminiClient)
    cfg=c.generation_config(model,types.GenerateContentConfig())
    assert (cfg.thinking_config.thinking_level.value.lower() if cfg.thinking_config and cfg.thinking_config.thinking_level else None)==level


def test_reasoning_pressure_same_asset_retry(monkeypatch,capsys):
    c,seen=client(monkeypatch,[response(True),response()],transcription_thinking_level="high",model_name="gemini-3.8-flash",fallback_policy="MODEL_LOCK")
    asset=object()
    c.generate_transcription(asset,"verbatim",'- block_id: "BLOCK_004"')
    assert len(seen)==2 and seen[0][1] is seen[1][1] is asset
    assert [cfg.thinking_config.thinking_level.value for _,_,cfg in seen]==["HIGH","LOW"]
    assert c.generation_config_retry_count==1 and c._provider_attempt==2
    assert c.call_history[0]["max_tokens_classification"]=="REASONING_OUTPUT_PRESSURE"
    assert c.call_history[0]["thought_to_visible_ratio"]==60
    assert "same_physical_block=true" in capsys.readouterr().out


def test_minimum_does_not_loop(monkeypatch):
    c,seen=client(monkeypatch,[response(True)])
    with pytest.raises(GeminiSizeError): c.generate_transcription("asset","verbatim")
    assert len(seen)==1 and c.generation_config_retry_count==0


def test_config_retry_is_bounded(monkeypatch):
    c,seen=client(monkeypatch,[response(True),response(True)],transcription_thinking_level="medium",model_name="gemini-3.8-flash",fallback_policy="MODEL_LOCK")
    with pytest.raises(GeminiSizeError): c.generate_transcription("asset","verbatim",'- block_id: "BLOCK_004"')
    assert len(seen)==2 and c.generation_config_retry_count==1


def test_undiscovered_cap_is_explicit(monkeypatch):
    c,seen=client(monkeypatch,[response()]); c.model_output_limits={}
    c.generate_transcription("asset","verbatim")
    assert seen[0][2].max_output_tokens is None
    assert c.last_response_metadata["effective_max_output_tokens"]=="PROVIDER_DEFAULT"
    assert c.last_response_metadata["model_output_token_limit"]=="UNDISCOVERED"


@pytest.mark.parametrize("high",[False,True])
def test_runtime_config_retry_budget_and_emergency(tmp_path,monkeypatch,high):
    real_init=GeminiClient.__init__; real_generate=GeminiClient.generate_transcription
    seen=[]; instances=[]
    def setup(config):
        config=replace(config,next_target_policy="MAX_FIRST",max_duration_seconds=120,
                       unknown_model_duration_sec=120,transcription_thinking_level="high" if high else "minimal",size_retry_limit=5)
        monkeypatch.setattr("src.main.load_config",lambda **kw:config)
        harness=GeminiClient.generate_transcription
        def generate_content(model,contents,config):
            asset,prompt=contents;seen.append(asset)
            if len(seen)==1: return response(True)
            text=harness(instances[0],gemini_file=asset,user_prompt=prompt)
            return NS(text=text,candidates=None,usage_metadata=None)
        models=[types.Model(name="models/gemini-3.5-flash",supported_actions=["generateContent"],output_token_limit=65536)]
        monkeypatch.setattr("src.gemini_client.genai.Client",lambda **kw:NS(models=NS(list=lambda:models,generate_content=generate_content)))
        def init(self,**kw): real_init(self,**kw);instances.append(self)
        monkeypatch.setattr(GeminiClient,"__init__",init)
        monkeypatch.setattr(GeminiClient,"generate_transcription",real_generate)
    result,_,uploads,_,cp=run_audio_case(tmp_path,monkeypatch,fail_count=0,minimum=120,total_duration=240,setup=setup)
    assert result==0
    metric=cp["block_metrics"][0]
    assert metric["generation_config_retry_count"]==int(high)
    assert metric["size_retry_count"]==int(not high)
    assert metric["physical_generation_count"]==1+int(not high)
    assert metric["coverage_generation_count"]==1
    assert metric["provider_attempt_count"]==2
    assert (seen[0] is seen[1])==high
    assert seen[1].bounds==([0,120] if high else [0,72])
    if not high: assert uploads[0].name!=uploads[1].name
    assert seen[2].bounds[1]-seen[2].bounds[0]==120  # Fresh MAX_FIRST unaffected.


def test_emergency_floor_checkpoint_and_coverage(tmp_path,monkeypatch):
    result,_,_,calls,cp=run_audio_case(tmp_path,monkeypatch,minimum=120,attempts=10,fail_count=100,total_duration=120)
    assert result==1
    assert [c[0].bounds for c in calls]==[[0,120],[0,72],[0,60]]
    assert "EMERGENCY_SIZE_FLOOR_EXHAUSTED" in cp["error_message"]
    assert cp["next_audio_start_us"]==0 and cp["confirmed_segments"]==[]


def test_exact_production_intervals_and_coverage_floor(tmp_path,monkeypatch):
    from src.config import load_config
    from src.block_builder import DabbAudioOrchestrator,AdaptiveBlockSlice,BlockBuilderError
    from src.failure_classifier import FailureType
    monkeypatch.setenv("GEMINI_API_KEY","mock-key")
    cfg=replace(load_config(tmp_path),min_duration_seconds=120,initial_target_input_tokens=3000)
    planner=DabbAudioOrchestrator(tmp_path/"audio.wav",cfg)
    # Same dataclass contract as production SourceBlock; physical slicer alone is mocked.
    from src.block_builder import SourceBlock
    old=SourceBlock(total_blocks=4,source_index=4,source_index_str="004",file_path=tmp_path/"old.wav",start_time_seconds=614.208,end_time_seconds=868.416)
    attempt=AdaptiveBlockSlice(old,4,254.208,636,10000,10000)
    paths=[]
    def cut(**kw):
        paths.append((kw["start_sec"],kw["end_sec"]));return tmp_path/f"slice{len(paths)}.wav"
    monkeypatch.setattr(planner,"_get_audio_builder",lambda *a:NS(slice_time_range=cut))
    planner.on_coverage_failure(254.208)
    planner.rebuild_audio_block(attempt,.5,reason="COVERAGE_FAILURE")
    assert attempt.current_duration_seconds==pytest.approx(127.104)
    planner.on_block_failure(FailureType.SIZE_FAILURE)
    planner.rebuild_audio_block(attempt)
    assert attempt.current_duration_seconds==pytest.approx(120)
    planner.on_block_failure(FailureType.SIZE_FAILURE)
    planner.rebuild_audio_block(attempt)
    assert attempt.current_duration_seconds==pytest.approx(72)
    assert attempt.generation==4 and all(start==614.208 for start,_ in paths)
    with pytest.raises(BlockBuilderError,match="MIN_BLOCK_REACHED"):
        planner.rebuild_audio_block(attempt,.5,reason="COVERAGE_FAILURE")


def test_failed_tail_preserves_614208_checkpoint(tmp_path,monkeypatch):
    real_init=GeminiClient.__init__; real_generate=GeminiClient.generate_transcription
    seen=[];instances=[]
    def setup(config):
        config=replace(config,gemini_model="gemini-3.5-flash",next_target_policy="MAX_FIRST",max_duration_seconds=614.208,size_retry_limit=10)
        monkeypatch.setattr("src.main.load_config",lambda **kw:config)
        harness=GeminiClient.generate_transcription
        def generate_content(model,contents,config):
            asset,prompt=contents
            asset.bounds=[round(value,6) for value in asset.bounds]
            seen.append(asset.bounds)
            if len(seen)>2: return response(True)
            data=json.loads(harness(instances[0],gemini_file=asset,user_prompt=prompt))
            second=int(asset.bounds[1]-2 if len(seen)==1 else asset.bounds[0]+1)
            data["segments"][0]["timestamp"]=f"{second//60:02d}:{second%60:02d}"
            return NS(text=json.dumps(data),candidates=None,usage_metadata=None)
        models=[types.Model(name="models/gemini-3.5-flash",supported_actions=["generateContent"],output_token_limit=65536)]
        monkeypatch.setattr("src.gemini_client.genai.Client",lambda **kw:NS(models=NS(list=lambda:models,generate_content=generate_content)))
        def init(self,**kw):real_init(self,**kw);instances.append(self)
        monkeypatch.setattr(GeminiClient,"__init__",init)
        def generate(self,**kw):
            # Pin the supplied fractional scenario; independent token-rounding
            # model replans are covered elsewhere, not part of this hotfix.
            self.on_model_selected=None
            return real_generate(self,**kw)
        monkeypatch.setattr(GeminiClient,"generate_transcription",generate)
        monkeypatch.setattr("src.coverage_validator.TailActivityAnalyzer.analyze",lambda self,path,offset,duration,threshold:duration)
    result,_,uploads,_,cp=run_audio_case(tmp_path,monkeypatch,minimum=120,fail_count=0,total_duration=868.416,setup=setup)
    assert result==1
    assert [end-start for start,end in seen[1:]]==pytest.approx([254.208,127.104,120,72,60])
    assert all(start==614.208 for start,end in seen[1:])
    assert cp["next_audio_start_us"]==614208000
    assert len(cp["block_metrics"])==1
    assert "EMERGENCY_SIZE_FLOOR_EXHAUSTED" in cp["error_message"]
    assert len({asset.name for asset in uploads})==len(seen)
