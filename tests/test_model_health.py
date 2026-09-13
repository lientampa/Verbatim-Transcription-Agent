from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.block_builder import SourceBlock
from src.coverage_validator import CoverageResult, CoverageDecision
from src.gemini_client import GeminiClient, GeminiTranscribeError
from src.model_health import ModelHealth
from src.response_parser import ResponseParser
from src.transcriber import GeminiTranscriber
from src.transcript_validator import TranscriptValidator, ExpectedBlockContext
from tests.test_audio_identity import payload
from tests.test_stage4_audio_retry import run_audio_case
from tests.test_stage43_adaptation import ProviderError


def health():
    return ModelHealth(SimpleNamespace(severe_coverage_ratio=.75,model_health_strike_limit=2))


def coverage(ratio):
    return CoverageResult(CoverageDecision.RETRY,"00:01",1200*ratio,"ACTIVE",1200*ratio,
                          tail_gap_ratio=ratio,physical_duration_seconds=1200)


def test_severe_strikes_are_model_specific_and_success_resets_streak():
    h=health()
    assert not h.coverage_failure('Flash',coverage(.899))
    h.success('Flash',600)
    assert h.profile('Flash').successful_blocks==1
    assert h.profile('Flash').consecutive_strikes==0
    assert not h.coverage_failure('Lite',coverage(.899))
    assert h.coverage_failure('Lite',coverage(.907))
    assert h.profile('Lite').degraded and not h.profile('Flash').degraded
    assert not health().profile('Lite').degraded  # New job, fresh health.


def test_ordinary_gap_is_not_health_strike():
    h=health()
    for _ in range(4):
        assert not h.coverage_failure('Flash',coverage(.08))
    assert h.profile('Flash').severe_coverage_failures==0


@pytest.mark.parametrize('timestamp,valid',[('20:00:26',False),('00:20:26',True)])
def test_timestamp_semantics_are_not_rewritten(timestamp,valid):
    data=payload([1,2,3])
    for segment,ts in zip(data['segments'],[timestamp,'00:25:10','00:39:58']):
        segment['timestamp']=ts
    parsed=ResponseParser().parse(json.dumps(data))
    before=deepcopy(parsed)
    ctx=ExpectedBlockContext('j','s','BLOCK_001',None,audio_start=1200,audio_end=2400,source_mode='AUDIO')
    result=TranscriptValidator().validate_block_result(parsed,expected_context=ctx)
    assert result.is_valid==valid and parsed==before
    if not valid:
        assert 'TIMESTAMP_OUT_OF_RANGE' in result.reason_codes
        h=health()
        block=SourceBlock(1,'001',1200,2400,Path('fixture'),1)
        assert not h.structural_failure('Lite',result,block,parsed)
        assert h.structural_failure('Lite',result,block,parsed)
        assert h.profile('Lite').timestamp_semantic_failures==2
        assert h.diagnostics[0]['response']==before.to_dict()


def test_actual_prompt_contains_machine_range_and_timestamp_contract(tmp_path):
    transcriber=GeminiTranscriber(SimpleNamespace(),tmp_path/'prompt')
    prompt=transcriber.build_user_prompt('j','s','BLOCK_002',1,start_offset_seconds=1200,end_offset_seconds=2400)
    for text in ['BLOCK_ABSOLUTE_START_SECONDS=1200','BLOCK_ABSOLUTE_END_SECONDS=2400',
        'BLOCK_ABSOLUTE_START_HHMMSS=00:20:00','BLOCK_ABSOLUTE_END_HHMMSS=00:40:00',
        '20 minutes 26 seconds = 00:20:26','NOT 20:00:26','00:20:03','00:39:58']:
        assert text in prompt


def test_many_timestamp_errors_have_compact_summary_and_full_diagnostics():
    data=payload(list(range(1,44)))
    for seg in data['segments']: seg['timestamp']='20:00:26'
    parsed=ResponseParser().parse(json.dumps(data))
    result=TranscriptValidator().validate_block_result(parsed,expected_context=ExpectedBlockContext(
        'j','s','BLOCK_001',None,audio_start=1200,audio_end=2400,source_mode='AUDIO'))
    assert len(result.errors)==43
    assert len(result.summary())<700
    assert "'TIMESTAMP_OUT_OF_RANGE': 43" in result.summary()


@pytest.mark.parametrize('replacement',[False,True])
@pytest.mark.parametrize('fault',['coverage','timestamp'])
def test_production_circuit_stops_shrink_and_preserves_unconfirmed_start(tmp_path,monkeypatch,replacement,fault):
    real_generate=GeminiClient.generate_transcription
    calls,clients=[],[]
    def setup(config):
        from tests.test_stage5_resume_identity import seed
        seed(config.checkpoint_file_path,config.audio_dir/'source.wav',duration=2400,ends=(1200,))
        cfg=replace(config,gemini_model='Lite',next_target_policy='MAX_FIRST',max_duration_seconds=1200,
                    model_health_strike_limit=2,coverage_max_generations=4,min_duration_seconds=120)
        monkeypatch.setattr('src.main.load_config',lambda **kwargs:cfg)
        original_generate=GeminiClient.generate_transcription
        def init(self,**kwargs):
            clients.append(self)
            self.model_name=self.requested_model='Lite'
            self.fallback_models=('Healthy',) if replacement else ()
            self.fallback_enabled=True
            self.fallback_policy='SPEED_FIRST'
            self.max_retries=1
            self.max_transient_retries=0
            self.initial_delay_seconds=1
            def generate_content(model,contents,**kwargs):
                asset,prompt=contents
                calls.append((model,asset.bounds[:]))
                data=json.loads(original_generate(self,gemini_file=asset,user_prompt=prompt))
                start,end=asset.bounds
                timestamp=int(end-2 if model=='Healthy' else start+(end-start)*.1)
                data['segments'][0]['timestamp']=f'{timestamp//3600:02d}:{timestamp%3600//60:02d}:{timestamp%60:02d}'
                if model=='Lite' and fault=='timestamp':
                    data['segments'][0]['timestamp']='20:00:26'
                return SimpleNamespace(candidates=[SimpleNamespace(finish_reason='STOP',content=SimpleNamespace(
                    parts=[SimpleNamespace(text=json.dumps(data),thought=False)]))],usage_metadata=None)
            self.client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        monkeypatch.setattr(GeminiClient,'__init__',init)
        monkeypatch.setattr(GeminiClient,'generate_transcription',real_generate)
        monkeypatch.setattr('src.coverage_validator.TailActivityAnalyzer.analyze',lambda self,path,offset,duration,threshold:duration)
    result,_,_,_,cp=run_audio_case(tmp_path,monkeypatch,setup=setup,fail_count=0,total_duration=2400)
    retry_end=2400 if fault=='timestamp' else 1800
    assert calls[:2]==[('Lite',[1200,2400]),('Lite',[1200,retry_end])]
    assert clients[0].model_health.profile('Lite').degraded
    if replacement:
        assert result==0 and calls[2]==('Healthy',[1200,retry_end])
        assert sum(model=='Lite' for model,_ in calls)==2
        assert cp['block_metrics'][1]['actual_start_offset']==1200
        assert clients[0].model_health.profile('Healthy').successful_blocks>=1
    else:
        assert result==1 and len(calls)==2
        assert cp['next_audio_start_us']==1200_000000 and len(cp['block_metrics'])==1
        assert 'MODEL_OUTPUT_UNAVAILABLE' in cp['error_message']


def test_exhausted_provider_models_cannot_be_reentered():
    h=health()
    h.begin_chain('BLOCK_002')
    h.visited.update(['Flash36','Flash38','Lite'])
    h.provider_failure('Flash36',dict(error_code=429,provider_retry_after_seconds=26,action='MODEL_FALLBACK',reason='QUOTA'))
    h.provider_failure('Flash38',dict(error_code=503,provider_retry_after_seconds=None,action='MODEL_FALLBACK',reason='EXHAUSTED'))
    h.coverage_failure('Lite',coverage(.899))
    h.coverage_failure('Lite',coverage(.907))
    client=GeminiClient.__new__(GeminiClient)
    client.model_health=h
    client.requested_model='Flash36'
    client.model_name='Lite'
    client.fallback_models=('Flash38','Lite','Flash36')
    client.fallback_enabled=True
    with pytest.raises(GeminiTranscribeError,match='MODEL_OUTPUT_UNAVAILABLE'):
        client.advance_after_output_failure()
    assert not h.eligible('Flash36') and not h.eligible('Flash38')
    assert h.profile('Flash36').severe_coverage_failures==0
