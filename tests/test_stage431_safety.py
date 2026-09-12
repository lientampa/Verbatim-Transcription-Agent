from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.block_builder import SourceBlock
from src.coverage_validator import CoverageValidator, CoverageDecision, TailActivityAnalyzer
from src.gemini_client import GeminiClient
from src.response_parser import ResponseParser
from src.transcript_validator import TranscriptValidator, ExpectedBlockContext
from tests.test_audio_identity import payload
from tests.test_stage4_audio_retry import run_audio_case
from tests.test_stage43_adaptation import planner, passed, client, ProviderError


@pytest.mark.parametrize('duration,gap,small', [(1200,6,True),(600,10,True),(300,6,True),
    (150,92,False),(120,97,False),(120,5,True),(120,12,True),(120,13,False)])
def test_relative_small_gap_matrix(duration, gap, small):
    cfg = SimpleNamespace(coverage_tail_gap_threshold_sec=120, coverage_small_gap_max_ratio=.1,
        coverage_active_tail_min_sec=30, coverage_silence_threshold_db=-40, coverage_shrink_factor=.5)
    start = 600
    stamp = start + duration - gap
    data = payload([1])
    data['segments'][0]['timestamp'] = f'{stamp // 60:02d}:{stamp % 60:02d}'
    parsed = ResponseParser().parse(json.dumps(data))
    before = deepcopy(parsed)
    analyses = []
    def analyze(path, offset, seconds, threshold):
        analyses.append((offset,seconds))
        return seconds
    result = CoverageValidator(cfg, SimpleNamespace(analyze=analyze)).validate(
        SourceBlock(1,'001',start,start+duration,Path('fixture'),1), parsed)
    assert result.tail_gap_ratio == pytest.approx(gap/duration)
    assert result.physical_duration_seconds == duration
    assert (result.tail_activity == 'NOT_ANALYZED_SMALL_GAP') == small
    assert bool(analyses) != small
    assert parsed == before
    if gap in (92,97):
        assert result.decision == CoverageDecision.RETRY
        assert analyses == [(duration-gap,gap)]


def test_absolute_limit_still_required_and_silent_tail_allowed():
    cfg = SimpleNamespace(coverage_tail_gap_threshold_sec=5, coverage_small_gap_max_ratio=.1,
        coverage_active_tail_min_sec=30, coverage_silence_threshold_db=-40, coverage_shrink_factor=.5)
    transcript = ResponseParser().parse(json.dumps(payload([1])))  # 09:52, tail 8 seconds
    result = CoverageValidator(cfg, SimpleNamespace(analyze=lambda *args: 0)).validate(
        SourceBlock(1,'001',0,600,Path('fixture'),1), transcript)
    assert result.decision == CoverageDecision.WARNING
    transcript.segments[0].timestamp = '00:23'
    result = CoverageValidator(cfg, SimpleNamespace(analyze=lambda *args: 0)).validate(
        SourceBlock(1,'001',0,120,Path('fixture'),1), transcript)
    assert result.decision == CoverageDecision.WARNING
    assert result.tail_gap_ratio == pytest.approx(97/120)


@pytest.mark.parametrize('declared', [128, 1])
def test_audio_endpoint_advisory_preserves_declared_value_and_text_stays_strict(declared):
    data = payload([121,122,123])
    data['last_source_index'] = declared
    parsed = ResponseParser().parse(json.dumps(data))
    original = deepcopy(parsed)
    context = ExpectedBlockContext('j','s','BLOCK_001',121,audio_end=600,source_mode='AUDIO')
    result = TranscriptValidator().validate_block_result(parsed, expected_context=context)
    assert result.is_valid
    assert any('AUDIO_LAST_SEGMENT_ORDINAL_MISMATCH' in w for w in result.warnings)
    assert parsed == original
    result = TranscriptValidator().validate_block_result(parsed, expected_context=replace(context,source_mode='TEXT_TIMESTAMP'))
    assert not result.is_valid and 'LAST_SOURCE_INDEX_MISMATCH' in result.reason_codes


@pytest.mark.parametrize('active', [False, True])
def test_endpoint_warning_and_relative_coverage_production(tmp_path, monkeypatch, active):
    responses = []
    def setup(config):
        cfg = replace(config, min_duration_seconds=120, initial_target_input_tokens=300,
                      validator_max_retries=3, coverage_max_generations=3)
        monkeypatch.setattr('src.main.load_config',lambda **kwargs: cfg)
        original = GeminiClient.generate_transcription
        def generate(self, **kwargs):
            data = json.loads(original(self,**kwargs))
            data.update(first_source_index=122,last_source_index=128,
                segments=[dict(source_index=122,text='Vâng.',timestamp='00:23' if active else '01:58')])
            self.last_response_metadata={'actual_model':cfg.gemini_model,'finish_reason':'STOP'}
            responses.append(deepcopy(data))
            return json.dumps(data)
        monkeypatch.setattr(GeminiClient,'generate_transcription',generate)
        monkeypatch.setattr(TailActivityAnalyzer,'analyze',lambda self,path,offset,duration,threshold:duration)
    result,_,_,calls,checkpoint=run_audio_case(tmp_path,monkeypatch,setup=setup,fail_count=0,total_duration=120)
    assert len(calls)==1 and responses[0]['last_source_index']==128
    if active:
        assert result==1 and checkpoint['next_audio_start_us']==0
        assert not checkpoint['confirmed_segments']
        assert 'COVERAGE_RETRY' in checkpoint['error_message']
    else:
        assert result==0 and checkpoint['next_audio_start_us']==120_000000
        metric=checkpoint['block_metrics'][0]
        assert metric['received_last_source_index']==128 and metric['final_segment_ordinal']==122
        assert metric['endpoint_diagnostics'][0]['reason_code']=='AUDIO_LAST_SEGMENT_ORDINAL_MISMATCH'
        assert checkpoint['confirmed_segments'][0]['source_index']==122
        from src.main import run_pipeline
        assert run_pipeline(base_dir=tmp_path)==0  # Reload validates derived ordinal, not declared 128.
        assert len(calls)==1


@pytest.mark.parametrize('start,duration,progress',[(300,150,58),(600,120,23)])
def test_real_absolute_short_block_retry_and_learned_size(tmp_path,monkeypatch,start,duration,progress):
    count=[]
    def setup(config):
        from tests.test_stage5_resume_identity import seed
        seed(config.checkpoint_file_path,config.audio_dir/'source.wav',duration=start+duration,ends=(start,))
        cfg=replace(config, min_duration_seconds=120,initial_target_input_tokens=int(duration*2.5),
            max_duration_seconds=duration,unknown_model_duration_sec=120,coverage_max_generations=3)
        monkeypatch.setattr('src.main.load_config',lambda **kwargs:cfg)
        original=GeminiClient.generate_transcription
        def generate(self,gemini_file,**kwargs):
            data=json.loads(original(self,gemini_file=gemini_file,**kwargs))
            count.append(1)
            ts=start+progress if len(count)==1 else int(gemini_file.bounds[1])-2
            data['segments'][0]['timestamp']=f'{ts//60:02d}:{ts%60:02d}'
            return json.dumps(data)
        monkeypatch.setattr(GeminiClient,'generate_transcription',generate)
        monkeypatch.setattr(TailActivityAnalyzer,'analyze',lambda self,path,offset,duration,threshold:duration)
    result,_,_,calls,cp=run_audio_case(tmp_path,monkeypatch,setup=setup,fail_count=0,total_duration=start+duration)
    assert calls[0][0].bounds==[start,start+duration]
    if duration==120:
        assert result==1 and cp['next_audio_start_us']==600_000000 and len(cp['block_metrics'])==1
    else:
        assert result==0 and calls[1][0].bounds==[300,420]
        assert cp['block_metrics'][1]['coverage_generation']==2


@pytest.mark.parametrize('duration',[300,150])
def test_small_success_target_is_retained(planner,duration):
    planner.on_coverage_success(duration,passed())
    assert planner.current_block_duration_seconds==duration


def test_backoff_and_selection_cause_are_distinct(monkeypatch):
    obj,calls,waits=client('PREFER_WAIT',[ProviderError(),True],monkeypatch)
    obj.generate_transcription('asset','verbatim')
    assert obj.last_response_metadata['backoff_before_call_seconds']==19
    assert obj.last_response_metadata['retry_after_seconds'] is None
    obj,calls,waits=client('PREFER_FALLBACK',[ProviderError(429),ProviderError(503,None),True,True],monkeypatch)
    obj.generate_transcription('asset','verbatim')
    assert calls==['A','B','C']
    assert obj.last_response_metadata['fallback_reason']=='503'
    obj.generate_transcription('asset2','verbatim')
    assert obj.last_response_metadata['fallback_reason']=='503'
    assert obj.last_response_metadata['backoff_before_call_seconds']==0
