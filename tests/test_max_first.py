from dataclasses import replace
import json

import pytest

from src.block_builder import DabbAudioOrchestrator
from src.config import load_config
from src.coverage_validator import TailActivityAnalyzer
from src.gemini_client import GeminiClient
from tests.test_stage4_audio_retry import run_audio_case


def run_max_case(tmp_path, monkeypatch, safe, total, always_fail_max=False):
    def setup(config):
        config = replace(config, next_target_policy="MAX_FIRST", max_duration_seconds=1200,
            initial_target_input_tokens=3000, min_duration_seconds=300, coverage_max_generations=3)
        monkeypatch.setattr("src.main.load_config", lambda **kwargs: config)
        original = GeminiClient.generate_transcription
        def generate(self, gemini_file, **kwargs):
            data = json.loads(original(self, gemini_file=gemini_file, **kwargs))
            start, end = gemini_file.bounds
            fails = end-start > safe and (always_fail_max or data["block_id"] == "BLOCK_001")
            seconds = int(start+1 if fails else end-2)
            data["segments"][0]["timestamp"] = f"{seconds//60:02d}:{seconds%60:02d}"
            return json.dumps(data)
        monkeypatch.setattr(GeminiClient, "generate_transcription", generate)
        monkeypatch.setattr(TailActivityAnalyzer, "analyze", lambda self,path,offset,duration,threshold: duration)
    return run_audio_case(tmp_path,monkeypatch,setup=setup,fail_count=0,total_duration=total)


@pytest.mark.parametrize("safe", [600,300])
def test_fallback_commit_then_next_max_is_contiguous(tmp_path,monkeypatch,safe):
    result,_,_,calls,cp=run_max_case(tmp_path,monkeypatch,safe,safe+2400)
    assert result==0
    prefix=[[0,1200],[0,600]] + ([[0,300]] if safe==300 else [])
    assert [c[0].bounds for c in calls] == prefix+[[safe,safe+1200],[safe+1200,safe+2400]]
    assert cp['block_metrics'][0]['actual_end_offset']==safe
    assert all(a['actual_end_offset']==b['actual_start_offset'] for a,b in zip(cp['block_metrics'],cp['block_metrics'][1:]))


def test_guard_counts_fresh_failures_once_and_recovers(tmp_path,monkeypatch):
    result,_,_,calls,cp=run_max_case(tmp_path,monkeypatch,300,4200,True)
    assert result==0
    durations=[c[0].bounds[1]-c[0].bounds[0] for c in calls]
    assert durations[:9]==[1200,600,300]*3
    assert durations[9:12]==[300,300,1200]
    assert all(a['actual_end_offset']==b['actual_start_offset'] for a,b in zip(cp['block_metrics'],cp['block_metrics'][1:]))


def test_eof_caps_max_proposal(tmp_path,monkeypatch):
    result,_,_,calls,cp=run_max_case(tmp_path,monkeypatch,600,430)
    assert result==0 and [c[0].bounds for c in calls]==[[0,430]]
    assert cp['next_audio_start_us']==430_000000


@pytest.mark.parametrize('limit', ['input','context','output'])
def test_max_respects_hard_budgets(tmp_path,monkeypatch,limit):
    monkeypatch.setenv('GEMINI_API_KEY','mock-key')
    cfg=replace(load_config(tmp_path),next_target_policy='MAX_FIRST',min_duration_seconds=30,
        max_duration_seconds=1200,initial_target_input_tokens=3000)
    if limit=='input': cfg=replace(cfg,max_block_tokens=1000)
    if limit=='context': cfg=replace(cfg,model_context_limit=3000,context_safety_margin=2000)
    if limit=='output': cfg=replace(cfg,model_max_output_tokens=2200,output_safety_margin=1000)
    planner=DabbAudioOrchestrator(tmp_path/'source.wav',cfg)
    assert 30<=planner.next_target_duration_seconds<=400
    maximum=planner.max_target_duration_seconds
    planner.on_coverage_failure(maximum)
    assert planner.max_target_duration_seconds==maximum


def test_model_guard_is_not_shared(tmp_path,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','mock-key')
    cfg=replace(load_config(tmp_path),gemini_model='Lite',next_target_policy='MAX_FIRST',
        max_duration_seconds=1200,initial_target_input_tokens=3000)
    planner=DabbAudioOrchestrator(tmp_path/'source.wav',cfg)
    planner.model_profiles['Lite'].update(suppressed=True,max_failure_streak=3,safe_duration=300)
    assert planner.next_target_duration_seconds==300
    planner.select_model('Flash')
    assert planner.next_target_duration_seconds==1200
    planner.select_model('Lite')
    assert planner.next_target_duration_seconds==300
