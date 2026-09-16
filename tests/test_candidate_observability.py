import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from google.genai import types
from experiments.candidate_001 import run as runner
from experiments.candidate_001.observability import Diagnostics, inspect_state


def events(work):
    return [json.loads(line) for line in (work/'diagnostic_events.jsonl').read_text().splitlines()]


def setup_run(tmp_path, monkeypatch, pipeline):
    work=tmp_path/'run';work.mkdir()
    for folder in ('prompts','schemas','state','output'):(work/folder).mkdir()
    (work/'prompts/system_prompt.txt').write_text(runner.VERSION,encoding='utf-8')
    (work/'schemas/transcription_result.schema.json').write_text('{}')
    (work/'state/checkpoint.json').write_text('{"next_audio_start_us":0,"error_message":"PROVIDER_UNAVAILABLE"}')
    monkeypatch.setattr(runner,'prepare',lambda root:work)
    monkeypatch.setattr('src.main.run_pipeline',pipeline)
    return work


@pytest.mark.parametrize('failure',[None,ValueError('api-key secret; Authorization: token; full prompt; human reference'),KeyboardInterrupt()])
def test_request_lifecycle_preserves_return_error_counters_and_checkpoint(tmp_path,failure):
    (tmp_path/'state').mkdir();cp=tmp_path/'state/checkpoint.json';cp.write_text('{"next_audio_start_us":0}')
    before=cp.read_bytes();counters=dict(provider_attempt=7,size_retry_count=0,coverage_retry_count=2,structural_retry_count=0,fidelity_retry_count=0)
    snapshot=dict(counters);diag=Diagnostics(tmp_path)
    inner=Mock();result=object()
    if failure is None:inner.generate.return_value=result
    else:inner.generate.side_effect=failure
    config=types.GenerateContentConfig(system_instruction=runner.VERSION)
    adapter=runner.GuardedAdapter(inner,'generate_content',runner.VERSION,lambda event:None,diag,lambda:counters)
    if failure is None:
        assert adapter.generate(None,'gemini-3.6-flash',None,'private user prompt',config) is result
    else:
        with pytest.raises(type(failure)) as caught:adapter.generate(None,'gemini-3.6-flash',None,'private user prompt',config)
        assert caught.value is failure
    kinds=[e['event'] for e in events(tmp_path)]
    assert kinds[-2:]==['REQUEST_START','REQUEST_ERROR' if failure else 'REQUEST_END']
    assert events(tmp_path)[-1]['elapsed_ms']>=0
    assert counters==snapshot and cp.read_bytes()==before
    text=(tmp_path/'diagnostic_events.jsonl').read_text()
    for private in ('api-key secret','Authorization: token','full prompt','human reference','private user prompt'):
        assert private not in text
    if failure:assert events(tmp_path)[-1]['traceback']


@pytest.mark.parametrize('error,reason,state',[(KeyboardInterrupt(),'KEYBOARD_INTERRUPT','RUN_INTERRUPTED'),
    (RuntimeError('application'),'APPLICATION_EXCEPTION','RUN_FAILED'),
    (runner.ExperimentStop(runner.INVALID),'EXPERIMENT_STOP','RUN_INTERRUPTED')])
def test_run_termination(tmp_path,monkeypatch,error,reason,state):
    def fail(**kw):raise error
    work=setup_run(tmp_path,monkeypatch,fail)
    before=(work/'state/checkpoint.json').read_bytes()
    if isinstance(error,runner.ExperimentStop):assert runner.run(True,tmp_path)==2
    else:
        with pytest.raises(type(error)) as caught:runner.run(True,tmp_path)
        assert caught.value is error
    result=inspect_state(work)
    assert result['final_state']==state and result['termination_reason']==reason
    assert result['checkpoint_audio_end']==0 and not result['candidate_artifact_exists']
    assert (work/'state/checkpoint.json').read_bytes()==before


def test_pipeline_failure_reason(tmp_path,monkeypatch):
    work=setup_run(tmp_path,monkeypatch,lambda **kw:1)
    assert runner.run(True,tmp_path)==2
    state=inspect_state(work)
    assert state['final_state']=='RUN_FAILED'
    assert state['engine_failure_reason_codes']==['PROVIDER_UNAVAILABLE']


def test_evaluation_failure_preserves_valid_transcript(tmp_path,monkeypatch):
    work=setup_run(tmp_path,monkeypatch,lambda **kw:0)
    transcript=work/'output/transcript.txt';transcript.write_text('valid candidate')
    error=ValueError('evaluation')
    def fail(*a):raise error
    monkeypatch.setattr(runner,'compare',fail)
    with pytest.raises(ValueError) as caught:runner.run(True,tmp_path)
    assert caught.value is error
    assert inspect_state(work)['termination_reason']=='EVALUATION_FAILED'
    assert transcript.read_text()=='valid candidate'
    assert 'TRANSCRIPTION_COMPLETED' in [e['event'] for e in events(work)]


def test_normal_completion(tmp_path,monkeypatch):
    work=setup_run(tmp_path,monkeypatch,lambda **kw:0)
    monkeypatch.setattr(runner,'compare',lambda *a:{'test':True})
    assert runner.run(True,tmp_path)==0
    assert inspect_state(work)['final_state']=='RUN_COMPLETED'


def test_write_failure_never_masks_original_exception(tmp_path,monkeypatch):
    diag=Diagnostics(tmp_path)
    def denied(*a,**kw):raise OSError('disk full')
    monkeypatch.setattr(Path,'open',denied)
    original=ValueError('provider failure')
    def fail():raise original
    with pytest.raises(ValueError) as caught:
        with diag.lifecycle():diag.request(fail,attempt=1)
    assert caught.value is original
    assert diag.write_failures>0


def test_incomplete_run_is_identifiable(tmp_path):
    Diagnostics(tmp_path)
    assert inspect_state(tmp_path)['inspection']=='RUN_INCOMPLETE_NO_FINALIZATION'
