import json
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from src.gemini_client import GeminiClient
from src.provider_adapters import TranscriptionModelAdapter,AnnotationSemanticError
from tests.test_transcribe_adapter import native,PROMPT,sdk_client
from tests.test_stage4_audio_retry import run_audio_case

@pytest.mark.parametrize('limit,all_fail',[(4,False),(5,False),(4,True)])
def test_coverage_recovery(tmp_path,monkeypatch,capsys,limit,all_fail):
    real=GeminiClient.generate_transcription;seen=[]
    def setup(config):
        cfg=replace(config,next_target_policy='MAX_FIRST',max_duration_seconds=1200,min_duration_seconds=120,coverage_max_generations=limit)
        monkeypatch.setattr('src.main.load_config',lambda **kw:cfg)
        harness=GeminiClient.generate_transcription
        def init(self,**kw):
            self.model_name=self.requested_model='gemini-3.6-flash'
            self.fallback_models=('gemini-3.5-flash','gemini-3.8-flash');self.effective_model_chain=(self.model_name,)+self.fallback_models
            self.max_retries=1;self.max_transient_retries=0;self.initial_delay_seconds=0
            self.fallback_enabled=True;self.fallback_policy='SPEED_FIRST'
            def call(model,contents,**kw):
                asset,prompt=contents;seen.append((model,asset,list(asset.bounds)))
                d=json.loads(harness(self,gemini_file=asset,user_prompt=prompt))
                start,end=asset.bounds;duration=end-start
                bad=all_fail or (model=='gemini-3.6-flash' and start==0 and duration>120)
                sec=int(start+{1200:717,600:211,300:108,150:83}.get(duration,duration*.5) if bad else end-2)
                d['segments'][0]['timestamp']=f'{sec//3600:02d}:{sec%3600//60:02d}:{sec%60:02d}'
                return NS(text=json.dumps(d),candidates=None,usage_metadata=None)
            self.client=NS(models=NS(generate_content=call))
        monkeypatch.setattr(GeminiClient,'__init__',init);monkeypatch.setattr(GeminiClient,'generate_transcription',real)
        monkeypatch.setattr('src.coverage_validator.TailActivityAnalyzer.analyze',lambda self,path,offset,duration,threshold:duration)
    result,_,_,_,cp=run_audio_case(tmp_path,monkeypatch,setup=setup,fail_count=0,total_duration=1200)
    assert [b for _,_,b in seen[:4]]==[[0,1200],[0,600],[0,300],[0,150]]
    logs=capsys.readouterr().out
    if limit==4:
        assert seen[4][0]=='gemini-3.5-flash' and seen[4][1] is seen[3][1]
        assert 'pending_retry_target=120' in logs and 'COVERAGE_RETRY_EXHAUSTED' in logs
    else:assert seen[4][0]=='gemini-3.6-flash' and seen[4][2]==[0,120]
    if all_fail:
        assert result==1 and cp['next_audio_start_us']==0
        assert 'COVERAGE_UNRESOLVED' in cp['error_message'] and 'NO_ELIGIBLE_FALLBACK_MODEL' in cp['error_message']
    else:
        assert result==0
        metric=cp['block_metrics'][0]
        assert metric['actual_end_offset']==(150 if limit==4 else 120)
        assert metric['physical_generation_count']==metric['coverage_generation_count']==limit
        assert metric['size_retry_count']==0
        assert seen[5][2][0]==metric['actual_end_offset']

@pytest.mark.parametrize('failures',[1,3])
def test_annotation_bounded_retry(tmp_path,monkeypatch,failures):
    real=GeminiClient.generate_transcription;seen=[];clients=[]
    def setup(config):
        harness=GeminiClient.generate_transcription
        def init(self,**kw):
            clients.append(self);self.model_name=self.requested_model='gemini-3.5-transcribe'
            self.fallback_models=('gemini-3.6-flash',);self.effective_model_chain=(self.model_name,)+self.fallback_models
            self.max_retries=1;self.max_transient_retries=0;self.initial_delay_seconds=0
            self.fallback_enabled=True;self.fallback_policy='SPEED_FIRST'
            self._registry().mark(self.model_name,'ELIGIBLE','LAST_CALL_SUCCEEDED')
        monkeypatch.setattr(GeminiClient,'__init__',init)
        def adapter_call(self,model,asset,prompt,config):
            seen.append((model,asset));text=harness(self,gemini_file=asset,user_prompt=prompt)
            self.last_response_metadata={'actual_model':model}
            if len(seen)<=failures:
                bad=native();bad['steps'][0]['content'][0]['annotations'][1]['end_offset']='0.5s'
                TranscriptionModelAdapter().canonical(bad,PROMPT)
            return text
        monkeypatch.setattr(GeminiClient,'_call_with_config_recovery',adapter_call)
        monkeypatch.setattr(GeminiClient,'generate_transcription',real)
    result,_,_,_,cp=run_audio_case(tmp_path,monkeypatch,setup=setup,fail_count=0,total_duration=300)
    assert result==0 and all(a is seen[0][1] for _,a in seen)
    assert [m for m,_ in seen]==['gemini-3.5-transcribe']*(failures if failures==3 else 2)+(['gemini-3.6-flash'] if failures==3 else [])
    c=clients[0];state=c._registry().states['gemini-3.5-transcribe']
    assert not state.quarantined and state.proven_callable and state.state!='ADAPTER_UNAVAILABLE'
    if failures==3:
        c.begin_fresh_block('BLOCK_002')
        assert c.model_name=='gemini-3.5-transcribe'

def test_overlap_and_text_time_order_are_nonblocking(capsys):
    d=native();words=d['steps'][0]['content'][0]['annotations']
    words[0]['end_offset']='3s' # overlaps next word, even with nested end.
    TranscriptionModelAdapter().canonical(d,PROMPT)
    words[1]['start_offset']='0.1s'
    TranscriptionModelAdapter().canonical(d,PROMPT)
    log=capsys.readouterr().out
    assert 'TRANSCRIBE_TEMPORAL_OVERLAP' in log and 'previous_end=3.0' in log
    assert 'Dạ' not in log
