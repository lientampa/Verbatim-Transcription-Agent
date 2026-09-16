import json
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from src.gemini_client import GeminiClient
from src.model_policy import PREFERRED_MODEL_CHAIN
from tests.test_stage4_audio_retry import run_audio_case
from tests.test_stage5_resume_identity import seed
from tests.test_transcribe_adapter import sdk_client
from tests.test_stage43_adaptation import ProviderError

@pytest.mark.parametrize('mode',['retry_success','fallback','two_exhausted','all_exhausted'])
def test_structural_physical_recovery(tmp_path,monkeypatch,capsys,mode):
    real=GeminiClient.generate_transcription
    seen=[];clients=[];counts={}
    def setup(config):
        seed(config.checkpoint_file_path,config.audio_dir/'source.wav',duration=5100,ends=(1200,2400,3600,3900))
        cfg=replace(config,next_target_policy='MAX_FIRST',max_duration_seconds=1200,validator_max_retries=3)
        monkeypatch.setattr('src.main.load_config',lambda **kw:cfg)
        harness=GeminiClient.generate_transcription
        def init(self,**kw):
            clients.append(self)
            self.model_name=self.requested_model='gemini-3.6-flash'
            self.fallback_models=PREFERRED_MODEL_CHAIN[3:]
            self.effective_model_chain=PREFERRED_MODEL_CHAIN[2:]
            self.fallback_enabled=True;self.fallback_policy='SPEED_FIRST'
            self.max_retries=1;self.max_transient_retries=0;self.initial_delay_seconds=0
            def call(model,contents,**kw):
                asset,prompt=contents
                state=json.loads(config.checkpoint_file_path.read_text(encoding='utf-8'))
                assert state['next_audio_start_us']==3900000000
                seen.append((model,asset));counts[model]=counts.get(model,0)+1
                data=json.loads(harness(self,gemini_file=asset,user_prompt=prompt))
                n=counts[model]
                failing=model.endswith('3.6-flash') or mode=='all_exhausted' or (mode=='two_exhausted' and model.endswith('3.5-flash'))
                if mode=='retry_success' and n>1: failing=False
                if failing and n==1:return NS(text='{',candidates=None,usage_metadata=None)
                if failing and n==2:
                    base=data['segments'][0]
                    data['segments']=[dict(base,source_index=i+1,timestamp='01:25:33') for i in range(21)]
                elif failing:
                    base=data['segments'][0]
                    data['segments']=[dict(base,timestamp='01:07:33'),dict(base,source_index=base['source_index']+1,timestamp='01:03:05')]
                else:data['segments'][0]['timestamp']='01:24:58'
                return NS(text=json.dumps(data),candidates=None,usage_metadata=None)
            self.client=NS(models=NS(generate_content=call))
        monkeypatch.setattr(GeminiClient,'__init__',init)
        monkeypatch.setattr(GeminiClient,'generate_transcription',real)
    result,slices,uploads,_,cp=run_audio_case(tmp_path,monkeypatch,setup=setup,fail_count=0,total_duration=5100)
    assert all(asset is seen[0][1] for _,asset in seen)
    assert seen[0][1].bounds==[3900,5100]
    assert len(slices)==len(uploads)==1
    assert counts['gemini-3.6-flash']==(2 if mode=='retry_success' else 3)
    h=clients[0].model_health.profile('gemini-3.6-flash')
    assert h.structural_generation_failures>=1
    if mode!='retry_success':assert h.timestamp_semantic_failures==2
    if mode=='all_exhausted':
        assert result==1 and cp['next_audio_start_us']==3900000000
        assert 'NO_ELIGIBLE_FALLBACK_MODEL' in cp['error_message']
        assert cp['error_message'].count('MODEL_STRUCTURAL_RETRY_EXHAUSTED')==3
    else:
        assert result==0 and cp['next_audio_start_us']==5100000000
        metric=cp['block_metrics'][-1]
        assert metric['size_retry_count']==0
        assert metric['physical_generation_count']==metric['coverage_generation_count']==1
    assert list(counts)==list(PREFERRED_MODEL_CHAIN[2:2+len(counts)])

@pytest.mark.parametrize('delay',[None,57])
def test_quota_recovers_only_fresh_block(monkeypatch,delay):
    now=[100.0]
    monkeypatch.setattr('src.gemini_client.time.monotonic',lambda:now[0])
    c,_=sdk_client(monkeypatch,[])
    c.quota_cooldown_seconds=30
    m='gemini-3.5-transcribe'
    c._registry().mark(m,'ELIGIBLE','LAST_CALL_SUCCEEDED')
    c.begin_fresh_block('BLOCK_001')
    decision=dict(error_code=429,action='MODEL_FALLBACK',failure_reason='PROVIDER_429',provider_retry_after_seconds=delay,reason='QUOTA')
    c._record_failure(m,ProviderError(),decision)
    s=c._registry().states[m]
    assert s.state=='QUOTA_LIMITED' and not s.quarantined and s.proven_callable
    assert s.blocked_until==100+(delay or 30)
    c.model_name='gemini-3.6-flash'
    c.begin_fresh_block('BLOCK_002')
    assert not c._eligible(m)
    now[0]=200
    c.begin_fresh_block('BLOCK_002')
    assert not c._eligible(m)
    c.begin_fresh_block('BLOCK_003')
    assert c.model_name==m and c._eligible(m)
    assert c._registry().states[m].proven_callable

def test_quota_health_and_quarantine_do_not_reenter_mid_chain(monkeypatch):
    from tests.test_model_health import health
    now=[100.0]
    monkeypatch.setattr('src.gemini_client.time.monotonic',lambda:now[0])
    c,_=sdk_client(monkeypatch,[]);c.model_health=health()
    c.begin_fresh_block('A')
    m='gemini-3.5-transcribe'
    d=dict(error_code=429,action='MODEL_FALLBACK',failure_reason='PROVIDER_429',provider_retry_after_seconds=57,reason='QUOTA')
    c._record_failure(m,ProviderError(),d);c.model_health.provider_failure(m,d)
    c._registry().mark('gemini-2.5-flash','UNAVAILABLE','MODEL_UNAVAILABLE',True,True)
    c.model_name='gemini-3.6-flash'
    c.advance_after_output_failure('MODEL_STRUCTURAL_RETRY_EXHAUSTED')
    assert c.model_name=='gemini-3.5-flash'
    now[0]=200
    c.begin_fresh_block('A')
    assert not c._eligible(m)
    c.begin_fresh_block('B')
    assert c.model_name==m and c._eligible(m)
    assert not c._eligible('gemini-2.5-flash')
