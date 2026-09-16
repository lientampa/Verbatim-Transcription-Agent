import json
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
from src.provider_adapters import TranscriptionModelAdapter,ProviderAdapterError,AnnotationSemanticError
from src.response_parser import ResponseParser
from src.coverage_validator import CoverageValidator
from src.config import load_config
from tests.test_transcribe_adapter import PROMPT


def fixture(times):
    words=[];text='';pos=0
    for i,(start,end,speaker) in enumerate(times):
        value=chr(65+i)
        words.append(dict(type='word_info',text=value,start_index=pos,end_index=pos+1,start_offset=f'{start}s',end_offset=f'{end}s',speaker=speaker))
        text+=value+' ';pos+=2
    return dict(status='completed',steps=[dict(type='model_output',content=[dict(type='text',text=text.rstrip(),annotations=words)])])

def prompt(start=0,end=1200):return PROMPT.replace('614.208',str(start)).replace('741.312',str(end))

@pytest.mark.parametrize('times',[
    [(638.3,638.4,'spk:2'),(631.0,631.2,'spk:0')],
    [(96.1,96.3,'spk:1'),(95.5,95.7,'spk:0')],
    [(100,110,'spk:0'),(95,105,'spk:0'),(106,108,'spk:0')]])
def test_text_order_preserved(times,capsys):
    d=fixture(times);before=json.dumps(d)
    adapter=TranscriptionModelAdapter();result=json.loads(adapter.canonical(d,prompt()))
    assert ' '.join(s['text'] for s in result['segments'])==' '.join(chr(65+i) for i in range(len(times)))
    assert json.dumps(d)==before
    assert adapter.evidence.endpoint==max(t[1] for t in times)
    assert 'TRANSCRIBE_TEMPORAL_OVERLAP' in capsys.readouterr().out

@pytest.mark.parametrize('start,end',[('1300','1301'),('100','90'),('-0.5','1'),('NaN','1'),('Inf','1')])
def test_hard_bounds(start,end):
    with pytest.raises(ProviderAdapterError):TranscriptionModelAdapter().canonical(fixture([(start,end,'spk:0')]),prompt())

@pytest.mark.parametrize('change',[{'start_index':2,'end_index':3},{'start_index':0,'end_index':999},{'start_index':1,'end_index':2}])
def test_bad_text_indexes(change):
    d=fixture([(10,11,'spk:0'),(12,13,'spk:1')]);d['steps'][0]['content'][0]['annotations'][0].update(change)
    with pytest.raises(AnnotationSemanticError):TranscriptionModelAdapter().canonical(d,prompt())

def test_endpoint_and_flash_isolation(tmp_path,monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY","mock-key")
    a=TranscriptionModelAdapter()
    t=ResponseParser().parse(a.canonical(fixture([(1180,1185,'spk:0'),(95,105,'spk:1')]),prompt(1200,2400)))
    assert a.evidence.endpoint==2385
    t._wordinfo_evidence=a.evidence
    cfg=load_config(tmp_path);v=CoverageValidator(cfg)
    v.analyzer=NS(analyze=lambda path,offset,duration,threshold:duration)
    block=NS(start_time_seconds=1200,end_time_seconds=2400,file_path=tmp_path/'fake')
    assert v.validate(block,t).tail_gap_seconds==15
    del t._wordinfo_evidence
    assert v.validate(block,t).tail_gap_seconds==20 # generic max segment START unchanged
    t._wordinfo_evidence=a.evidence
    t.segments[0].timestamp='00:20:00' # stale evidence cannot authorize altered payload
    assert v.validate(block,t).tail_gap_seconds==1105


def test_live_path_overlap_commit_and_render(tmp_path,monkeypatch):
    from tests.test_stage4_audio_retry import run_audio_case
    from src.gemini_client import GeminiClient
    real=GeminiClient.generate_transcription;clients=[];requests=[]
    def setup(config):
        cfg=replace(config,gemini_model='gemini-3.5-transcribe',next_target_policy='MAX_FIRST',max_duration_seconds=1200)
        monkeypatch.setattr('src.main.load_config',lambda **kw:cfg)
        harness=GeminiClient.generate_transcription
        upload=GeminiClient.upload_audio
        def upload_native(self,audio_path):
            asset=upload(self,audio_path);asset.uri=asset.name;asset.mime_type='audio/wav';self.asset=asset;return asset
        monkeypatch.setattr(GeminiClient,'upload_audio',upload_native)
        def init(self,**kw):
            clients.append(self);self.model_name=self.requested_model='gemini-3.5-transcribe'
            self.effective_model_chain=(self.model_name,);self.fallback_models=()
            self.max_retries=1;self.max_transient_retries=0;self.initial_delay_seconds=0
            self.fallback_enabled=True;self.fallback_policy='SPEED_FIRST'
            def native_call(**kw):
                requests.append(kw)
                # Register harness expected block bounds without changing native result.
                harness(self,gemini_file=self.asset,user_prompt=prompt().replace('BLOCK_004','BLOCK_001')+'\n- first_source_index: 1')
                return fixture([(638.3,1185,'spk:2'),(631,631.2,'spk:0')])
            self.client=NS(interactions=NS(create=native_call))
        monkeypatch.setattr(GeminiClient,'__init__',init);monkeypatch.setattr(GeminiClient,'generate_transcription',real)
    result,_,_,_,cp=run_audio_case(tmp_path,monkeypatch,setup=setup,fail_count=0,total_duration=1200)
    assert result==0 and len(requests)==1 and cp['next_audio_start_us']==1200000000
    m=cp['block_metrics'][0]
    assert m['structural_retry_count']==0 and m['coverage']['tail_gap_seconds']==15
    assert m['provider_metadata']['coverage_endpoint_seconds']==1185
    state=clients[0]._registry().states['gemini-3.5-transcribe']
    assert state.provider_callable and state.validated_success


def test_provider_callable_before_semantic_success(monkeypatch):
    from tests.test_transcribe_adapter import sdk_client
    c,_=sdk_client(monkeypatch,[])
    c.model_name='gemini-3.5-transcribe'
    c.client.interactions.create=lambda **kw:fixture([(100,90,'spk:0')])
    with pytest.raises(AnnotationSemanticError):c.generate_transcription(NS(uri='files/fake',mime_type='audio/wav'),'verbatim',prompt())
    state=c._registry().states[c.model_name]
    assert state.provider_callable and state.proven_callable and not state.validated_success and not state.quarantined
