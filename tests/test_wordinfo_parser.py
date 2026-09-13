import copy
import json
import pytest
from src.provider_adapters import seconds, word_timing, TranscriptionModelAdapter, ProviderAdapterError
from tests.test_transcribe_adapter import native, PROMPT, sdk_client

@pytest.mark.parametrize('raw,expected', [('0s',0),('0.001s',.001),('0.010s',.01),('0.100s',.1),('0.450s',.45),('0.999s',.999),('1s',1),('1.001s',1.001),('1.250s',1.25),('12.345s',12.345),('60.500s',60.5),('120.000s',120),('367.125s',367.125),('868.416s',868.416)])
def test_duration(raw,expected):
    assert seconds(raw)==pytest.approx(expected)

@pytest.mark.parametrize('raw',['abc','12','12ms','',None,'-1s','NaNs','9'*400+'s'])
def test_invalid_duration(raw):
    with pytest.raises(ProviderAdapterError): seconds(raw)

@pytest.mark.parametrize('base,duration,start,end,expected',[(0,868.416,'0.100s','0.450s',(.1,.45)),(600,300,'100.000s','101.000s',(700,701)),(741.312,120,'10.000s','10.500s',(751.312,751.812))])
def test_explicit_fields_and_nonzero(base,duration,start,end,expected):
    word=dict(start_offset=start,end_offset=end,start_index=0,end_index=450,end=99999)
    original=copy.deepcopy(word)
    result=word_timing(word,base,duration,4)
    assert result[2:]==pytest.approx(expected)
    assert word==original

@pytest.mark.parametrize('end,accepted',[('120s',True),('120.000001s',True),('120.001s',False),('120.003s',False),('130s',False),('150s',False),('0s',False)])
def test_bounds(end,accepted,capsys):
    word=dict(text='PRIVATE WORD',start_offset='1s',end_offset=end,end_index=450)
    if accepted: word_timing(word,0,120,4)
    else:
        with pytest.raises(ProviderAdapterError,match='END_OFFSET_OUT_OF_BOUNDS'):word_timing(word,0,120,4)
        log=capsys.readouterr().out
        assert 'TRANSCRIBE_WORDINFO_PARSE_ERROR' in log and 'word_index=4' in log and 'end_index=450' in log
        assert 'PRIVATE WORD' not in log

def test_real_duration_text_preserved():
    raw='Hello,  hello... hello! hello? hello [???]'
    words=[dict(type='word_info',text=w,speaker='spk_1',start_offset=f'{i}.100s',end_offset=f'{i}.450s') for i,w in enumerate(['Hello']+['hello']*4)]
    d=dict(status='completed',output_text=raw,steps=[dict(type='model_output',content=[dict(type='text',text=raw,annotations=[dict(type='citation',end_index=99999)]+words)])])
    prompt=PROMPT.replace('614.208','0').replace('741.312','868.416')
    out=json.loads(TranscriptionModelAdapter().canonical(d,prompt))
    assert out['segments'][0]['text']==raw
    assert out['segments'][0]['timestamp']=='00:00:00'

def test_missing_offsets_and_backwards():
    d=native(); words=d['steps'][0]['content'][0]['annotations']
    words[1]['start_offset']='0.5s'
    with pytest.raises(ProviderAdapterError,match='NON_MONOTONIC_START'):TranscriptionModelAdapter().canonical(d,PROMPT)
    words[0].pop('start_offset')
    with pytest.raises(ProviderAdapterError,match='TIMESTAMP_UNSUPPORTED'):TranscriptionModelAdapter().canonical(d,PROMPT)

def test_discovered_then_success(monkeypatch):
    c,_=sdk_client(monkeypatch,[True])
    assert c._registry().states['gemini-2.5-flash'].state=='DISCOVERED_UNVERIFIED'
    c.generate_transcription('asset','verbatim')
    assert c._registry().states['gemini-2.5-flash'].state=='ELIGIBLE'

def test_metadata_does_not_confirm_runtime(monkeypatch):
    c,_=sdk_client(monkeypatch,[])
    c.client.models.get=lambda **kw: object()
    c._preflight()
    state=c._registry().states['gemini-2.5-flash']
    assert state.state=='DISCOVERED_UNVERIFIED'
    assert state.reason=='MODEL_METADATA_ACCESS_CONFIRMED'
