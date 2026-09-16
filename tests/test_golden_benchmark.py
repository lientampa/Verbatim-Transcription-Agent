import json
import pytest
from src.golden_benchmark.contracts import parse_transcript,ErrorType,Severity
from src.golden_benchmark.alignment import align,speaker_metrics,AlignmentConfig
from src.golden_benchmark.evaluator import classify_pair,evaluate
from src.golden_evaluation.text_metrics import tokens
from src.fidelity_validator import ContentFidelityValidator
from src.response_parser import TranscriptionBlockResult,TranscriptSegment


def parsed(lines):return parse_transcript('\n'.join(lines),120)[0]

def test_repetition_exact_online_and_offline():
    text='Nên nên nên đầu tư.'
    assert classify_pair(text,text)==('EXACT_MATCH','INFO')
    b=TranscriptionBlockResult('1.0','j','s','BLOCK_001',1,1,'CONFIRMED',[TranscriptSegment(1,text,'00:00:01')])
    assert ContentFidelityValidator(source_mode='AUDIO').validate(b).allows_confirmation

@pytest.mark.parametrize('h,m,expected',[
    ('Nó có bọt rồi đó.','Nó có bột rồi đó.','WORD_SUBSTITUTION'),
    ('Khách họ cần thực tế hơn.','Khách họ cần sự thực tế hơn.','WORD_INSERTION'),
    ('Hình như phổ nhĩ quýt thơm.','Em uống phô mai rứa thôi.','PHRASE_SUBSTITUTION'),
    ('Giá 100 đồng','Giá 200 đồng','NUMBER_ERROR'),
    ('Dạ [không rõ]','Dạ biết rồi','UNCERTAINTY_ERROR')])
def test_conservative_types(h,m,expected):assert classify_pair(h,m)[0]==expected

def test_explicit_annotations_do_not_need_online_ground_truth():
    h=parsed(['[00:00:01 - Người nói 1]: Dạ em ở Quảng Nam.'])
    m=parsed(['[00:00:02 - Anh]: Dạ em Đà Nẵng luôn.'])
    a=dict(annotation_id='critical',human_text=h[0].text,machine_text=m[0].text,primary_error_type='PHRASE_SUBSTITUTION',severity='CRITICAL',semantic_risk='HIGH')
    result=evaluate(h,m,[a])
    assert result['offline_fidelity_decision']=='FAIL_CRITICAL_REFERENCE_ERROR'
    assert result['metrics']['critical_error_count']==1 and result['metrics']['high_semantic_risk_count']==1
    assert a=={k:v for k,v in result['reviewed_examples'][0].items() if k not in ('human_segments','machine_segments')}

@pytest.mark.parametrize('h,m,kind',[
    ('qua coi thực tế','qua xem thực tế','OVER_NORMALIZATION'),
    ('Thiềm Thừ','thiểm thử','ENTITY_ERROR')])
def test_reference_specific_labels(h,m,kind):
    hs=parsed([f'[00:00:01 - A]: {h}']);ms=parsed([f'[00:00:01 - B]: {m}'])
    annotation=dict(annotation_id='e',human_text=h,machine_text=m,primary_error_type=kind,severity='MAJOR',semantic_risk='HIGH')
    assert evaluate(hs,ms,[annotation])['metrics']['annotated_example_counts']=={kind:1}

def test_unknown_is_one_unit():assert tokens('Dạ [không rõ] ờ',True)==['dạ','[không rõ]','ờ']

def test_speaker_permutation_and_fragmentation():
    h=parsed(['[00:00:01 - Người nói 1]: Chào.','[00:00:02 - Người nói 2]: Vâng.','[00:00:03 - Người nói 1]: Cảm ơn.'])
    m=parsed(['[00:00:02 - Chị]: Chào.','[00:00:03 - Anh]: Vâng.','[00:00:04 - Chị]: Cảm ơn.'])
    groups=align(h,m)
    assert speaker_metrics(h,m,groups)['errors']==0
    m=parsed(['[00:00:02 - Chị]: Chào.','[00:00:03 - Anh]: Vâng.','[00:00:04 - Khách]: Cảm ơn.'])
    assert speaker_metrics(h,m,align(h,m))['fragmentation_candidates']==1

def test_human_anomalies_do_not_destroy_lexical_alignment():
    raw='TITLE\n[00:00:10 - A]: chào bạn [00:00:09 - A] cảm ơn\n[00:00: 11 - A]: vâng'
    h,anomalies=parse_transcript(raw,120)
    assert len(h)==3 and len([a for a in anomalies if a['reason']=='TIMESTAMP_REFERENCE_ANOMALY'])==2
    m=parsed(['[00:00:10 - B]: chào bạn','[00:00:11 - B]: cảm ơn','[00:00:12 - B]: vâng'])
    assert all(g['resolved'] for g in align(h,m))
    assert raw.startswith('TITLE')

def test_split_and_merge_alignment():
    h=parsed(['[00:00:01 - A]: một hai ba bốn'])
    m=parsed(['[00:00:02 - B]: một hai','[00:00:03 - B]: ba bốn'])
    assert align(h,m)==[dict(human=[0],machine=[0,1],resolved=True)]
    assert align(m,h)==[dict(human=[0,1],machine=[0],resolved=True)]

def test_absent_annotation_rejected():
    h=parsed(['[00:00:01 - A]: chào']);m=parsed(['[00:00:01 - B]: chào'])
    with pytest.raises(ValueError,match='EVIDENCE_MISSING'):
        evaluate(h,m,[dict(annotation_id='missing',human_text='absent',machine_text='chào',primary_error_type='ENTITY_ERROR',severity='MAJOR',semantic_risk='HIGH')])

def test_invalid_timestamp_and_config_rejected():
    with pytest.raises(ValueError):parsed(['[00:60:01 - A]: chào'])
    with pytest.raises(ValueError):AlignmentConfig(tolerance_seconds=float('nan'))

def test_manifest_input_hash_guard(tmp_path):
    from src.golden_benchmark.__main__ import run
    root=tmp_path/'fixture';root.mkdir();(root/'human.txt').write_text('modified')
    (root/'manifest.json').write_text(json.dumps(dict(paths={'human':'human.txt','machine_baseline':'machine.txt','audio':'audio'},input_sha256={'human':'incorrect'})))
    with pytest.raises(ValueError,match='INPUT_CHANGED'):run(root,tmp_path/'reports')


def test_speaker_whitespace_is_metadata_only():
    h=parsed(['[00:00:01 - A]: chào','[00:00:02 - A ]: vâng'])
    assert h[0].speaker_id==h[1].speaker_id
    assert h[1].speaker_display_name=='A '


def test_low_similarity_segments_can_remain_unmatched():
    h=parsed(['[00:00:01 - A]: một hai ba'])
    m=parsed(['[00:01:50 - B]: khác hoàn toàn'])
    assert all(not g['resolved'] for g in align(h,m))
