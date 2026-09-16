"""Offline comparison and explicit evidence annotations, with no production imports for evaluation."""
from collections import Counter
from dataclasses import asdict
import re
from src.golden_evaluation.text_metrics import tokens,normalize,edit_counts
from .contracts import ErrorType,Severity,SemanticRisk
from .alignment import align,speaker_metrics,AlignmentConfig


def classify_pair(human,machine):
    """Conservative lexical categories; never infer hallucination or factual truth."""
    r,t=tokens(human,True),tokens(machine,True)
    if human==machine:return ErrorType.EXACT_MATCH.value,Severity.INFO.value
    if r==t:return ErrorType.MINOR_ORTHOGRAPHIC.value,Severity.MINOR.value
    if ('[không rõ]' in r)!=('[không rõ]' in t):return ErrorType.UNCERTAINTY_ERROR.value,Severity.MAJOR.value
    if re.findall(r'\d+',human)!=re.findall(r'\d+',machine):return ErrorType.NUMBER_ERROR.value,Severity.MAJOR.value
    c=edit_counts(r,t)
    if not c['substitutions'] and not c['insertions']:kind='OMISSION';count=c['deletions']
    elif not c['substitutions'] and not c['deletions']:kind='INSERTION';count=c['insertions']
    else:kind='SUBSTITUTION';count=sum(c[k] for k in ('substitutions','insertions','deletions'))
    return ('WORD_' if count==1 else 'PHRASE_')+kind, 'MINOR' if count==1 else 'MAJOR'


def evaluate(human,machine,annotations,config=AlignmentConfig(),excluded_speaker_labels=()):
    groups=align(human,machine,config);issues=[]
    for number,g in enumerate(groups):
        h=' '.join(human[i].text for i in g['human']);m=' '.join(machine[i].text for i in g['machine'])
        kind,severity=classify_pair(h,m)
        issues.append(dict(group=number,primary_error_type=kind,severity=severity,semantic_risk=None,
                           evidence='OFFLINE_ONLY_LEXICAL_COMPARISON',human_text=h,machine_text=m))
    checked=[]
    for item in annotations:
        ErrorType(item['primary_error_type']);Severity(item['severity']);SemanticRisk(item['semantic_risk'])
        hi=[s.index for s in human if item['human_text'] in s.text]
        mi=[s.index for s in machine if item['machine_text'] in s.text]
        if not hi or not mi:raise ValueError('BENCHMARK_ANNOTATION_EVIDENCE_MISSING:'+item['annotation_id'])
        checked.append(dict(item,human_segments=hi,machine_segments=mi))
    htext=' '.join(s.text for s in human);mtext=' '.join(s.text for s in machine)
    lexical={name:edit_counts(tokens(htext,relaxed),tokens(mtext,relaxed)) for name,relaxed in [('strict_wer',False),('relaxed_wer',True)]}
    deltas=[]
    for g in groups:
        if g['resolved'] and all(human[i].timestamp_reliable for i in g['human']) and all(machine[i].timestamp_reliable for i in g['machine']):
            deltas.append(abs(human[g['human'][0]].seconds-machine[g['machine'][0]].seconds))
    return dict(offline_fidelity_decision='FAIL_CRITICAL_REFERENCE_ERROR' if any(i['severity']=='CRITICAL' for i in checked) else 'REVIEW_REQUIRED' if any(i['primary_error_type'] not in ('EXACT_MATCH','MINOR_ORTHOGRAPHIC') for i in issues) else 'NO_ERROR_DETECTED',
        alignment=groups,automatic_group_classifications=issues,reviewed_examples=checked,
        metrics=dict(lexical=lexical,aligned_groups=sum(g['resolved'] for g in groups),
                     unmatched_human_segments=sum(len(g['human']) for g in groups if not g['resolved']),
                     unmatched_machine_segments=sum(len(g['machine']) for g in groups if not g['resolved']),
                     automatic_group_counts=dict(Counter(i['primary_error_type'] for i in issues)),
                     annotated_example_counts=dict(Counter(i['primary_error_type'] for i in checked)),
                     critical_error_count=sum(i['severity']=='CRITICAL' for i in checked),
                     high_semantic_risk_count=sum(i['semantic_risk']=='HIGH' for i in checked),
                     uncertainty=dict(human=tokens(htext,True).count('[không rõ]'),machine=tokens(mtext,True).count('[không rõ]'),
                                      adjudication='Counts are not rewards; confident replacement versus unnecessary uncertainty requires audio review.'),
                     speaker=speaker_metrics(human,machine,groups,excluded_speaker_labels),
                     timestamp=dict(eligible_groups=len(deltas),absolute_deltas_seconds=deltas,
                                    limitation='Human timing is PARTIAL; deltas are disagreements, not adjudicated machine errors.'),
                     coverage=dict(acoustic_coverage=None,reason='Not inferred from text or unadjudicated reference timestamps.')),
        normalization='NFC; strict whitespace tokens; relaxed casefold/punctuation. Fillers/repetitions/entities/numbers retained; [không rõ] is one token.',
        counting_policy='WER edits, automatic group classifications and evidence-annotated examples are separate views and must not be summed. One primary label per example; no composite score.',
        alignment_config=asdict(config))
