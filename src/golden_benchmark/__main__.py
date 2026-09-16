"""Explicit offline CLI. Reads immutable inputs; no Gemini or production state access."""
import argparse
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
from .contracts import parse_transcript
from .alignment import AlignmentConfig
from .evaluator import evaluate


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output==root or root in output.parents:raise ValueError('BENCHMARK_OUTPUT_MUST_BE_SEPARATE')
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
    paths={k:(root/v).resolve() for k,v in manifest['paths'].items()}
    for k in ('human','machine_baseline'):
        if digest(paths[k])!=manifest['input_sha256'][k]:raise ValueError('BENCHMARK_INPUT_CHANGED:'+k)
    if digest(paths['audio'])!=manifest['source']['fingerprint_sha256']:raise ValueError('BENCHMARK_AUDIO_CHANGED')
    duration=manifest['source']['duration_seconds']
    human,ha=parse_transcript(paths['human'].read_text(encoding='utf-8-sig'),duration)
    machine,ma=parse_transcript(paths['machine_baseline'].read_text(encoding='utf-8-sig'),duration)
    print(f'[BENCHMARK_LOAD] id={manifest["benchmark_id"]} human={len(human)} machine={len(machine)}')
    annotations=json.loads(paths['annotations'].read_text(encoding='utf-8-sig'))['annotations']
    result=evaluate(human,machine,annotations,AlignmentConfig(tolerance_seconds=manifest['alignment']['tolerance_seconds']),manifest.get('excluded_speaker_labels',[]))
    result.update(benchmark_id=manifest['benchmark_id'],source=manifest['source'],input_sha256=manifest['input_sha256'],
                  human_segments=[asdict(s) for s in human],machine_segments=[asdict(s) for s in machine],
                  human_anomalies=ha,machine_anomalies=ma)
    print(f'[BENCHMARK_ALIGN] groups={len(result["alignment"])}')
    # Offline audit deliberately demonstrates limits of reference-free online checks.
    from src.fidelity_validator import ContentFidelityValidator
    from src.response_parser import TranscriptionBlockResult,TranscriptSegment
    validator=ContentFidelityValidator(source_mode='AUDIO');audit=[]
    for a in result['reviewed_examples']:
        block=TranscriptionBlockResult('1.0','BENCHMARK','OFFLINE','BLOCK_001',1,1,'CONFIRMED',[TranscriptSegment(1,a['machine_text'],'00:00:00')])
        online=validator.validate(block)
        audit.append(dict(annotation_id=a['annotation_id'],online_without_reference=online.decision.value,
                          issues=[i.indicator for i in online.issues],offline_primary_error=a['primary_error_type']))
        print(f'[BENCHMARK_ERROR] id={a["annotation_id"]} type={a["primary_error_type"]} severity={a["severity"]}')
    result['fidelity_audit']=audit
    output.mkdir(parents=True,exist_ok=True)
    (output/'golden_001_baseline.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    metrics=result['metrics'];wer=metrics['lexical']['relaxed_wer']
    lines=['# GOLDEN_001 baseline evaluation','',f'Source duration: {duration} seconds. Human lexical authority: PRIMARY; timestamp authority: PARTIAL.',
           f'Human segments: {len(human)}; machine segments: {len(machine)}.',
           f'Aligned groups: {metrics["aligned_groups"]}; unmatched human: {metrics["unmatched_human_segments"]}; unmatched machine: {metrics["unmatched_machine_segments"]}.','',
           '## Lexical metrics','',f'Relaxed syllable/whitespace WER: {wer["error_rate"]["value"]:.4%}. S={wer["substitutions"]}, D={wer["deletions"]}, I={wer["insertions"]}, N={wer["reference_count"]}.',
           f'Strict WER: {metrics["lexical"]["strict_wer"]["error_rate"]["value"]:.4%}.',
           result['normalization'],'',result['counting_policy'],'',
           '## Critical and verbatim examples','',f'Offline fidelity decision: {result["offline_fidelity_decision"]}.','',f'Explicit annotated CRITICAL errors: {metrics["critical_error_count"]}; HIGH semantic risk: {metrics["high_semantic_risk_count"]}. These are not exhaustive corpus totals.','',
           '| Example | Human | Machine | Primary label | Severity |','|---|---|---|---|---|']
    lines += [f'| {a["annotation_id"]} | {a["human_text"]} | {a["machine_text"]} | {a["primary_error_type"]} | {a["severity"]} |' for a in result['reviewed_examples']]
    lines += ['', '## Speaker, uncertainty and timestamp observations','',
              f'Speaker metrics: `{json.dumps(metrics["speaker"],ensure_ascii=False)}`.',
              f'Uncertainty units: human={metrics["uncertainty"]["human"]}; machine={metrics["uncertainty"]["machine"]}. No accuracy reward for unknown markers.',
              f'Human import anomalies: `{json.dumps(ha,ensure_ascii=False)}`.',
              'Timestamp disagreements do not establish machine error. Acoustic coverage gaps are UNAVAILABLE without audio adjudication.',
              '', '## Validator implications','',
              'Structural PASS and coverage PASS do not establish fidelity. Reference-free heuristic results below cannot detect arbitrary factual or entity substitutions.',
              f'`{json.dumps(audit,ensure_ascii=False)}`','',
              'Human-vs-machine lexical/semantic comparison is OFFLINE_ONLY. Repetition/normalization heuristics are ONLINE_HEURISTIC. Schema and physical identity checks are ONLINE_SAFE.',
              'Do not hard-code benchmark entities into production. Existing native Transcribe uses provider verbatim configuration and does not receive the Flash system instruction.',
              '', '## Recommended changes and limitations','',
              'Keep production acceptance/retry/checkpoint/coverage policy unchanged: these examples do not justify guessing semantic correctness from machine text alone.',
              'The Flash prompt already forbids contextual invention. Its absolute 100% certainty wording should be calibrated using acoustic review before changing it; excessive uncertainty is not automatically correct.',
              'Optional future context_terms should be user-supplied hints only, never forced replacement, and capability-gated (native vocabulary cannot be combined with current diarization/word timestamps).',
              'Review unresolved alignments, speaker fragmentation candidates and human timing anomalies against audio. No live inference or listening adjudication performed.',
              'Detailed alignment, normalized metrics, classifications, fingerprints and anomalies are in golden_001_baseline.json.']
    (output/'golden_001_baseline.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'[BENCHMARK_METRICS] WER={wer["error_rate"]["value"]:.6f} critical={metrics["critical_error_count"]}')
    print('[BENCHMARK_COMPLETE] offline_only=true')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-dir',type=Path,default=Path('tests/golden/golden_001'))
    parser.add_argument('--output-dir',type=Path,default=Path('reports'))
    args=parser.parse_args();run(args.fixture_dir,args.output_dir)
