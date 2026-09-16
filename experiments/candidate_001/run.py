"""Isolated production-path candidate; no reference reads during transcription."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
from uuid import uuid4

from dotenv import load_dotenv
from google.genai import types
from src.gemini_client import GeminiClient
from src.provider_adapters import capabilities
from experiments.candidate_001.observability import Diagnostics

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'VERBATIM_AUDIO_PROMPT_CANDIDATE_001'
INVALID = 'CANDIDATE_INVALID_PROMPT_NOT_APPLIED'


class ExperimentStop(BaseException):
    """Must escape provider fallback/structural retry Exception handlers."""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def delivery(model, config, expected, adapter_path):
    instruction = getattr(config, 'system_instruction', None)
    attached = adapter_path == 'generate_content' and instruction == expected
    return dict(actual_model=model, provider_path=adapter_path,
                system_instruction_attached=attached,
                prompt_version=expected.splitlines()[0],
                prompt_sha256=hashlib.sha256(expected.encode('utf-8')).hexdigest(),
                attached_instruction_sha256=hashlib.sha256(instruction.encode('utf-8')).hexdigest()
                if isinstance(instruction, str) else None)


class GuardedAdapter:
    def __init__(self, inner, path, expected, record, diagnostics=None, context=None):
        self.inner, self.path, self.expected, self.record = inner, path, expected, record
        self.diagnostics, self.context = diagnostics, context

    def generate(self, sdk, model, asset, prompt, config):
        evidence = delivery(model, config, self.expected, self.path)
        evidence['phase'] = 'OUTGOING_TRANSCRIPTION_REQUEST'
        self.record(evidence)
        if not evidence['system_instruction_attached'] or '-lite' in model:
            raise ExperimentStop(INVALID)
        call = lambda: self.inner.generate(sdk, model, asset, prompt, config)
        if self.diagnostics is None:
            return call()
        metadata = self.diagnostics.request_context(self.context() if self.context else {})
        return self.diagnostics.request(call, **metadata, model=model, adapter=self.path,
            api_method='models.generate_content' if self.path == 'generate_content' else 'interactions.create')


def prepare(root=ROOT):
    experiment = root / 'experiments/candidate_001'
    work = experiment / 'runtime' / uuid4().hex
    for folder in ('audio', 'prompts', 'schemas'):
        (work / folder).mkdir(parents=True, exist_ok=False)
    # Deliberately no fixture manifest/reference/annotation reads here.
    shutil.copyfile(root / 'tests/golden/golden_001/local/audio.m4a', work / 'audio/audio.m4a')
    shutil.copyfile(experiment / 'system_prompt.txt', work / 'prompts/system_prompt.txt')
    shutil.copyfile(root / 'schemas/transcription_result.schema.json', work / 'schemas/transcription_result.schema.json')
    return work


def compare(root, work):
    """Only called after successful complete production transcription."""
    from src.golden_benchmark.contracts import parse_transcript
    from src.golden_benchmark.alignment import AlignmentConfig
    from src.golden_benchmark.evaluator import evaluate
    fixture = root / 'tests/golden/golden_001'
    manifest = json.loads((fixture / 'manifest.json').read_text(encoding='utf-8-sig'))
    for key in ('human', 'machine_baseline'):
        if sha(fixture / manifest['paths'][key]) != manifest['input_sha256'][key]:
            raise ValueError('BENCHMARK_INPUT_CHANGED:' + key)
    if sha(work / 'audio/audio.m4a') != manifest['source']['fingerprint_sha256']:
        raise ValueError('BENCHMARK_AUDIO_CHANGED')
    duration = manifest['source']['duration_seconds']
    human, _ = parse_transcript((fixture / 'human.txt').read_text(encoding='utf-8-sig'), duration)
    baseline, _ = parse_transcript((fixture / 'machine_baseline.txt').read_text(encoding='utf-8-sig'), duration)
    candidate, anomalies = parse_transcript((work / 'output/transcript.txt').read_text(encoding='utf-8-sig'), duration)
    config = AlignmentConfig(tolerance_seconds=manifest['alignment']['tolerance_seconds'])
    result = evaluate(human, candidate, [], config, manifest.get('excluded_speaker_labels', []))
    # No unadjudicated zero counts/automatic decisions presented as semantic evidence.
    result.pop('offline_fidelity_decision')
    for key in ('critical_error_count', 'high_semantic_risk_count', 'annotated_example_counts'):
        result['metrics'].pop(key, None)
    annotations = json.loads((fixture / 'annotations.json').read_text(encoding='utf-8-sig'))['annotations']
    result['candidate_adjudication'] = [dict(annotation_id=a['annotation_id'], status='REQUIRES_CANDIDATE_ADJUDICATION') for a in annotations]
    def uncertainty(segments):
        return dict(total=sum(s.text.count('[không rõ]') for s in segments),
                    segments=sum('[không rõ]' in s.text for s in segments),
                    lexical_span_impact='UNAVAILABLE_REQUIRES_ACOUSTIC_ADJUDICATION')
    result['uncertainty_comparison'] = dict(baseline=uncertainty(baseline), candidate=uncertainty(candidate))
    result['uncertainty_comparison']['count_delta'] = uncertainty(candidate)['total'] - uncertainty(baseline)['total']
    result['candidate_anomalies'] = anomalies
    result['baseline_automatic'] = evaluate(human, baseline, [], config, manifest.get('excluded_speaker_labels', []))['metrics']
    result['BASELINE_PROMPT_PROVENANCE'] = 'UNKNOWN'
    result['model_confound'] = 'BASELINE_MODEL_PROVENANCE_UNKNOWN'
    result['recommendation'] = 'INCONCLUSIVE_MODEL_CONFOUND'
    write_json(work / 'comparison.json', result)
    return result


def run(live=False, root=ROOT):
    work = prepare(root)
    diagnostics = Diagnostics(work)
    with diagnostics.lifecycle():
        return _run(live, root, work, diagnostics)


def _run(live, root, work, diagnostics):
    from src import main as production
    load_dotenv(root / '.env', override=True)
    expected = (work / 'prompts/system_prompt.txt').read_text(encoding='utf-8').strip()
    if expected.splitlines()[0] != VERSION:
        raise ValueError('CANDIDATE_VERSION_MISMATCH')
    schema = json.loads((work / 'schemas/transcription_result.schema.json').read_text(encoding='utf-8'))
    events = []
    def record(event):
        events.append(event)
        diagnostics.safe_write(lambda: write_json(work / 'provenance.json', events))
        diagnostics.safe_write(lambda: print('[CANDIDATE_PREFLIGHT] ' + json.dumps(event, ensure_ascii=True), flush=True))
    class CandidateClient(GeminiClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            base = types.GenerateContentConfig(system_instruction=expected, temperature=0.0, response_mime_type='application/json')
            config = self.generation_config(self.model_name, base)
            event = delivery(self.model_name, config, expected, capabilities(self.model_name).provider_adapter)
            event.update(phase='SELECTED_RUNTIME_CONFIG_BEFORE_UPLOAD', benchmark='GOLDEN_001',
                         candidate='CANDIDATE_001', audio_sha256=sha(work / 'audio/audio.m4a'),
                         prompt_file_sha256=sha(work / 'prompts/system_prompt.txt'),
                         schema_sha256=sha(work / 'schemas/transcription_result.schema.json'),
                         schema_version=schema['properties']['schema_version']['enum'][0],
                         lite_allowed=self.allow_lite_models,
                         runtime_input_files=sorted(str(p.relative_to(work)) for p in work.rglob('*') if p.is_file()),
                         human_reference_exposed=any(p.name == 'human.txt' for p in work.rglob('*')),
                         baseline_exposed=any(p.name in ('machine_baseline.txt', 'golden_001_baseline.md') for p in work.rglob('*')),
                         annotations_exposed=any(p.name == 'annotations.json' for p in work.rglob('*')))
            record(event)
            if not event['system_instruction_attached'] or self.allow_lite_models:
                raise ExperimentStop(INVALID)
            if any(event[key] for key in ('human_reference_exposed', 'baseline_exposed', 'annotations_exposed')):
                raise ExperimentStop('CANDIDATE_REFERENCE_EXPOSURE')
            self.adapters = {name: GuardedAdapter(adapter, name, expected, record, diagnostics,
                lambda: getattr(self, 'last_response_metadata', {})) for name, adapter in self.adapters.items()}
            capability = self.capability_gate.check(self.model_name, capabilities(self.model_name).provider_adapter)
            checkpoint = json.loads((work/'state/checkpoint.json').read_text(encoding='utf-8'))
            rerun = dict(candidate='CANDIDATE_001', provider_capability='VERIFIED' if capability.eligible else 'UNKNOWN',
                provider_model=self.model_name, adapter=capability.path.adapter, api_method=capability.path.api_method,
                prompt_version=event['prompt_version'], system_instruction_attached=event['system_instruction_attached'],
                diagnostic_request_lifecycle=all(a.diagnostics is diagnostics for a in self.adapters.values()),
                diagnostic_exception_capture=diagnostics.state['final_state']=='RUNNING',
                diagnostic_run_finalization=diagnostics.write_failures==0,
                checkpoint_start=checkpoint['next_audio_start_us']/1_000_000,
                **{k:event[k] for k in ('human_reference_exposed','baseline_exposed','annotations_exposed','lite_allowed')})
            diagnostics.event('CANDIDATE_RERUN_PREFLIGHT', **rerun)
            print('[CANDIDATE_RERUN_PREFLIGHT] ' + json.dumps(rerun), flush=True)
            if diagnostics.write_failures:
                raise ExperimentStop('DIAGNOSTICS_UNAVAILABLE')
            if not live:
                raise ExperimentStop('PREFLIGHT_PASS_NO_TRANSCRIPTION')
    status = 'NOT_STARTED'
    with (work / 'runtime.log').open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log), patch.object(production, 'GeminiClient', CandidateClient):
            try:
                diagnostics.stage('TRANSCRIPTION')
                code = production.run_pipeline(base_dir=work)
                status = 'TRANSCRIPTION_COMPLETE' if code == 0 else 'PIPELINE_FAILED'
                if code:
                    diagnostics.state['engine_failure_reason_codes'] = diagnostics.checkpoint_reason()
                    diagnostics.conclude('RUN_FAILED', 'PIPELINE_FAILURE')
                else:
                    diagnostics.stage('TRANSCRIPTION_COMPLETED')
            except ExperimentStop as exc:
                status = str(exc)
                diagnostics.event('EXPERIMENT_STOP', **diagnostics.exception(exc, diagnostics.state['last_stage']))
                diagnostics.state['experiment_stop_reason'] = status
                diagnostics.conclude('RUN_COMPLETED' if status=='PREFLIGHT_PASS_NO_TRANSCRIPTION' else 'RUN_INTERRUPTED', 'EXPERIMENT_STOP')
    summary = dict(status=status, runtime=str(work), provenance=events,
                   BASELINE_PROMPT_PROVENANCE='UNKNOWN', model_confound='BASELINE_MODEL_PROVENANCE_UNKNOWN',
                   EXPERIMENT_INFRASTRUCTURE_CHANGE='ISOLATED_DELIVERY_GUARDS',
                   TRANSCRIPTION_PROMPT_CHANGE=VERSION, production_promoted=False)
    if status == 'TRANSCRIPTION_COMPLETE':
        diagnostics.stage('EVALUATION')
        summary['comparison'] = compare(root, work)
        diagnostics.conclude('RUN_COMPLETED', 'NORMAL_COMPLETION')
    diagnostics.stage('RESULT_PERSISTENCE')
    write_json(work / 'result.json', summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if status in ('PREFLIGHT_PASS_NO_TRANSCRIPTION', 'TRANSCRIPTION_COMPLETE') else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--live', action='store_true')
    args = parser.parse_args()
    sys.exit(run(live=args.live))
