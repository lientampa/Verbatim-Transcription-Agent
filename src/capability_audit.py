"""Explicit tiny-audio capability audit. Never runs GOLDEN_001 or reads checkpoint."""
import base64
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from src.model_policy import effective_model_chain
from src.provider_adapters import capabilities, TranscriptionModelAdapter
from src.provider_capability import CapabilityGate, tiny_audio


def run():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env', override=True)
    sdk = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    models = list(sdk.models.list())
    chain = effective_model_chain(models, TranscriptionModelAdapter.preflight(sdk)[0])
    schema = json.loads((root / 'schemas/transcription_result.schema.json').read_text(encoding='utf-8'))
    options = sdk._api_client._http_options
    gate = CapabilityGate(sdk, schema, str(options.base_url) + '/' + str(options.api_version))
    result = dict(discovered=[dict(model=m.name, supported_actions=m.supported_actions) for m in models],
                  paths=[], native_provider_probe=None)
    output = root / 'reports/provider_capability_audit.json'
    def save():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for model in chain:
        print('[MODEL_DISCOVERED] model=' + model, flush=True)
        result['paths'].append(gate.check(model, capabilities(model).provider_adapter).report())
        save()
    # Separately test the generic SDK parameter on the discovered native model.
    # No adapter modification or fabricated user-message system instruction.
    if 'gemini-3.5-transcribe' in chain:
        try:
            response = sdk.interactions.create(model='gemini-3.5-transcribe',
                timeout=30,
                input=[dict(type='audio', data=base64.b64encode(tiny_audio()).decode(), mime_type='audio/wav')],
                system_instruction='CAPABILITY_NATIVE_SYSTEM_PROBE_V1\nThis is synthetic audio. Preserve verbatim transcription.',
                generation_config={'transcription_config':{'mode':{'type':'verbatim','diarization_mode':'speaker','timestamp_granularities':['word']}}})
            result['native_provider_probe'] = dict(request_accepted=True, status=str(getattr(response,'status',None)),
                instruction_semantically_verified=False, schema_compatible=None)
        except Exception as exc:
            result['native_provider_probe'] = dict(request_accepted=False,
                error_code=getattr(exc,'code',getattr(exc,'status_code',None)),
                error_type=type(exc).__name__, detail=str(exc)[:1500])
        save()
    result['readiness'] = 'CANDIDATE_001_READY' if any(r['eligible'] for r in result['paths']) else 'CANDIDATE_001_BLOCKED_PROVIDER_CAPABILITY_UNKNOWN'
    save()
    print(json.dumps(dict(paths=result['paths'],native_provider_probe=result['native_provider_probe'],readiness=result['readiness']),indent=2))


if __name__ == '__main__':
    run()
