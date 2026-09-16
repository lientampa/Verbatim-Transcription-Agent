import json
import re
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from src.provider_capability import CapabilityGate, CapabilityEvidence, RequestPath, tiny_audio
from src.gemini_client import GeminiClient
from src.runtime_models import RuntimeModels

SCHEMA = json.loads(Path('schemas/transcription_result.schema.json').read_text())


def sdk_success():
    sdk = Mock()
    def generate(**kw):
        marker = re.search(r'probe_[a-f0-9]+', kw['config'].system_instruction)[0]
        assert marker not in str(kw['contents'])
        return NS(text=json.dumps(dict(schema_version='1.0', job_id='PROBE', session_id='PROBE', block_id='PROBE',
            first_source_index=1,last_source_index=1,status='CONFIRMED',segments=[dict(source_index=1,text=marker,timestamp='00:00:00',speaker='Probe')])))
    sdk.models.generate_content.side_effect = generate
    return sdk


@pytest.mark.parametrize('field', ['audio_input','system_instruction','structured_output','schema_compatible','request_endpoint_usable'])
@pytest.mark.parametrize('value', [False, None])
def test_incomplete_contract_never_eligible(field, value):
    evidence = CapabilityEvidence(RequestPath('gemini-3.6-flash','generate_content','models.generate_content','test'), True, True, True, True, True)
    setattr(evidence, field, value)
    assert not evidence.eligible
    gate=CapabilityGate(sdk_success(),SCHEMA,'test')
    gate.results[evidence.path]=evidence
    client=GeminiClient.__new__(GeminiClient)
    client.capability_gate=gate
    client._runtime_models=RuntimeModels([evidence.path.model],[evidence.path.model])
    assert not client._eligible(evidence.path.model)


def test_discovery_is_not_eligibility_and_probe_does_not_spend_budgets(tmp_path):
    sdk = sdk_success()
    gate = CapabilityGate(sdk, SCHEMA, 'v1beta')
    c = GeminiClient.__new__(GeminiClient)
    c._runtime_models = RuntimeModels(['gemini-3.6-flash'], ['gemini-3.6-flash'])
    c.capability_gate = gate
    c._provider_attempt, c.size_retry_count, c.coverage_generation_count = 0, 0, 1
    checkpoint = tmp_path/'checkpoint.json';checkpoint.write_text('{"last_confirmed_audio_end":120}')
    before = checkpoint.read_bytes()
    assert c._eligible('gemini-3.6-flash')
    assert (c._provider_attempt,c.size_retry_count,c.coverage_generation_count)==(0,0,1)
    assert checkpoint.read_bytes()==before
    assert c._eligible('gemini-3.6-flash')
    assert sdk.models.generate_content.call_count==1
    assert len(tiny_audio())==32044


def test_native_local_path_and_lite_are_rejected_without_probe():
    sdk = sdk_success();gate=CapabilityGate(sdk,SCHEMA,'v1beta')
    assert gate.check('gemini-3.5-transcribe','transcribe').reason=='SYSTEM_INSTRUCTION_UNSUPPORTED'
    assert gate.check('gemini-3.6-FLASH-LITE','generate_content').reason=='LITE_MODEL_FORBIDDEN'
    sdk.models.generate_content.assert_not_called()


@pytest.mark.parametrize('code,reason',[(404,'ENDPOINT_MODEL_NOT_FOUND'),(429,'TRANSIENT_PROVIDER_FAILURE'),(503,'TRANSIENT_PROVIDER_FAILURE')])
def test_failed_probe_is_not_selected_or_cached_globally(code,reason):
    sdk=sdk_success();error=RuntimeError('provider');error.code=code
    sdk.models.generate_content.side_effect=error
    gate=CapabilityGate(sdk,SCHEMA,'v1beta')
    c=GeminiClient.__new__(GeminiClient)
    c._runtime_models=RuntimeModels(['gemini-3.6-flash'],['gemini-3.6-flash']);c.capability_gate=gate
    c._provider_attempt=0
    assert not c._eligible('gemini-3.6-flash')
    assert c._provider_attempt==0
    assert gate.check('gemini-3.6-flash','generate_content').reason==reason
    assert CapabilityGate(sdk_success(),SCHEMA,'v1beta').check('gemini-3.6-flash','generate_content').eligible


def test_quarantine_does_not_cross_path_or_version():
    gate=CapabilityGate(sdk_success(),SCHEMA,'v1beta')
    first=gate.check('gemini-3.6-flash','generate_content')
    other=RequestPath(first.path.model,'other','other.create','v1beta')
    gate.results[other]=CapabilityEvidence(other,True,True,True,True,True)
    gate.invalidate(first.path.model,'generate_content',404)
    assert not first.eligible
    assert gate.results[other].eligible
    assert CapabilityGate(sdk_success(),SCHEMA,'v1').check(first.path.model,'generate_content').eligible


def test_bad_schema_or_missing_system_nonce_cannot_pass():
    for text in ('{}',json.dumps(dict(schema_version='1.0',job_id='PROBE',session_id='PROBE',block_id='PROBE',first_source_index=1,last_source_index=1,status='CONFIRMED',segments=[]))):
        sdk=sdk_success();sdk.models.generate_content.side_effect=None;sdk.models.generate_content.return_value=NS(text=text)
        assert not CapabilityGate(sdk,SCHEMA,'v1beta').check('gemini-3.6-flash','generate_content').eligible


def test_production_client_filters_before_selection(monkeypatch):
    from google.genai.types import Model
    sdk=sdk_success()
    sdk._api_client._http_options=NS(base_url='https://example.invalid',api_version='v1beta')
    sdk.models.list.return_value=[Model(name=m,supported_actions=['generateContent']) for m in
        ('gemini-2.5-flash','gemini-3.5-transcribe','gemini-3.6-flash')]
    success=sdk.models.generate_content.side_effect
    def generate(**kw):
        if kw['model']=='gemini-2.5-flash':
            error=RuntimeError('not available');error.code=404;raise error
        return success(**kw)
    sdk.models.generate_content.side_effect=generate
    monkeypatch.setattr('src.gemini_client.genai.Client',lambda **kw:sdk)
    monkeypatch.setattr('src.gemini_client.TranscriptionModelAdapter.preflight',lambda sdk:(True,'test'))
    client=GeminiClient('test-key',prompt_controlled_audio=True)
    assert client.model_name=='gemini-3.6-flash'
    assert client.runtime_eligible_chain==('gemini-3.6-flash',)
    assert getattr(client,'_provider_attempt',0)==0
    assert client.request_schema==SCHEMA
    sdk.interactions.create.assert_not_called()


def test_files_api_404_does_not_quarantine_generation_endpoint():
    sdk=sdk_success();error=RuntimeError('file upload');error.code=404
    sdk.files.upload.side_effect=error
    evidence=CapabilityGate(sdk,SCHEMA,'v1beta').check('gemini-3.6-flash','generate_content')
    assert evidence.request_endpoint_usable is None
    assert evidence.reason=='CAPABILITY_PROBE_FAILED'
    sdk.models.generate_content.assert_not_called()
