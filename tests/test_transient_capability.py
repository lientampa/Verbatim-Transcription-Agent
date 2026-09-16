from collections import Counter
from types import SimpleNamespace as NS
import pytest
import httpx
from google.genai.types import Model
from src.provider_capability import CapabilityGate
from src.gemini_client import GeminiClient, GeminiTranscribeError
from src.model_policy import PREFERRED_MODEL_CHAIN
from tests.test_provider_capability import sdk_success, SCHEMA
from tests.test_stage43_adaptation import ProviderError


@pytest.mark.parametrize('error',[ProviderError(503),ProviderError(504),ProviderError(429),TimeoutError(),httpx.ReadTimeout('timeout')])
def test_transient_is_bounded_and_fail_closed(error):
    sdk=sdk_success();sdk.models.generate_content.side_effect=error
    gate=CapabilityGate(sdk,SCHEMA,'test');model='gemini-3.6-flash'
    assert gate.check(model,'generate_content').state=='UNVERIFIED_TRANSIENT'
    for _ in range(5):
        evidence=gate.check(model,'generate_content',reprobe=True)
        assert evidence.state=='UNVERIFIED_TRANSIENT' and not evidence.eligible
    assert sdk.models.generate_content.call_count==2


def test_deterministic_results_are_not_reprobed():
    sdk=sdk_success();sdk.models.generate_content.side_effect=ProviderError(404)
    gate=CapabilityGate(sdk,SCHEMA,'test')
    for _ in range(3):
        assert gate.check('gemini-2.5-flash','generate_content',reprobe=True).state=='VERIFIED_INELIGIBLE'
        assert gate.check('gemini-3.5-transcribe','transcribe',reprobe=True).state=='VERIFIED_INELIGIBLE'
        assert gate.check('gemini-3.6-FLASH-LITE','generate_content',reprobe=True).state=='VERIFIED_INELIGIBLE'
    assert sdk.models.generate_content.call_count==1


@pytest.mark.parametrize('recovery',['gemini-3.5-flash','gemini-3.8-flash',None])
def test_real_router_quota_then_tiny_reprobe(monkeypatch,tmp_path,capsys,recovery):
    sdk=sdk_success();success=sdk.models.generate_content.side_effect
    sdk._api_client._http_options=NS(base_url='test',api_version='v1beta')
    sdk.models.list.return_value=[Model(name=m,supported_actions=['generateContent']) for m in PREFERRED_MODEL_CHAIN]
    probes=Counter();transcriptions=[];asset=object()
    def generate(**kw):
        model=kw['model']
        if kw['config'].system_instruction.startswith('AUDIO_REQUEST_CONTRACT_PROBE_V1'):
            probes[model]+=1
            assert kw['contents'][0] is not asset
            if model=='gemini-2.5-flash':raise ProviderError(404)
            if model in ('gemini-3.5-flash','gemini-3.8-flash') and (probes[model]==1 or model!=recovery):
                raise ProviderError(503)
            return success(**kw)
        assert kw['contents'][0] is asset
        transcriptions.append(model)
        if model=='gemini-3.6-flash':raise ProviderError(429,'37s')
        return NS(text='{}',candidates=None,usage_metadata=None)
    sdk.models.generate_content.side_effect=generate
    monkeypatch.setattr('src.gemini_client.genai.Client',lambda **kw:sdk)
    monkeypatch.setattr('src.gemini_client.TranscriptionModelAdapter.preflight',lambda sdk:(True,'TEST'))
    sleep=[];monkeypatch.setattr('src.gemini_client.time.sleep',sleep.append)
    client=GeminiClient('test',prompt_controlled_audio=True)
    cp=tmp_path/'checkpoint.json';cp.write_text('{"next_audio_start_us":0}');before=cp.read_bytes()
    client.size_retry_count=0;client.coverage_generation_count=1;client.structural_retry_count=0;client.fidelity_retry_count=0
    assert client.model_name=='gemini-3.6-flash'
    assert all(v==1 for v in probes.values())
    for _ in range(3):assert client.runtime_eligible_chain==('gemini-3.6-flash',)
    assert all(v==1 for v in probes.values())  # Merely inspecting state never re-probes.
    if recovery:
        assert client.generate_transcription(asset,'verbatim','- block_id: "BLOCK_001"')=='{}'
        assert transcriptions==['gemini-3.6-flash',recovery]
        assert client.capability_gate.check(recovery,'generate_content').state=='VERIFIED_ELIGIBLE'
    else:
        with pytest.raises(GeminiTranscribeError):client.generate_transcription(asset,'verbatim','- block_id: "BLOCK_001"')
        assert transcriptions==['gemini-3.6-flash']
    assert client._provider_attempt==len(transcriptions)
    assert (client.size_retry_count,client.coverage_generation_count,client.structural_retry_count,client.fidelity_retry_count)==(0,1,0,0)
    assert cp.read_bytes()==before and sleep==[]
    assert probes['gemini-2.5-flash']==1
    assert probes['gemini-3.5-flash']==2
    assert probes['gemini-3.8-flash']==(1 if recovery=='gemini-3.5-flash' else 2)
    assert client._registry().states['gemini-3.6-flash'].state=='QUOTA_LIMITED'
    assert client.capability_gate.check('gemini-3.6-flash','generate_content').state=='VERIFIED_ELIGIBLE'
    sdk.interactions.create.assert_not_called()
    output=capsys.readouterr().out
    assert 'configured_primary=gemini-2.5-flash effective_primary=gemini-3.6-flash' in output
    assert 'PROVIDER_DELAY_EXCEEDS_LIMIT' in output
    assert 'eligibility_reason=CAPABILITY_ELIGIBLE routing_reason=' in output
