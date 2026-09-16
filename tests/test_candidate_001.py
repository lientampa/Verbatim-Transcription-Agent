from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai import types
from experiments.candidate_001.run import GuardedAdapter, ExperimentStop, INVALID, VERSION, prepare
from src.gemini_client import GeminiClient
from src.provider_adapters import GeminiGenerateContentAdapter, TranscriptionModelAdapter
from src.transcriber import GeminiTranscriber


def test_real_flash_request_receives_candidate_without_reference(tmp_path, monkeypatch):
    # Actual preparation and request construction are exercised while reference reads are forbidden.
    root = tmp_path
    for folder in ('tests/golden/golden_001/local', 'experiments/candidate_001', 'schemas'):
        (root / folder).mkdir(parents=True)
    (root / 'tests/golden/golden_001/local/audio.m4a').write_bytes(b'audio')
    prompt = Path('experiments/candidate_001/system_prompt.txt').read_text(encoding='utf-8').strip()
    (root / 'experiments/candidate_001/system_prompt.txt').write_text(prompt, encoding='utf-8')
    (root / 'schemas/transcription_result.schema.json').write_text('{}', encoding='utf-8')
    forbidden = {'human.txt', 'machine_baseline.txt', 'annotations.json', 'golden_001_baseline.md'}
    original = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path.name not in forbidden
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    work = prepare(root)
    assert not any(p.name in forbidden for p in work.rglob('*'))
    transcriber = GeminiTranscriber(Mock(), work / 'prompts/system_prompt.txt')
    user = transcriber.build_user_prompt('job', 'session', 'BLOCK_001', 1, start_offset_seconds=0, end_offset_seconds=120)
    client = GeminiClient.__new__(GeminiClient)
    config = client.generation_config('gemini-3.6-flash', types.GenerateContentConfig(system_instruction=transcriber.load_system_prompt()))
    sdk = Mock()
    events = []
    adapter = GuardedAdapter(GeminiGenerateContentAdapter(), 'generate_content', prompt, events.append)
    adapter.generate(sdk, 'gemini-3.6-flash', 'audio_asset', user, config)
    outgoing = sdk.models.generate_content.call_args.kwargs
    assert outgoing['config'].system_instruction == prompt
    assert outgoing['contents'] == ['audio_asset', user]
    assert events[0]['system_instruction_attached'] is True
    assert events[0]['prompt_version'] == VERSION
    assert all(name not in str(outgoing) for name in forbidden)


@pytest.mark.parametrize('path,config,model', [
    ('transcribe', None, 'gemini-3.5-transcribe'),
    ('generate_content', types.GenerateContentConfig(system_instruction='wrong'), 'gemini-3.6-flash'),
    ('generate_content', types.GenerateContentConfig(system_instruction=VERSION), 'gemini-3.6-flash-lite'),
])
def test_invalid_path_aborts_before_provider_without_fallback(path, config, model):
    inner = Mock()
    events = []
    adapter = GuardedAdapter(inner, path, VERSION, events.append)
    with pytest.raises(ExperimentStop, match=INVALID):
        adapter.generate(Mock(), model, 'asset', 'request', config)
    inner.generate.assert_not_called()
    assert not issubclass(ExperimentStop, Exception)


def test_valid_then_native_fallback_is_stopped():
    sdk = Mock()
    events = []
    flash = GuardedAdapter(GeminiGenerateContentAdapter(), 'generate_content', VERSION, events.append)
    flash.generate(sdk, 'gemini-2.5-flash', 'asset', 'request', types.GenerateContentConfig(system_instruction=VERSION))
    native = GuardedAdapter(TranscriptionModelAdapter(), 'transcribe', VERSION, events.append)
    with pytest.raises(ExperimentStop, match=INVALID):
        native.generate(sdk, 'gemini-3.5-transcribe', SimpleNamespace(uri='files/test', mime_type='audio/mp4'), 'request', None)
    sdk.interactions.create.assert_not_called()
    assert [e['system_instruction_attached'] for e in events] == [True, False]
