from copy import deepcopy
from dataclasses import replace
import json

import pytest

from src.output_renderer import OutputRenderer
from src.response_parser import TranscriptSegment
from src.gemini_client import GeminiClient
from tests.test_stage4_audio_retry import run_audio_case


def test_absolute_display_and_verbatim_are_preserved():
    segment = TranscriptSegment(5, "Dạ, em... em nghĩ là, ờ, chưa, chưa xong.", "20:03", "Chí")
    before = deepcopy(segment)
    assert OutputRenderer(strict_speaker_format=True).format_txt_line(segment) == (
        "[00:20:03 - Chí]: Dạ, em... em nghĩ là, ờ, chưa, chưa xong.")
    assert segment == before


@pytest.mark.parametrize("label", [None, "", "SPEAKER_01", "speaker_a", "BLOCK_003", "UNKNOWN_1", "segment_12"])
def test_no_fabricated_name_or_technical_label(label):
    with pytest.raises(ValueError, match="SPEAKER_LABEL_REQUIRED"):
        OutputRenderer(strict_speaker_format=True).format_txt_line(TranscriptSegment(1,"Vâng.","01:00",label))


def test_no_fabricated_timestamp():
    with pytest.raises(ValueError, match="ABSOLUTE_TIMESTAMP_REQUIRED"):
        OutputRenderer(strict_speaker_format=True).format_txt_line(TranscriptSegment(1,"Vâng.",None,"Người nói 1"))


@pytest.mark.parametrize("missing", [False, True])
def test_pipeline_context_and_confirmation_gate(tmp_path, monkeypatch, missing):
    prompts=[]
    def setup(config):
        monkeypatch.setattr("src.main.load_config", lambda **kwargs: replace(config,strict_speaker_format=True))
        original=GeminiClient.generate_transcription
        def generate(self, user_prompt, **kwargs):
            prompts.append(user_prompt)
            data=json.loads(original(self,user_prompt=user_prompt,**kwargs))
            data["segments"][0]["speaker"]=None if missing else "Người nói 1"
            return json.dumps(data)
        monkeypatch.setattr(GeminiClient,"generate_transcription",generate)
    result,_,_,_,checkpoint=run_audio_case(tmp_path,monkeypatch,setup=setup,fail_count=0)
    if missing:
        assert result==1 and checkpoint["next_audio_start_us"]==0
        assert not checkpoint["confirmed_segments"]
    else:
        assert result==0
        assert "['Người nói 1']" in prompts[1]
        text=(tmp_path/'output'/'transcript.txt').read_text(encoding='utf-8')
        assert text.startswith('[00:00:00 - Người nói 1]: Vâng.')
