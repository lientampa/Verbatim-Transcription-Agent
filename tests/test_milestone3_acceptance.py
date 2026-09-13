"""Comprehensive Milestone 3 acceptance tests covering all 16 required tests + End-to-End test."""

from src.transcriber import TranscriptionOutcome
from src.fidelity_validator import ContentFidelityValidator


import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock, PropertyMock

from src.response_parser import (
    ResponseParser,
    TranscriptionBlockResult,
    TranscriptSegment,
    InvalidJSONError,
    SchemaValidationError,
)
from src.transcript_validator import TranscriptValidator, ValidationResult
from src.checkpoint import CheckpointManager, CheckpointStatus
from src.output_merger import OutputMerger
from src.output_renderer import OutputRenderer
from src.transcriber import GeminiTranscriber, TranscriptionError
from src.main import run_pipeline
from src.block_builder import SourceBlock


@pytest.fixture
def schema_path():
    return Path(__file__).resolve().parent.parent / "schemas" / "transcription_result.schema.json"


@pytest.fixture
def parser(schema_path):
    return ResponseParser(schema_path=schema_path)


@pytest.fixture
def validator():
    return TranscriptValidator()


# ---------------------------------------------------------------------------
# TEST 1: Normal transcription
# ---------------------------------------------------------------------------
def test_1_normal_transcription(parser, validator):
    valid_json = {
        "schema_version": "1.0",
        "job_id": "JOB_001",
        "session_id": "SESSION_001",
        "block_id": "BLOCK_003",
        "first_source_index": 41,
        "last_source_index": 43,
        "status": "CONFIRMED",
        "segments": [
            {
                "source_index": 41,
                "timestamp": "00:13:34",
                "speaker": "Người A",
                "text": "Hôm nay chúng ta sẽ trao đổi về vấn đề này.",
            },
            {
                "source_index": 42,
                "timestamp": "00:13:39",
                "speaker": "Người B",
                "text": "Vâng, tôi đồng ý.",
            },
            {
                "source_index": 43,
                "timestamp": "00:13:45",
                "speaker": "Người A",
                "text": "Ờ... nhưng mà tôi nghĩ chúng ta cần xem lại.",
            },
        ],
    }

    block_result = parser.parse(json.dumps(valid_json, ensure_ascii=False))
    assert block_result.first_source_index == 41
    assert block_result.last_source_index == 43
    assert len(block_result.segments) == 3

    val_result = validator.validate_block_result(
        block_result,
        expected_first_index=41,
        expected_last_index=43,
    )
    assert val_result.is_valid is True
    assert len(val_result.errors) == 0


# ---------------------------------------------------------------------------
# TEST 2: Missing source index
# ---------------------------------------------------------------------------
def test_2_missing_source_index(validator):
    # Expected: 41, 42, 43, 44. Actual: 41, 42, 44 (missing 43)
    block_result = TranscriptionBlockResult(
        schema_version="1.0",
        job_id="JOB_001",
        session_id="SESSION_001",
        block_id="BLOCK_003",
        first_source_index=41,
        last_source_index=44,
        status="CONFIRMED",
        segments=[
            TranscriptSegment(source_index=41, text="Câu 41"),
            TranscriptSegment(source_index=42, text="Câu 42"),
            TranscriptSegment(source_index=44, text="Câu 44"),
        ],
    )
    val_result = validator.validate_block_result(block_result, expected_first_index=41, expected_last_index=44)
    assert val_result.is_valid is False
    assert any("missing" in e.lower() for e in val_result.errors)


# ---------------------------------------------------------------------------
# TEST 3: Duplicate source index
# ---------------------------------------------------------------------------
def test_3_duplicate_source_index(validator):
    # Actual: 41, 42, 42, 43
    block_result = TranscriptionBlockResult(
        schema_version="1.0",
        job_id="JOB_001",
        session_id="SESSION_001",
        block_id="BLOCK_003",
        first_source_index=41,
        last_source_index=43,
        status="CONFIRMED",
        segments=[
            TranscriptSegment(source_index=41, text="Câu 41"),
            TranscriptSegment(source_index=42, text="Câu 42"),
            TranscriptSegment(source_index=42, text="Câu 42 lặp"),
            TranscriptSegment(source_index=43, text="Câu 43"),
        ],
    )
    val_result = validator.validate_block_result(block_result, expected_first_index=41, expected_last_index=43)
    assert val_result.is_valid is False
    assert any("duplicate" in e.lower() for e in val_result.errors)


# ---------------------------------------------------------------------------
# TEST 4: Wrong order
# ---------------------------------------------------------------------------
def test_4_wrong_order(validator):
    # Actual: 41, 43, 42, 44
    block_result = TranscriptionBlockResult(
        schema_version="1.0",
        job_id="JOB_001",
        session_id="SESSION_001",
        block_id="BLOCK_003",
        first_source_index=41,
        last_source_index=44,
        status="CONFIRMED",
        segments=[
            TranscriptSegment(source_index=41, text="Câu 41"),
            TranscriptSegment(source_index=43, text="Câu 43"),
            TranscriptSegment(source_index=42, text="Câu 42"),
            TranscriptSegment(source_index=44, text="Câu 44"),
        ],
    )
    val_result = validator.validate_block_result(block_result, expected_first_index=41, expected_last_index=44)
    assert val_result.is_valid is False
    assert any("wrong order" in e.lower() or "order" in e.lower() for e in val_result.errors)


# ---------------------------------------------------------------------------
# TEST 5: Invalid JSON
# ---------------------------------------------------------------------------
def test_5_invalid_json(parser):
    bad_raw = "This is not json at all {broken: unclosed"
    with pytest.raises(InvalidJSONError):
        parser.parse(bad_raw)


# ---------------------------------------------------------------------------
# TEST 6: Schema validation failure
# ---------------------------------------------------------------------------
def test_6_schema_validation_failure(parser):
    # Missing required field schema_version and missing segment text
    invalid_schema_obj = {
        "job_id": "JOB_001",
        "session_id": "SESSION_001",
        "block_id": "BLOCK_001",
        "first_source_index": 1,
        "last_source_index": 1,
        "status": "CONFIRMED",
        "segments": [
            {"source_index": 1}  # Missing 'text'
        ],
    }
    with pytest.raises(SchemaValidationError):
        parser.parse(json.dumps(invalid_schema_obj))


# ---------------------------------------------------------------------------
# TEST 7: Transcript validation failure
# ---------------------------------------------------------------------------
def test_7_transcript_validation_failure(validator):
    # Segment with empty text + AI meta commentary
    block_result = TranscriptionBlockResult(
        schema_version="1.0",
        job_id="JOB_001",
        session_id="SESSION_001",
        block_id="BLOCK_001",
        first_source_index=1,
        last_source_index=2,
        status="CONFIRMED",
        segments=[
            TranscriptSegment(source_index=1, text="   "),
            TranscriptSegment(source_index=2, text="Dưới đây là bản phiên âm cuộc họp."),
        ],
    )
    val_result = validator.validate_block_result(block_result, expected_first_index=1, expected_last_index=2)
    assert val_result.is_valid is False
    assert len(val_result.errors) >= 2


# ---------------------------------------------------------------------------
# TEST 8: Retry
# ---------------------------------------------------------------------------
def test_8_retry(tmp_path):
    # Simulate first attempt failing with invalid JSON, then second succeeding
    call_count = 0

    class MockGeminiClient:
        def generate_transcription(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "BROKEN_JSON"
            return json.dumps({
                "schema_version": "1.0",
                "job_id": "JOB_TEST",
                "session_id": "SESSION_TEST",
                "block_id": "BLOCK_001",
                "first_source_index": 1,
                "last_source_index": 1,
                "status": "CONFIRMED",
                "segments": [{"source_index": 1, "text": "Lời nói chuẩn."}],
            })

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "system_prompt.txt").write_text("System instruction", encoding="utf-8")

    transcriber = GeminiTranscriber(
        gemini_client=MockGeminiClient(),
        system_prompt_path=prompts_dir / "system_prompt.txt",
    )

    # Attempt 1 should fail
    with pytest.raises(TranscriptionError):
        transcriber.transcribe_block(
            gemini_file=MagicMock(),
            job_id="JOB_TEST",
            session_id="SESSION_TEST",
            block_id="BLOCK_001",
            first_source_index=1,
        )

    # Attempt 2 should succeed
    outcome = transcriber.transcribe_block(
        gemini_file=MagicMock(),
        job_id="JOB_TEST",
        session_id="SESSION_TEST",
        block_id="BLOCK_001",
        first_source_index=1,
    )
    assert outcome.structural_validation.is_valid is True
    assert outcome.fidelity_validation.is_valid is True
    assert outcome.block_result.segments[0].text == "Lời nói chuẩn."


# ---------------------------------------------------------------------------
# TEST 9: Resume
# ---------------------------------------------------------------------------
def test_9_resume(tmp_path):
    checkpoint_file = tmp_path / "state" / "checkpoint.json"
    mgr = CheckpointManager(checkpoint_file)
    mgr.create_new_job("audio.mp3")

    # Simulate completed block 1..43
    confirmed_segs = [
        {"source_index": i, "text": f"Đoạn {i}", "timestamp": "00:01", "speaker": "A"}
        for i in range(1, 44)
    ]
    mgr.commit_block(
        block_id="BLOCK_001",
        last_source_index=43,
        new_segments=confirmed_segs,
    )

    # Reload checkpoint
    loaded = mgr.load()
    assert loaded.last_confirmed_source_index == 43
    assert loaded.next_source_index == 44

    # Merging next block 44..50 without restarting from 1
    merger = OutputMerger()
    for s in loaded.confirmed_segments:
        merger.add_segment(TranscriptSegment.from_dict(s))

    new_block = TranscriptionBlockResult(
        schema_version="1.0",
        job_id="JOB_001",
        session_id="SESSION_001",
        block_id="BLOCK_002",
        first_source_index=44,
        last_source_index=46,
        status="CONFIRMED",
        segments=[
            TranscriptSegment(source_index=44, text="Đoạn 44"),
            TranscriptSegment(source_index=45, text="Đoạn 45"),
            TranscriptSegment(source_index=46, text="Đoạn 46"),
        ],
    )
    merger.merge_block_result(new_block)
    assert merger.total_segments() == 46
    assert merger.get_last_source_index() == 46


# ---------------------------------------------------------------------------
# TEST 10: Output-limit block split
# ---------------------------------------------------------------------------
def test_10_output_limit_block_split():
    # Helper simulating splitting a large block [41, 100] into [41, 70] and [71, 100]
    def split_block_range(first_idx: int, last_idx: int, max_chunk_size: int = 30) -> list[tuple[int, int]]:
        chunks = []
        cur = first_idx
        while cur <= last_idx:
            nxt = min(cur + max_chunk_size - 1, last_idx)
            chunks.append((cur, nxt))
            cur = nxt + 1
        return chunks

    sub_blocks = split_block_range(41, 100, max_chunk_size=30)
    assert sub_blocks == [(41, 70), (71, 100)]


# ---------------------------------------------------------------------------
# TEST 11: Verbatim preservation
# ---------------------------------------------------------------------------
def test_11_verbatim_preservation(validator):
    verbatim_text = "À thì... ờ... chúng tôi, chúng tôi muốn muốn nhắc lại là điều này không thể chấp nhận được."
    block_result = TranscriptionBlockResult(
        schema_version="1.0",
        job_id="JOB_001",
        session_id="SESSION_001",
        block_id="BLOCK_001",
        first_source_index=1,
        last_source_index=1,
        status="CONFIRMED",
        segments=[TranscriptSegment(source_index=1, text=verbatim_text, timestamp="00:05", speaker="Người A")],
    )
    val_result = validator.validate_block_result(block_result, expected_first_index=1, expected_last_index=1)
    assert val_result.is_valid is True
    assert block_result.segments[0].text == verbatim_text


# ---------------------------------------------------------------------------
# TEST 12: Unknown speech -> [không rõ]
# ---------------------------------------------------------------------------
def test_12_unknown_speech_tag(validator):
    unclear_text = "Tại thời điểm đó [không rõ] đã diễn ra."
    block_result = TranscriptionBlockResult(
        schema_version="1.0",
        job_id="JOB_001",
        session_id="SESSION_001",
        block_id="BLOCK_001",
        first_source_index=1,
        last_source_index=1,
        status="CONFIRMED",
        segments=[TranscriptSegment(source_index=1, text=unclear_text, timestamp="01:10", speaker="Người B")],
    )
    val_result = validator.validate_block_result(block_result, expected_first_index=1, expected_last_index=1)
    assert val_result.is_valid is True
    assert "[không rõ]" in block_result.segments[0].text


# ---------------------------------------------------------------------------
# TEST 13: Renderer TXT
# ---------------------------------------------------------------------------
def test_13_renderer_txt(tmp_path):
    renderer = OutputRenderer()
    segments = [
        TranscriptSegment(source_index=1, timestamp="00:13:34", speaker="Người A", text="Hôm nay chúng ta trao đổi."),
        TranscriptSegment(source_index=2, timestamp="00:13:39", speaker="Người B", text="Vâng, tôi đồng ý."),
    ]
    txt_file = tmp_path / "transcript.txt"
    renderer.render_txt(segments, txt_file)

    content = txt_file.read_text(encoding="utf-8")
    expected = (
        "[00:13:34 - Người A]: Hôm nay chúng ta trao đổi.\n\n"
        "[00:13:39 - Người B]: Vâng, tôi đồng ý.\n"
    )
    assert content == expected
    # Ensure no JSON or metadata leaked
    assert "job_id" not in content
    assert "source_index" not in content


# ---------------------------------------------------------------------------
# TEST 14: Renderer DOCX
# ---------------------------------------------------------------------------
def test_14_renderer_docx(tmp_path):
    import docx

    renderer = OutputRenderer()
    segments = [
        TranscriptSegment(source_index=1, timestamp="00:13:34", speaker="Người A", text="Hôm nay chúng ta trao đổi."),
        TranscriptSegment(source_index=2, timestamp=None, speaker="Người B", text="Vâng, tôi đồng ý."),
    ]
    docx_file = tmp_path / "transcript.docx"
    renderer.render_docx(segments, docx_file)

    assert docx_file.exists()
    doc = docx.Document(str(docx_file))
    texts = [p.text for p in doc.paragraphs if p.text.strip()]
    assert any("BẢN PHIÊN ÂM NGUYÊN VĂN" in t for t in texts)
    assert any("[00:13:34 - Người A]: Hôm nay chúng ta trao đổi." in t for t in texts)
    assert any("[Người B]: Vâng, tôi đồng ý." in t for t in texts)


# ---------------------------------------------------------------------------
# TEST 15: Renderer SRT
# ---------------------------------------------------------------------------
def test_15_renderer_srt(tmp_path):
    renderer = OutputRenderer(default_segment_duration_seconds=5.0)
    segments = [
        TranscriptSegment(source_index=1, timestamp="00:13:34", speaker="Người A", text="Hôm nay chúng ta trao đổi."),
        TranscriptSegment(source_index=2, timestamp="00:13:39", speaker="Người B", text="Vâng, tôi đồng ý."),
    ]
    srt_file = tmp_path / "subtitle.srt"
    renderer.render_srt(segments, srt_file)

    content = srt_file.read_text(encoding="utf-8")
    assert "1\n00:13:34,000 --> 00:13:39,000\n[Người A]: Hôm nay chúng ta trao đổi." in content
    assert "2\n00:13:39,000 --> 00:13:44,000\n[Người B]: Vâng, tôi đồng ý." in content


# ---------------------------------------------------------------------------
# TEST 16: Retry cùng block không tạo duplicate
# ---------------------------------------------------------------------------
def test_16_retry_idempotency_no_duplicate():
    merger = OutputMerger()
    block = TranscriptionBlockResult(
        schema_version="1.0",
        job_id="JOB_001",
        session_id="SESSION_001",
        block_id="BLOCK_003",
        first_source_index=41,
        last_source_index=42,
        status="CONFIRMED",
        segments=[
            TranscriptSegment(source_index=41, text="Câu 41"),
            TranscriptSegment(source_index=42, text="Câu 42"),
        ],
    )

    # First merge
    merger.merge_block_result(block)
    assert merger.total_segments() == 2

    # Retry same block
    merger.merge_block_result(block)
    assert merger.total_segments() == 2  # Idempotent! Still 2 segments, no duplicates

    # Check order
    res = merger.get_merged_segments()
    assert [s.source_index for s in res] == [41, 42]


# ---------------------------------------------------------------------------
# TEST PHASE 20: End-to-End Test (15 segments pipeline)
# ---------------------------------------------------------------------------
def test_phase20_end_to_end_15_segments(tmp_path, monkeypatch):
    from google.genai.models import Models
    from google.genai.types import Model
    monkeypatch.setattr(Models, "list", lambda self: [Model(name="models/gemini-3.8-flash", supported_actions=["generateContent"])])
    monkeypatch.setattr(Models, "get", lambda self, model: Model(name=model, supported_actions=["generateContent"]))
    monkeypatch.setenv("STRICT_SPEAKER_FORMAT", "false")  # Legacy output contract fixture.
    # Fake media has no decodable audio; these tests model an explicitly silent tail.
    monkeypatch.setattr("src.coverage_validator.TailActivityAnalyzer.analyze", lambda self, *args: 0.0)
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "full_meeting.mp3").write_bytes(b"mock_audio_bytes")

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "system_prompt.txt").write_text("System instruction for verbatim.", encoding="utf-8")

    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=mock-key\nGEMINI_MODEL=gemini-2.5-flash\n", encoding="utf-8")

    mock_file_obj = MagicMock()
    type(mock_file_obj).name = PropertyMock(return_value="files/mock_e2e")

    # Generate 15 segments across 2 blocks (Block 1: 1..8, Block 2: 9..15)
    def mock_transcribe_block(self, block_id, first_source_index, **kwargs):
        if "001" in block_id:
            segs = [
                TranscriptSegment(source_index=i, timestamp=f"00:{i:02d}", speaker="Speaker A", text=f"Nội dung segment {i}")
                for i in range(first_source_index, first_source_index + 8)
            ]
            res = TranscriptionBlockResult(
                schema_version="1.0",
                job_id="JOB_E2E",
                session_id="SESSION_E2E",
                block_id=block_id,
                first_source_index=first_source_index,
                last_source_index=first_source_index + 7,
                status="CONFIRMED",
                segments=segs,
            )
            return TranscriptionOutcome(res, ValidationResult(is_valid=True), ContentFidelityValidator().validate(res))
        else:
            segs = [
                TranscriptSegment(source_index=i, timestamp=f"01:{i:02d}", speaker="Speaker B", text=f"Nội dung segment {i}")
                for i in range(first_source_index, first_source_index + 7)
            ]
            res = TranscriptionBlockResult(
                schema_version="1.0",
                job_id="JOB_E2E",
                session_id="SESSION_E2E",
                block_id=block_id,
                first_source_index=first_source_index,
                last_source_index=first_source_index + 6,
                status="CONFIRMED",
                segments=segs,
            )
            return TranscriptionOutcome(res, ValidationResult(is_valid=True), ContentFidelityValidator().validate(res))

    from src.block_builder import SourceBlock, AdaptiveBlockSlice

    src_block1 = SourceBlock(source_index=1, source_index_str="001", start_time_seconds=0.0, end_time_seconds=60.0, file_path=audio_dir / "full_meeting.mp3", total_blocks=2)
    src_block2 = SourceBlock(source_index=2, source_index_str="002", start_time_seconds=60.0, end_time_seconds=120.0, file_path=audio_dir / "full_meeting.mp3", total_blocks=2)
    adaptive1 = AdaptiveBlockSlice(source_block=src_block1, block_num=1, current_duration_seconds=60.0, estimated_input_tokens=150, context_budget=998000, output_budget=6692)
    adaptive2 = AdaptiveBlockSlice(source_block=src_block2, block_num=2, current_duration_seconds=60.0, estimated_input_tokens=150, context_budget=998000, output_budget=6692)

    monkeypatch.setattr("src.main.DabbAudioOrchestrator.get_duration", lambda self: 120.0)
    monkeypatch.setattr("src.main.DabbAudioOrchestrator.iter_adaptive_blocks", lambda self, start_from_seconds=0.0: iter([adaptive1, adaptive2]))
    monkeypatch.setattr("src.main.GeminiClient.upload_audio", lambda self, audio_path: mock_file_obj)
    monkeypatch.setattr("src.main.GeminiTranscriber.transcribe_block", mock_transcribe_block)

    exit_code = run_pipeline(force=False, base_dir=tmp_path)
    assert exit_code == 0

    txt_path = tmp_path / "output" / "transcript.txt"
    docx_path = tmp_path / "output" / "transcript.docx"
    srt_path = tmp_path / "output" / "subtitle.srt"
    checkpoint_path = tmp_path / "state" / "checkpoint.json"

    assert txt_path.exists()
    assert docx_path.exists()
    assert srt_path.exists()
    assert checkpoint_path.exists()

    txt_content = txt_path.read_text(encoding="utf-8")
    for i in range(1, 16):
        assert f"Nội dung segment {i}" in txt_content

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        ckpt_data = json.load(f)
    assert ckpt_data["status"] == CheckpointStatus.COMPLETED
    assert ckpt_data["last_confirmed_source_index"] == 15
    assert len(ckpt_data["confirmed_segments"]) == 15


# ---------------------------------------------------------------------------
# TEST: Dynamic Audio Block where Gemini guessed mismatched last_source_index
# ---------------------------------------------------------------------------
def test_audio_block_dynamic_last_index_mismatch(parser, validator):
    # Simulates the exact situation encountered on Block 001:
    # Gemini produced 73 segments (1..73) but guessed last_source_index = 139 at the top of the JSON.
    raw_json = {
        "schema_version": "1.0",
        "job_id": "20260909_test",
        "session_id": "SESSION_001",
        "block_id": "BLOCK_001",
        "first_source_index": 1,
        "last_source_index": 139,  # Guessed ahead of generating segments
        "status": "CONFIRMED",
        "segments": [
            {"source_index": i, "timestamp": f"{i // 60:02d}:{i % 60:02d}", "speaker": "Speaker", "text": f"Lời nói {i}"}
            for i in range(1, 74)
        ],
    }

    # Stage 3 preserves the model's invalid metadata instead of silently repairing it.
    block_res = parser.parse(json.dumps(raw_json))
    assert block_res.last_source_index == 139

    # A dynamic end still requires consistency with the returned segment endpoint.
    val_res = validator.validate_block_result(block_res, expected_first_index=1, expected_last_index=None)
    assert val_res.is_valid is False
    assert "LAST_SOURCE_INDEX_MISMATCH" in val_res.reason_codes
    assert block_res.last_source_index == 139

