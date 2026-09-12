"""Unit tests for checkpoint management, configuration, and audio discovery."""

from src.transcriber import TranscriptionOutcome
from src.fidelity_validator import ContentFidelityValidator


import os
import json
from pathlib import Path
import pytest

from src.checkpoint import (
    CheckpointManager,
    CheckpointStatus,
    CheckpointData,
    generate_job_id,
)
from src.config import load_config, ConfigurationError
from src.audio_manager import AudioManager, AudioNotFoundError, UnsupportedAudioFormatError


# ---------------------------------------------------------------------------
# Test 1: Checkpoint Create
# ---------------------------------------------------------------------------
def test_checkpoint_create(tmp_path: Path):
    checkpoint_file = tmp_path / "checkpoint.json"
    manager = CheckpointManager(checkpoint_file)

    data = manager.create_new_job("interview_sample.mp3")

    assert data.job_id is not None
    assert "interview_sample" in data.job_id
    assert data.file_name == "interview_sample.mp3"
    assert data.status == CheckpointStatus.NOT_STARTED
    assert data.file_id is None
    assert data.created_at is not None
    assert data.updated_at is not None
    assert checkpoint_file.exists()


# ---------------------------------------------------------------------------
# Test 2: Checkpoint Save
# ---------------------------------------------------------------------------
def test_checkpoint_save(tmp_path: Path):
    checkpoint_file = tmp_path / "checkpoint.json"
    manager = CheckpointManager(checkpoint_file)

    data = manager.create_new_job("test_audio.wav")
    assert checkpoint_file.exists()

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        saved_json = json.load(f)

    assert saved_json["job_id"] == data.job_id
    assert saved_json["file_name"] == "test_audio.wav"
    assert saved_json["status"] == CheckpointStatus.NOT_STARTED

    # Ensure no temporary leftover files exist
    assert not (tmp_path / "checkpoint.tmp").exists()


# ---------------------------------------------------------------------------
# Test 3: Checkpoint Load
# ---------------------------------------------------------------------------
def test_checkpoint_load(tmp_path: Path):
    checkpoint_file = tmp_path / "checkpoint.json"
    manager = CheckpointManager(checkpoint_file)

    # When file does not exist, returns empty default state
    empty_data = manager.load()
    assert empty_data.status == CheckpointStatus.NOT_STARTED
    assert empty_data.job_id is None

    # Write a known state
    known_payload = {
        "job_id": "20260909_001_sample",
        "file_name": "sample.mp3",
        "file_id": "files/gemini_12345",
        "status": CheckpointStatus.COMPLETED,
        "last_confirmed_source_index": None,
        "total_source_segments": None,
        "transcript_file": "transcript.txt",
        "created_at": "2026-09-09T00:00:00Z",
        "updated_at": "2026-09-09T00:01:00Z",
        "error_message": None,
    }
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(known_payload, f)

    loaded_data = manager.load()
    assert loaded_data.job_id == "20260909_001_sample"
    assert loaded_data.file_name == "sample.mp3"
    assert loaded_data.file_id == "files/gemini_12345"
    assert loaded_data.status == CheckpointStatus.COMPLETED
    assert loaded_data.transcript_file == "transcript.txt"


# ---------------------------------------------------------------------------
# Test 4: Checkpoint Update
# ---------------------------------------------------------------------------
def test_checkpoint_update(tmp_path: Path):
    checkpoint_file = tmp_path / "checkpoint.json"
    manager = CheckpointManager(checkpoint_file)
    manager.create_new_job("voice.m4a")

    # Update to UPLOADING
    manager.update_status(CheckpointStatus.UPLOADING)
    assert manager.current_data.status == CheckpointStatus.UPLOADING

    # Update to PROCESSING with file_id
    manager.update_status(CheckpointStatus.PROCESSING, file_id="files/gemini_abc")
    assert manager.current_data.status == CheckpointStatus.PROCESSING
    assert manager.current_data.file_id == "files/gemini_abc"

    # Update to COMPLETED with transcript_file
    manager.update_status(CheckpointStatus.COMPLETED, transcript_file="transcript.txt")
    assert manager.current_data.status == CheckpointStatus.COMPLETED
    assert manager.current_data.transcript_file == "transcript.txt"

    # Verify invalid status raises error
    with pytest.raises(ValueError):
        manager.update_status("UNKNOWN_STATUS")


# ---------------------------------------------------------------------------
# Test 5: Missing API Key
# ---------------------------------------------------------------------------
def test_missing_api_key(tmp_path: Path, monkeypatch):
    # Ensure GEMINI_API_KEY is not in environment
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError) as exc_info:
        load_config(base_dir=tmp_path)

    assert "GEMINI_API_KEY is not set" in str(exc_info.value)

    # Also test placeholder value rejection
    monkeypatch.setenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    with pytest.raises(ConfigurationError) as exc_info_placeholder:
        load_config(base_dir=tmp_path)
    assert "placeholder" in str(exc_info_placeholder.value).lower()


# ---------------------------------------------------------------------------
# Test 6: Missing Audio
# ---------------------------------------------------------------------------
def test_missing_audio(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manager = AudioManager(audio_dir)
    with pytest.raises(AudioNotFoundError) as exc_info:
        manager.find_target_audio()

    assert "No audio file found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 7: Unsupported Audio Format
# ---------------------------------------------------------------------------
def test_unsupported_audio_format(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "document.pdf").write_text("dummy", encoding="utf-8")
    (audio_dir / "script.py").write_text("print()", encoding="utf-8")

    manager = AudioManager(audio_dir)
    with pytest.raises(UnsupportedAudioFormatError) as exc_info:
        manager.find_target_audio()

    assert "none have supported extensions" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 8: Idempotency Check
# ---------------------------------------------------------------------------
def test_idempotency_detection(tmp_path: Path):
    checkpoint_file = tmp_path / "checkpoint.json"
    transcript_file = tmp_path / "transcript.txt"
    transcript_file.write_text("Nội dung phiên âm mẫu.", encoding="utf-8")

    manager = CheckpointManager(checkpoint_file)
    manager.create_new_job("audio.mp3")
    manager.update_status(CheckpointStatus.COMPLETED, transcript_file="transcript.txt")

    # Should detect as completed
    assert manager.is_completed_for("audio.mp3", transcript_file) is True

    # Different audio file name should return False
    assert manager.is_completed_for("another.mp3", transcript_file) is False

    # Empty transcript file should return False
    empty_transcript = tmp_path / "empty.txt"
    empty_transcript.touch()
    assert manager.is_completed_for("audio.mp3", empty_transcript) is False


# ---------------------------------------------------------------------------
# Test 9: End-to-End Pipeline Execution (Mocked Gemini)
# ---------------------------------------------------------------------------
def test_end_to_end_pipeline_mock(tmp_path: Path, monkeypatch):
    from unittest.mock import MagicMock, PropertyMock
    from src.main import run_pipeline

    # Setup directories
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "sample_interview.mp3").write_bytes(b"dummy_audio_bytes_12345")

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "system_prompt.txt").write_text("System instruction for verbatim.", encoding="utf-8")

    # Setup .env in tmp_path
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=test-mock-api-key\nGEMINI_MODEL=gemini-2.5-flash\n", encoding="utf-8")

    mock_file_obj = MagicMock()
    type(mock_file_obj).name = PropertyMock(return_value="files/gemini_mock_abc123")
    from src.response_parser import TranscriptionBlockResult, TranscriptSegment
    from src.transcript_validator import ValidationResult

    mock_block_result = TranscriptionBlockResult(
        schema_version="1.0",
        job_id="20260909_sample",
        session_id="SESSION_001",
        block_id="BLOCK_001",
        first_source_index=1,
        last_source_index=2,
        status="CONFIRMED",
        segments=[
            TranscriptSegment(source_index=1, text="À thì hôm nay chúng ta thảo luận nhé.", timestamp="00:00", speaker="Người nói 1"),
            TranscriptSegment(source_index=2, text="Dạ vâng ạ.", timestamp="00:03", speaker="Người nói 2"),
        ],
    )

    from src.block_builder import SourceBlock, AdaptiveBlockSlice

    dummy_source_block = SourceBlock(
        source_index=1,
        source_index_str="001",
        start_time_seconds=0.0,
        end_time_seconds=60.0,
        file_path=audio_dir / "sample_interview.mp3",
        total_blocks=1,
    )
    dummy_adaptive_slice = AdaptiveBlockSlice(
        source_block=dummy_source_block,
        block_num=1,
        current_duration_seconds=60.0,
        estimated_input_tokens=150,
        context_budget=998000,
        output_budget=6692,
    )

    monkeypatch.setattr("src.main.DabbAudioOrchestrator.get_duration", lambda self: 60.0)
    monkeypatch.setattr("src.main.DabbAudioOrchestrator.iter_adaptive_blocks", lambda self, start_from_seconds=0.0: iter([dummy_adaptive_slice]))
    monkeypatch.setattr("src.main.GeminiClient.upload_audio", lambda self, audio_path: mock_file_obj)
    monkeypatch.setattr(
        "src.main.GeminiTranscriber.transcribe_block",
        lambda self, **kwargs: TranscriptionOutcome(mock_block_result, ValidationResult(is_valid=True), ContentFidelityValidator().validate(mock_block_result)),
    )

    # 1. First run: should succeed and create transcript + docx + srt + checkpoint
    exit_code = run_pipeline(force=False, base_dir=tmp_path)
    assert exit_code == 0

    transcript_path = tmp_path / "output" / "transcript.txt"
    docx_path = tmp_path / "output" / "transcript.docx"
    srt_path = tmp_path / "output" / "subtitle.srt"
    checkpoint_path = tmp_path / "state" / "checkpoint.json"

    assert transcript_path.exists()
    assert docx_path.exists()
    assert srt_path.exists()
    assert "[00:00 - Người nói 1]: À thì hôm nay chúng ta thảo luận nhé." in transcript_path.read_text(encoding="utf-8")

    assert checkpoint_path.exists()
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)

    assert ckpt["status"] == CheckpointStatus.COMPLETED
    assert ckpt["file_name"] == "sample_interview.mp3"
    assert ckpt["file_id"] == "files/gemini_mock_abc123"
    assert ckpt["transcript_file"] == "transcript.txt"

    # 2. Second run without force: should detect idempotency and return 0
    exit_code_cached = run_pipeline(force=False, base_dir=tmp_path)
    assert exit_code_cached == 0

    # 3. Third run with force: should re-process successfully
    exit_code_forced = run_pipeline(force=True, base_dir=tmp_path)
    assert exit_code_forced == 0

