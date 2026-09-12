"""Unit tests for BlockBuilder and Milestone 2 chunking."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from src.block_builder import AudioBlockBuilder, SourceBlock, BlockBuilderError


def test_block_builder_initialization_missing_file(tmp_path: Path):
    non_existent = tmp_path / "missing.mp3"
    with pytest.raises(BlockBuilderError):
        AudioBlockBuilder(audio_path=non_existent)


def test_block_builder_total_blocks_calculation(tmp_path: Path):
    dummy_audio = tmp_path / "sample.mp3"
    dummy_audio.write_bytes(b"dummy audio content")

    builder = AudioBlockBuilder(audio_path=dummy_audio, block_size_seconds=300)

    # Mock get_duration to return 650 seconds (should produce 3 blocks: 300 + 300 + 50)
    builder._duration = 650.0

    assert builder.total_blocks() == 3


def test_block_builder_slice_and_iter(tmp_path: Path):
    dummy_audio = tmp_path / "interview.wav"
    dummy_audio.write_bytes(b"RIFF dummy wav data")

    builder = AudioBlockBuilder(
        audio_path=dummy_audio,
        block_size_seconds=60,
        cache_dir=tmp_path / "cache",
        job_id="test_job",
    )
    builder._duration = 150.0  # 3 blocks: 0-60, 60-120, 120-150

    # Mock subprocess.run for ffmpeg slice
    with patch("subprocess.run") as mock_run:
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        blocks = builder.build_blocks()
        assert len(blocks) == 3

        # Check block 1
        assert blocks[0].source_index == 1
        assert blocks[0].source_index_str == "001"
        assert blocks[0].start_time_seconds == 0.0
        assert blocks[0].end_time_seconds == 60.0

        # Check block 2
        assert blocks[1].source_index == 2
        assert blocks[1].source_index_str == "002"
        assert blocks[1].start_time_seconds == 60.0
        assert blocks[1].end_time_seconds == 120.0

        # Check block 3
        assert blocks[2].source_index == 3
        assert blocks[2].source_index_str == "003"
        assert blocks[2].start_time_seconds == 120.0
        assert blocks[2].end_time_seconds == 150.0
