"""Acceptance tests for Dynamic Adaptive Block Builder (DABB).

Tests all 12 core requirements + End-to-End Low/Medium/High density pipeline test.
"""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from src.config import AppConfig
from src.token_estimator import TokenEstimator, OutputRatioTracker
from src.failure_classifier import FailureType, classify_failure
from src.adaptive_block_builder import (
    SourceSegment,
    AdaptiveBlock,
    DynamicAdaptiveBlockBuilder,
)
from src.checkpoint import CheckpointManager, CheckpointStatus
from src.output_merger import OutputMerger
from src.response_parser import ResponseParser, TranscriptionBlockResult, TranscriptSegment
from src.transcript_validator import TranscriptValidator


@pytest.fixture
def mock_config():
    return AppConfig(
        gemini_api_key="mock_key",
        gemini_model="gemini-3.6-flash",
        base_dir=Path("."),
        audio_dir=Path("./audio"),
        output_dir=Path("./output"),
        state_dir=Path("./state"),
        prompts_dir=Path("./prompts"),
        system_prompt_path=Path("./prompts/system_prompt.txt"),
        output_transcript_path=Path("./output/transcript.txt"),
        output_docx_path=Path("./output/transcript.docx"),
        output_srt_path=Path("./output/subtitle.srt"),
        checkpoint_file_path=Path("./state/checkpoint.json"),
        schema_path=Path("./schemas/transcription_result.schema.json"),
        retry_max_attempts=3,
        retry_initial_delay_seconds=1.0,
        timeout_seconds=300,
        block_duration_seconds=300,
        cache_dir=Path("./.cache/blocks"),
        validator_max_retries=2,
        # DABB settings
        model_context_limit=100_000,
        model_max_output_tokens=4000,
        context_safety_margin=2000,
        output_safety_margin=1000,
        initial_target_input_tokens=1000,
        min_block_tokens=200,
        max_block_tokens=3000,
        block_growth_step=300,
        block_shrink_factor=0.5,
        default_output_ratio=1.2,
        min_duration_seconds=60.0,
        target_duration_seconds=300.0,
        max_duration_seconds=900.0,  # 15 mins
    )


# ---------------------------------------------------------------------------
# TEST 1: Low-density transcript -> block có thể dài hơn 5 phút (300s)
# ---------------------------------------------------------------------------
def test_1_low_density_transcript_exceeds_5_minutes(mock_config):
    # 10 short segments spaced over 600s (10 minutes)
    segments = [
        SourceSegment(
            source_index=i,
            timestamp=f"{i}:00",
            speaker="Speaker",
            source_text=f"Câu ngắn số {i}",
            start_time_seconds=(i - 1) * 60.0,
            end_time_seconds=i * 60.0,
        )
        for i in range(1, 11)
    ]
    builder = DynamicAdaptiveBlockBuilder(source_segments=segments, config=mock_config)
    block = builder.build_next_block(start_source_index=1)

    assert block is not None
    # Block duration is 10 minutes (600s), definitely > 300s (5 minutes)
    duration = block.end_time_seconds - block.start_time_seconds
    assert duration > 300.0
    assert duration == 600.0
    assert len(block.segments) == 10


# ---------------------------------------------------------------------------
# TEST 2: High-density transcript -> block tự động ngắn hơn (< 300s)
# ---------------------------------------------------------------------------
def test_2_high_density_transcript_shorter_than_5_minutes(mock_config):
    # Dense rapid dialogue: each segment has ~120 words and takes only 10s
    long_text = "Thảo luận chi tiết về các giải pháp kỹ thuật công nghệ thông tin trong phiên họp. " * 8
    segments = [
        SourceSegment(
            source_index=i,
            timestamp=f"00:{i*10:02d}",
            speaker="Speaker",
            source_text=long_text,
            start_time_seconds=(i - 1) * 10.0,
            end_time_seconds=i * 10.0,
        )
        for i in range(1, 40)
    ]
    builder = DynamicAdaptiveBlockBuilder(source_segments=segments, config=mock_config)
    block = builder.build_next_block(start_source_index=1)

    assert block is not None
    duration = block.end_time_seconds - block.start_time_seconds
    # Because target input tokens is 1000, it reaches capacity before 5 minutes (300s)
    assert duration < 300.0
    assert block.estimated_input_tokens <= mock_config.initial_target_input_tokens + 250


# ---------------------------------------------------------------------------
# TEST 3: Output estimate vượt budget -> block tự shrink
# ---------------------------------------------------------------------------
def test_3_output_estimate_budget_safety(mock_config):
    # Set small model_max_output_tokens
    tight_config = AppConfig(
        **{**mock_config.__dict__, "model_max_output_tokens": 500, "output_safety_margin": 100}
    )
    # Output budget = 500 - 100 = 400 tokens
    segments = [
        SourceSegment(source_index=i, source_text="Đoạn văn bản vừa phải có độ dài tương đối để test output budget." * 3)
        for i in range(1, 20)
    ]
    builder = DynamicAdaptiveBlockBuilder(source_segments=segments, config=tight_config)
    block = builder.build_next_block(start_source_index=1)

    assert block is not None
    # Block must strictly respect output budget
    safe_output_ceiling = tight_config.model_max_output_tokens - tight_config.output_safety_margin
    assert block.estimated_output_tokens <= safe_output_ceiling


# ---------------------------------------------------------------------------
# TEST 4: Context estimate vượt budget -> block tự shrink
# ---------------------------------------------------------------------------
def test_4_context_estimate_budget_safety(mock_config):
    # Set very tight context limit
    tight_config = AppConfig(
        **{**mock_config.__dict__, "model_context_limit": 800, "context_safety_margin": 200}
    )
    # Context budget = 800 - 200 = 600 tokens
    segments = [
        SourceSegment(source_index=i, source_text="Kiểm tra giới hạn tổng context window bao gồm cả prompt và output." * 2)
        for i in range(1, 30)
    ]
    builder = DynamicAdaptiveBlockBuilder(source_segments=segments, config=tight_config)
    block = builder.build_next_block(start_source_index=1)

    assert block is not None
    total_est = block.estimated_input_tokens + block.estimated_output_tokens
    assert total_est <= (tight_config.model_context_limit - tight_config.context_safety_margin)


# ---------------------------------------------------------------------------
# TEST 5: Block PASS -> block sau tăng kích thước (Adaptive Growth)
# ---------------------------------------------------------------------------
def test_5_adaptive_growth_on_pass(mock_config):
    segments = [
        SourceSegment(source_index=i, source_text=f"Nội dung segment {i}")
        for i in range(1, 50)
    ]
    builder = DynamicAdaptiveBlockBuilder(source_segments=segments, config=mock_config)
    initial_target = builder.current_target_tokens

    block = builder.build_next_block(start_source_index=1)
    assert block is not None

    # Simulate block PASS with small output (well below safe output limit)
    builder.on_block_confirmed(block, actual_input_tokens=200, actual_output_tokens=250)

    # Next target must increase by block_growth_step
    assert builder.current_target_tokens == initial_target + mock_config.block_growth_step


# ---------------------------------------------------------------------------
# TEST 6: Block FAIL vì output truncation -> retry với block nhỏ hơn (Adaptive Shrink)
# ---------------------------------------------------------------------------
def test_6_adaptive_shrink_on_size_failure(mock_config):
    segments = [
        SourceSegment(source_index=i, source_text="Nội dung dài cho mỗi câu nói trong buổi họp." * 5)
        for i in range(1, 30)
    ]
    builder = DynamicAdaptiveBlockBuilder(source_segments=segments, config=mock_config)
    initial_target = builder.current_target_tokens

    block1 = builder.build_next_block(start_source_index=1)
    assert block1 is not None

    # Classify error as SIZE_FAILURE
    failure = classify_failure("OUTPUT_TRUNCATED: Response exceeded maximum token output limit")
    assert failure == FailureType.SIZE_FAILURE

    shrunk = builder.on_block_failure(block1, failure)
    assert shrunk is True
    assert builder.current_target_tokens == int(initial_target * mock_config.block_shrink_factor)

    # Rebuilding starting from same source index should yield smaller or equal segment count
    block_retry = builder.build_next_block(start_source_index=1, block_id=block1.block_id)
    assert block_retry is not None
    assert len(block_retry.segments) <= len(block1.segments)


# ---------------------------------------------------------------------------
# TEST 7: Validation failure không liên quan size -> KHÔNG shrink
# ---------------------------------------------------------------------------
def test_7_no_shrink_on_structural_failure(mock_config):
    segments = [SourceSegment(source_index=i, source_text=f"Câu {i}") for i in range(1, 10)]
    builder = DynamicAdaptiveBlockBuilder(source_segments=segments, config=mock_config)
    initial_target = builder.current_target_tokens

    block = builder.build_next_block(start_source_index=1)
    assert block is not None

    # Structural error: duplicate source_index
    failure = classify_failure("Duplicate source_index detected: [42]")
    assert failure == FailureType.STRUCTURAL_FAILURE

    shrunk = builder.on_block_failure(block, failure)
    assert shrunk is False
    assert builder.current_target_tokens == initial_target  # Unchanged!


# ---------------------------------------------------------------------------
# TEST 8: Resume từ source_index chính xác
# ---------------------------------------------------------------------------
def test_8_resume_from_exact_source_index(mock_config, tmp_path):
    checkpoint_file = tmp_path / "checkpoint.json"
    mgr = CheckpointManager(checkpoint_file)
    mgr.create_new_job("meeting.mp3")

    # Simulate completed up to 196
    confirmed = [{"source_index": i, "text": f"Câu {i}"} for i in range(1, 197)]
    mgr.commit_block(block_id="BLOCK_005", last_source_index=196, new_segments=confirmed)

    loaded = mgr.load()
    assert loaded.last_confirmed_source_index == 196
    assert loaded.next_source_index == 197

    # DABB builds next block from 197
    all_segments = [SourceSegment(source_index=i, source_text=f"Câu {i}") for i in range(1, 250)]
    builder = DynamicAdaptiveBlockBuilder(source_segments=all_segments, config=mock_config)

    next_block = builder.build_next_block(start_source_index=loaded.next_source_index)
    assert next_block is not None
    assert next_block.first_source_index == 197


# ---------------------------------------------------------------------------
# TEST 9: Retry không tạo duplicate
# ---------------------------------------------------------------------------
def test_9_retry_no_duplicates():
    merger = OutputMerger()
    block = AdaptiveBlock(
        block_id="BLOCK_004",
        first_source_index=10,
        last_source_index=12,
        segments=[
            SourceSegment(source_index=10, source_text="Câu 10"),
            SourceSegment(source_index=11, source_text="Câu 11"),
            SourceSegment(source_index=12, source_text="Câu 12"),
        ],
    )

    # First merge
    for s in block.segments:
        merger.add_segment(TranscriptSegment(source_index=s.source_index, text=s.source_text))
    assert merger.total_segments() == 3

    # Retry of same block
    for s in block.segments:
        merger.add_segment(TranscriptSegment(source_index=s.source_index, text=s.source_text))
    assert merger.total_segments() == 3  # Strictly 3, no duplicates!


# ---------------------------------------------------------------------------
# TEST 10: Block luôn kết thúc tại source segment boundary
# ---------------------------------------------------------------------------
def test_10_segment_boundary_preservation(mock_config):
    segments = [
        SourceSegment(source_index=i, source_text=f"Nội dung đầy đủ của câu số {i}.")
        for i in range(1, 25)
    ]
    builder = DynamicAdaptiveBlockBuilder(source_segments=segments, config=mock_config)
    block = builder.build_next_block(start_source_index=1)

    assert block is not None
    # Verify last_source_index equals the source_index of the final segment in the block
    assert block.last_source_index == block.segments[-1].source_index
    assert block.segments[-1].source_text == f"Nội dung đầy đủ của câu số {block.last_source_index}."


# ---------------------------------------------------------------------------
# TEST 11: Timestamp không được dùng làm segment arithmetic
# ---------------------------------------------------------------------------
def test_11_no_timestamp_arithmetic(mock_config):
    # Segments with non-linear or irregular timestamps
    segments = [
        SourceSegment(source_index=1, timestamp="00:10", source_text="A"),
        SourceSegment(source_index=2, timestamp="00:15", source_text="B"),
        SourceSegment(source_index=3, timestamp="00:45", source_text="C"),  # Gap in time
        SourceSegment(source_index=4, timestamp="00:50", source_text="D"),
    ]
    builder = DynamicAdaptiveBlockBuilder(source_segments=segments, config=mock_config)
    block = builder.build_next_block(start_source_index=1)

    assert block is not None
    # Index progression must be purely integer-based (1, 2, 3, 4), ignoring timestamp deltas
    indices = [s.source_index for s in block.segments]
    assert indices == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# TEST 12: Duration không override token safety
# ---------------------------------------------------------------------------
def test_12_duration_does_not_override_token_safety(mock_config):
    # Segment duration is only 20 seconds, but text is massive
    huge_text = "Một đoạn hội thoại cực kỳ dài vượt qua toàn bộ giới hạn token cho phép. " * 30
    segments = [
        SourceSegment(source_index=1, start_time_seconds=0.0, end_time_seconds=10.0, source_text=huge_text),
        SourceSegment(source_index=2, start_time_seconds=10.0, end_time_seconds=20.0, source_text=huge_text),
    ]
    # Set target input tokens to 250 tokens
    tight_config = AppConfig(
        **{**mock_config.__dict__, "initial_target_input_tokens": 250, "min_duration_seconds": 60.0}
    )
    builder = DynamicAdaptiveBlockBuilder(source_segments=segments, config=tight_config)
    block = builder.build_next_block(start_source_index=1)

    assert block is not None
    # Even though duration is only 10s (below min_duration 60s), token safety took priority!
    assert len(block.segments) == 1
    assert block.last_source_index == 1


# ---------------------------------------------------------------------------
# SECTION 16: END-TO-END TEST (Low / Medium / High Density Sources)
# ---------------------------------------------------------------------------
def test_section_16_end_to_end_multidensity_pipeline(mock_config, tmp_path):
    schema_path = Path(__file__).resolve().parent.parent / "schemas" / "transcription_result.schema.json"
    parser = ResponseParser(schema_path=schema_path)
    validator = TranscriptValidator()
    checkpoint_file = tmp_path / "checkpoint.json"
    ckpt_mgr = CheckpointManager(checkpoint_file)
    ckpt_mgr.create_new_job("multidensity.mp3")
    merger = OutputMerger()

    # Create Multi-density source (30 segments total)
    # 1-10: Low density (short phrases)
    low_segs = [
        SourceSegment(source_index=i, timestamp=f"00:{i:02d}", speaker="A", source_text=f"Ngắn {i}")
        for i in range(1, 11)
    ]
    # 11-20: Medium density
    med_segs = [
        SourceSegment(source_index=i, timestamp=f"02:{i:02d}", speaker="B", source_text=f"Nội dung trung bình vừa phải của câu thảo luận số {i}")
        for i in range(11, 21)
    ]
    # 21-30: High density (long speeches)
    high_segs = [
        SourceSegment(source_index=i, timestamp=f"05:{i:02d}", speaker="C", source_text=f"Báo cáo chi tiết toàn diện về các giải pháp kỹ thuật công nghệ thông tin trong phiên họp số {i}. " * 3)
        for i in range(21, 31)
    ]
    all_source = low_segs + med_segs + high_segs

    builder = DynamicAdaptiveBlockBuilder(source_segments=all_source, config=mock_config)

    next_idx = 1
    total_blocks_processed = 0

    while next_idx <= 30:
        block = builder.build_next_block(start_source_index=next_idx)
        if block is None:
            break

        # Simulate Gemini returning structured JSON matching this block
        mock_response = {
            "schema_version": "1.0",
            "job_id": "JOB_DABB",
            "session_id": "SESSION_DABB",
            "block_id": block.block_id,
            "first_source_index": block.first_source_index,
            "last_source_index": block.last_source_index,
            "status": "CONFIRMED",
            "segments": [
                {
                    "source_index": s.source_index,
                    "timestamp": s.timestamp,
                    "speaker": s.speaker,
                    "text": s.source_text,
                }
                for s in block.segments
            ],
        }

        # Flow: Parse JSON -> JSON Schema -> Validator -> Merge -> Checkpoint
        parsed_block = parser.parse(json.dumps(mock_response, ensure_ascii=False))
        val_result = validator.validate_block_result(
            parsed_block,
            expected_first_index=block.first_source_index,
            expected_last_index=block.last_source_index,
        )
        assert val_result.is_valid is True

        merger.merge_block_result(parsed_block)

        ckpt_mgr.commit_block(
            block_id=block.block_id,
            last_source_index=block.last_source_index,
            new_segments=[s.to_dict() for s in parsed_block.segments],
        )

        builder.on_block_confirmed(
            block,
            actual_input_tokens=block.estimated_input_tokens,
            actual_output_tokens=block.estimated_output_tokens,
        )

        total_blocks_processed += 1
        next_idx = block.last_source_index + 1

    # Assertions on pipeline outcome
    assert merger.total_segments() == 30
    assert total_blocks_processed >= 2  # Adaptive splitting occurred across density variations
    merged = merger.get_merged_segments()
    assert [s.source_index for s in merged] == list(range(1, 31))  # No missing, strictly ordered!

    # Check logging was produced for all blocks
    assert len(builder.logs) >= total_blocks_processed
    for log_item in builder.logs:
        assert "block_id" in log_item
        assert "first_source_index" in log_item
        assert "last_source_index" in log_item
        assert "estimated_input_tokens" in log_item
        assert "decision" in log_item
