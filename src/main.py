"""Main CLI orchestration pipeline for Vietnamese Verbatim Transcription Agent (Milestone 3)."""

import sys
import shutil
import argparse
import time
from pathlib import Path

# Ensure Windows console prints UTF-8 Vietnamese characters properly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import load_config, ConfigurationError
from src.audio_manager import AudioManager, AudioError
from src.gemini_client import GeminiClient, GeminiClientError
from src.transcriber import GeminiTranscriber, TranscriptionError
from src.fidelity_validator import FidelityDecision
from src.checkpoint import CheckpointManager, CheckpointStatus, CheckpointError
from src.block_builder import DabbAudioOrchestrator, BlockBuilderError
from src.failure_classifier import classify_failure, FailureType
from src.response_parser import ResponseParser, TranscriptSegment
from src.transcript_validator import TranscriptValidator
from src.output_merger import OutputMerger
from src.output_renderer import OutputRenderer
from src.control_router import StandardControlRouter, ControlCommand


BANNER = """====================================
VIETNAMESE VERBATIM TRANSCRIPTION
(Milestone 3 - Structured Pipeline)
====================================="""


def run_pipeline(force: bool = False, base_dir: Path | None = None) -> int:
    """Execute the end-to-end transcription pipeline with Milestone 3 architecture:
    SOURCE AUDIO -> BLOCK BUILDER -> GEMINI -> STRUCTURED JSON -> SCHEMA -> VALIDATOR -> CHECKPOINT -> MERGER -> RENDERER (TXT, DOCX, SRT).

    Args:
        force: If True, bypasses completed checkpoint and re-processes audio.
        base_dir: Optional base directory path for configuration and file resolution.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    print(BANNER)

    # -------------------------------------------------------------
    # Step 1: Checking configuration & audio
    # -------------------------------------------------------------
    print("[1/5] Checking configuration...")
    try:
        config = load_config(base_dir=base_dir)
    except ConfigurationError as exc:
        print(f"\n[ERROR] Configuration failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error while loading config: {exc}", file=sys.stderr)
        return 1

    audio_mgr = AudioManager(audio_dir=config.audio_dir)
    try:
        target_audio = audio_mgr.find_target_audio()
    except AudioError as exc:
        print(f"\n[ERROR] Audio validation failed: {exc}", file=sys.stderr)
        return 1

    checkpoint_mgr = CheckpointManager(checkpoint_path=config.checkpoint_file_path)
    existing_data = checkpoint_mgr.load()

    # Idempotency check: avoid redundant billing / processing
    if not force and checkpoint_mgr.is_completed_for(
        file_name=target_audio.file_name,
        transcript_path=config.output_transcript_path,
    ):
        print(f"\nJob ID: {existing_data.job_id}")
        print(f"Audio:  {target_audio.file_name}")
        print(
            f"\n[NOTICE] This audio file has already been successfully transcribed.\n"
            f"Transcript: {config.output_transcript_path}\n"
            f"DOCX:       {config.output_docx_path}\n"
            f"SRT:        {config.output_srt_path}\n"
            f"Checkpoint: {config.checkpoint_file_path}\n"
            f"To re-run and overwrite existing outputs, run with: python -m src.main --force"
        )
        print("\nSTATUS: COMPLETED (Cached)")
        return 0

    # -------------------------------------------------------------
    # Step 2: Audio Inspection & Block Building (DABB-driven)
    # -------------------------------------------------------------
    print("[2/5] Inspecting audio & initializing DABB Orchestrator...")

    # Load system prompt early so DABB can factor it into token budget
    system_prompt_text = ""
    if config.system_prompt_path.exists():
        try:
            system_prompt_text = config.system_prompt_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    try:
        orchestrator = DabbAudioOrchestrator(
            audio_path=target_audio.path,
            config=config,
            system_prompt=system_prompt_text,
            cache_dir=config.cache_dir,
        )
        total_duration = orchestrator.get_duration()
    except BlockBuilderError as exc:
        print(f"\n[ERROR] Failed to initialize DABB Orchestrator: {exc}", file=sys.stderr)
        return 1

    # Estimate total blocks for display (approximate — DABB adjusts dynamically)
    initial_block_dur = orchestrator.current_block_duration_seconds
    total_blocks_approx = max(1, int(total_duration / max(1.0, initial_block_dur)))

    # Check if we should resume an existing partially completed job
    is_resuming = False
    start_from_block = 1
    next_source_index = 1
    merger = OutputMerger()

    if (
        not force
        and existing_data.file_name == target_audio.file_name
        and existing_data.last_confirmed_source_index is not None
        and existing_data.last_confirmed_source_index > 0
        and existing_data.status != CheckpointStatus.COMPLETED
    ):
        is_resuming = True
        current_job = existing_data
        next_source_index = (
            existing_data.next_source_index
            if existing_data.next_source_index is not None
            else (existing_data.last_confirmed_source_index + 1)
        )

        # Estimate starting block from last_confirmed_source_index
        # or find last confirmed block ID
        if existing_data.current_block_id and existing_data.current_block_id.startswith("BLOCK_"):
            try:
                last_block_num = int(existing_data.current_block_id.replace("BLOCK_", ""))
                start_from_block = last_block_num + 1
            except ValueError:
                start_from_block = 1
        else:
            start_from_block = 1

        # Populate merger with confirmed segments from checkpoint
        if existing_data.confirmed_segments:
            for s_dict in existing_data.confirmed_segments:
                merger.add_segment(TranscriptSegment.from_dict(s_dict))

        print(
            f"\n[RESUME] Phát hiện checkpoint hợp lệ. Đã hoàn thành {existing_data.last_confirmed_source_index} segment(s)."
        )
        print(f"[RESUME] Tiếp tục phiên âm từ next_source_index = {next_source_index} (Block {start_from_block:03d})...")
    else:
        # Initialize new job
        current_job = checkpoint_mgr.create_new_job(
            file_name=target_audio.file_name,
            total_source_segments=total_blocks_approx,
            status=CheckpointStatus.RUNNING,
        )

    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    print(f"\nJob ID:  {current_job.job_id}")
    print(f"Session: {current_job.session_id}")
    print(f"Audio:   {target_audio.file_name} ({target_audio.size_bytes / (1024 * 1024):.2f} MB, {minutes}m {seconds:02d}s)")
    print(f"Blocks:  ~{total_blocks_approx} block(s) est. (DABB adaptive, initial ~{initial_block_dur:.0f}s/block)")
    print(f"Model:   {config.gemini_model}\n")

    gemini_client = GeminiClient(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
        max_retries=config.retry_max_attempts,
        initial_delay_seconds=config.retry_initial_delay_seconds,
        timeout_seconds=config.timeout_seconds,
    )

    response_parser = ResponseParser(schema_path=config.schema_path)
    transcript_validator = TranscriptValidator()

    transcriber = GeminiTranscriber(
        gemini_client=gemini_client,
        system_prompt_path=config.system_prompt_path,
        validator=transcript_validator,
        response_parser=response_parser,
    )

    renderer = OutputRenderer()

    # -------------------------------------------------------------
    # Step 3: Sequential Block Transcription Loop (DABB-driven)
    # -------------------------------------------------------------
    print(f"[3/5] Transcribing blocks (DABB adaptive, resuming from {next_source_index})...")
    checkpoint_mgr.update_status(CheckpointStatus.PROCESSING)

    # Determine resume offset in seconds from checkpoint
    resume_from_seconds = 0.0
    if is_resuming and existing_data.last_confirmed_source_index:
        # Rough estimate: use last confirmed block duration from checkpoint if available
        # We'll skip blocks whose end time is before our resume point by tracking block_num
        pass

    last_file_id: str | None = None
    try:
        for adaptive_slice in orchestrator.iter_adaptive_blocks(start_from_seconds=resume_from_seconds):
            block = adaptive_slice.source_block
            block_id = f"BLOCK_{block.source_index_str}"

            # Skip already-processed blocks when resuming
            if is_resuming and adaptive_slice.block_num < start_from_block:
                continue

            start_m = int(block.start_time_seconds // 60)
            start_s = int(block.start_time_seconds % 60)
            end_m = int(block.end_time_seconds // 60)
            end_s = int(block.end_time_seconds % 60)
            time_range = f"{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}"
            dur_m = int(adaptive_slice.current_duration_seconds // 60)
            dur_s = int(adaptive_slice.current_duration_seconds % 60)

            print(
                f"  --> Block {block.source_index_str} [{time_range}] "
                f"({dur_m}m{dur_s:02d}s, ~{adaptive_slice.estimated_input_tokens}tok est.) "
                f"Uploading...",
                end="",
                flush=True,
            )
            gemini_file = gemini_client.upload_audio(audio_path=block.file_path)
            raw_name = getattr(gemini_file, "name", None)
            last_file_id = str(raw_name) if raw_name is not None else str(gemini_file)

            # Transcribe & Validate with retry on FAIL
            validated = False
            last_val_result = None
            failure_summary = "Validation failed"
            block_result = None
            last_exc: Exception | None = None

            for v_attempt in range(1, config.validator_max_retries + 1):
                print(f" Transcribing...", end="", flush=True)
                try:
                    outcome = transcriber.transcribe_block(
                        gemini_file=gemini_file,
                        job_id=current_job.job_id or "JOB_001",
                        session_id=current_job.session_id or "SESSION_001",
                        block_id=block_id,
                        first_source_index=next_source_index,
                        start_offset_seconds=block.start_time_seconds,
                    )
                    block_result = outcome.block_result
                    last_val_result = outcome.structural_validation
                    fidelity_result = outcome.fidelity_validation
                    last_exc = None
                except Exception as exc:
                    last_exc = exc
                    print(f" [Error: {exc}]", end="", flush=True)
                    # Classify and potentially shrink block for next attempt
                    failure_type = classify_failure(exc)
                    orchestrator.on_block_failure(failure_type)
                    if v_attempt < config.validator_max_retries:
                        print(f" Retrying ({v_attempt+1}/{config.validator_max_retries})...", end="", flush=True)
                        time.sleep(config.retry_initial_delay_seconds)
                        continue
                    else:
                        raise TranscriptionError(
                            f"Block {block_id} failed after {config.validator_max_retries} attempts: {exc}"
                        ) from exc

                if last_val_result.is_valid and fidelity_result.allows_confirmation:
                    validated = True
                    warn_tag = f" ({len(last_val_result.warnings)} warn)" if last_val_result.warnings else ""
                    next_dur = orchestrator.current_block_duration_seconds
                    print(f" [PASS{warn_tag}] Done. (next block ~{next_dur:.0f}s)")
                    if fidelity_result.decision == FidelityDecision.ACCEPT_WITH_WARNING:
                        print(f"  [ACCEPT_WITH_WARNING] {fidelity_result.summary()}")
                    break
                else:
                    # Classify validation failure for DABB feedback
                    failure_summary = (
                        f"Structural: {last_val_result.summary()}; "
                        f"Fidelity: {fidelity_result.summary()}"
                    )
                    failure_type = (
                        FailureType.FIDELITY_FAILURE
                        if not fidelity_result.allows_confirmation
                        else classify_failure(last_val_result.summary(), last_val_result)
                    )
                    orchestrator.on_block_failure(failure_type)
                    print(f" [{failure_summary}]", end="", flush=True)
                    if fidelity_result.decision == FidelityDecision.FAIL:
                        failure_summary = "DETERMINISTIC_FIDELITY_FAILURE: " + failure_summary
                        break
                    if v_attempt < config.validator_max_retries:
                        print(f" Retrying validation attempt {v_attempt+1}...", end="", flush=True)
                        time.sleep(config.retry_initial_delay_seconds)
                    else:
                        print(f" Max validation retries reached.")
                        if fidelity_result.decision == FidelityDecision.RETRY_REVIEW:
                            failure_summary = "BLOCK_REVIEW_REQUIRED: " + failure_summary

            if not validated or block_result is None:
                # Do NOT commit checkpoint on FAIL
                fail_summary = failure_summary
                raise TranscriptionError(f"Block {block.source_index_str} failed validation: {fail_summary}")

            # Feed success back to DABB (actual token usage from response metadata)
            actual_in = getattr(block_result, "actual_input_tokens", 0) or adaptive_slice.estimated_input_tokens
            actual_out = getattr(block_result, "actual_output_tokens", 0) or 0
            orchestrator.on_block_success(
                actual_input_tokens=actual_in,
                actual_output_tokens=actual_out,
            )

            # Merge block segments idempotently
            merger.merge_block_result(block_result)
            next_source_index = block_result.last_source_index + 1

            # Atomic Checkpoint Commit ONLY after validation & merge PASS
            # Persist quality signals in existing metadata using the same atomic commit.
            checkpoint_mgr.current_data.block_metrics.append({
                "block_id": block_id,
                "fidelity_status": fidelity_result.status.value,
                "fidelity_decision": fidelity_result.decision.value,
                "requires_quality_review": fidelity_result.needs_audio_review,
                "unknown_token_count": fidelity_result.unknown_token_count,
                "issues": [
                    {"source_index": i.source_index, "reason": i.indicator, "detail": i.detail}
                    for i in fidelity_result.issues
                ],
            })
            checkpoint_mgr.commit_block(
                block_id=block_id,
                last_source_index=block_result.last_source_index,
                new_segments=[s.to_dict() for s in block_result.segments],
                file_id=last_file_id,
            )

    except (GeminiClientError, TranscriptionError, BlockBuilderError, OSError) as exc:
        print(f"\n[ERROR] Pipeline failed during block transcription: {exc}", file=sys.stderr)
        checkpoint_mgr.update_status(
            CheckpointStatus.FAILED,
            file_id=last_file_id,
            error_message=str(exc),
        )
        return 1

    # -------------------------------------------------------------
    # Step 4: Render outputs (TXT, DOCX, SRT)
    # -------------------------------------------------------------
    print("[4/5] Rendering outputs (transcript.txt, transcript.docx, subtitle.srt)...")
    merged_segments = merger.get_merged_segments()
    try:
        rendered_paths = renderer.render_all(
            segments=merged_segments,
            base_dir=config.output_dir,
            job_id=current_job.job_id,
        )
        print(f"  --> TXT:  {rendered_paths['txt']}")
        print(f"  --> DOCX: {rendered_paths['docx']}")
        print(f"  --> SRT:  {rendered_paths['srt']}")
    except Exception as exc:
        print(f"\n[ERROR] Output rendering failed: {exc}", file=sys.stderr)
        checkpoint_mgr.update_status(
            CheckpointStatus.FAILED,
            file_id=last_file_id,
            error_message=f"Rendering failed: {exc}",
        )
        return 1

    # -------------------------------------------------------------
    # Step 5: Cleanup temporary audio blocks & Finalize Checkpoint
    # -------------------------------------------------------------
    print("[5/5] Cleaning cache & saving final checkpoint...")
    try:
        cache_path = orchestrator.cache_dir / orchestrator.job_id
        if cache_path.exists():
            shutil.rmtree(cache_path, ignore_errors=True)
    except Exception:
        pass

    try:
        total_segs = merger.total_segments()
        last_idx = merger.get_last_source_index()
        checkpoint_mgr.update_status(
            CheckpointStatus.COMPLETED,
            transcript_file=str(config.output_transcript_path.name),
            last_confirmed_source_index=last_idx,
            total_source_segments=total_segs,
        )
    except CheckpointError as exc:
        print(f"\n[ERROR] Failed to save final checkpoint: {exc}", file=sys.stderr)
        return 1

    print("\nSTATUS: COMPLETED")
    print(f"Total transcribed segments: {merger.total_segments()}")
    print(f"Outputs saved to: {config.output_dir}")
    print(f"Checkpoint saved: {config.checkpoint_file_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Vietnamese Verbatim Transcription Agent (Milestone 3)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force transcription even if checkpoint shows already completed.",
    )
    args = parser.parse_args()

    exit_code = run_pipeline(force=args.force)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
