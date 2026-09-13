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

from src.source_identity import SourceIdentity, UNITS_PER_SECOND, sha256_file
from src.gemini_client import ModelReplanRequired, GeminiTranscribeError
from src.model_health import ModelHealth
from src.coverage_validator import CoverageValidator, CoverageDecision
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

    # Content identity is authoritative; path/name are diagnostic only.
    try:
        source_identity = SourceIdentity.capture(target_audio.path, total_duration)
        orchestrator.source_identity = source_identity
        print(f"[SOURCE_IDENTITY] mode=AUDIO fingerprint={source_identity.fingerprint} duration={total_duration}")
        existing_data = checkpoint_mgr.load() if not force else checkpoint_mgr.current_data
        has_checkpoint = config.checkpoint_file_path.exists() and not force
        if has_checkpoint:
            checkpoint_mgr.validate_resume(source_identity)
            if any(m.get("coverage", {}).get("decision") not in ("COVERAGE_PASS", "COVERAGE_WARNING")
                   for m in existing_data.block_metrics):
                raise CheckpointError("COVERAGE_UNVERIFIED_CHECKPOINT: historical progress lacks coverage evidence; explicit reprocessing required")
            print(f"[SOURCE_MATCH] fingerprint={source_identity.fingerprint[:12]} version={existing_data.schema_version}")
            print(f"[CHECKPOINT_LOAD] status=VALID last_committed_block={existing_data.current_block_id} last_confirmed_audio_end={existing_data.next_audio_start_us / UNITS_PER_SECOND}")
            if existing_data.next_audio_start_us == source_identity.duration_us:
                # Checkpoint is the durable transcript manifest. Rebuild derived outputs
                # without creating a provider client, even after a render-time crash.
                recovery_renderer = OutputRenderer(strict_speaker_format=config.strict_speaker_format, srt_end_policy=config.srt_end_policy, srt_min_duration=config.srt_min_duration, default_segment_duration_seconds=config.srt_estimated_duration, source_end=total_duration)
                recovery_renderer.provenance = {"source_fingerprint": source_identity.fingerprint, "checkpoint_version": existing_data.schema_version, "committed_block_count": len(existing_data.block_metrics)}
                recovery_renderer.render_all(
                    segments=OutputMerger.from_checkpoint(existing_data, require_complete=True).get_merged_segments(),
                    base_dir=config.output_dir, job_id=existing_data.job_id)
                checkpoint_mgr.update_status(CheckpointStatus.COMPLETED,
                    transcript_file=config.output_transcript_path.name)
                print(f"[RESUME] status=SOURCE_COMPLETE last_confirmed_audio_end={total_duration}")
                return 0
    except (CheckpointError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    is_resuming = has_checkpoint
    merger = OutputMerger(source_mode="AUDIO")
    if is_resuming:
        current_job = existing_data
        next_source_index = existing_data.next_source_index
        resume_from_seconds = existing_data.next_audio_start_us / UNITS_PER_SECOND
        start_from_block = len(existing_data.block_metrics) + 1
        for segment in existing_data.confirmed_segments:
            merger.add_segment(TranscriptSegment.from_dict(segment), segment.get("block_id"))
        # Adaptive hints are disposable and were not versioned/model-scoped.
        # Fresh resume uses the same new-block policy as normal continuation.
        print(f"[RESUME] next_start={resume_from_seconds} next_block={start_from_block} reason=LAST_CONFIRMED_AUDIO_END")
    else:
        next_source_index, resume_from_seconds, start_from_block = 1, 0.0, 1
        if force and config.checkpoint_file_path.exists():
            from uuid import uuid4
            backup = config.checkpoint_file_path.with_name(f"checkpoint.before-force.{uuid4().hex}.json")
            shutil.copy2(config.checkpoint_file_path, backup)
            print(f"[CHECKPOINT_ARCHIVE] path={backup}")
        current_job = checkpoint_mgr.create_new_job(
            file_name=target_audio.file_name, total_source_segments=total_blocks_approx,
            status=CheckpointStatus.RUNNING, source_identity=source_identity)
    current_job.contract_metadata = {"transcription_contract_version": 1, "system_prompt_sha256": sha256_file(config.system_prompt_path), "schema_sha256": sha256_file(config.schema_path) if config.schema_path.exists() else None}
    orchestrator.next_block_number = start_from_block

    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    print(f"\nJob ID:  {current_job.job_id}")
    print(f"Session: {current_job.session_id}")
    print(f"Audio:   {target_audio.file_name} ({target_audio.size_bytes / (1024 * 1024):.2f} MB, {minutes}m {seconds:02d}s)")
    print(f"Blocks:  ~{total_blocks_approx} block(s) est. (DABB adaptive, initial ~{initial_block_dur:.0f}s/block)")
    print(f"Model:   {config.gemini_model}\n")

    try:
        gemini_client = GeminiClient(
            api_key=config.gemini_api_key,
            model_name=config.gemini_model,
            max_retries=config.retry_max_attempts,
            initial_delay_seconds=config.retry_initial_delay_seconds,
            timeout_seconds=config.timeout_seconds,
            fallback_enabled=config.model_fallback_enabled,
            allow_lite_models=config.allow_lite_models,
            fallback_policy=config.provider_fallback_policy,
            max_backoff_seconds=config.max_provider_backoff_seconds,
            retry_max_delay_seconds=config.retry_max_delay_seconds,
            backoff_multiplier=config.retry_backoff_multiplier,
            max_transient_retries=config.max_transient_retries_per_model,
            fallback_models=config.fallback_models,
            requested_max_output_tokens=config.gemini_max_output_tokens,
        )
    except GeminiClientError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    model_health = ModelHealth(config)
    gemini_client.model_health = model_health
    response_parser = ResponseParser(schema_path=config.schema_path)
    transcript_validator = TranscriptValidator()

    transcriber = GeminiTranscriber(
        gemini_client=gemini_client,
        system_prompt_path=config.system_prompt_path,
        validator=transcript_validator,
        response_parser=response_parser,
    )

    renderer = OutputRenderer(strict_speaker_format=config.strict_speaker_format, srt_end_policy=config.srt_end_policy, srt_min_duration=config.srt_min_duration, default_segment_duration_seconds=config.srt_estimated_duration, source_end=total_duration)
    coverage_validator = CoverageValidator(config)

    # -------------------------------------------------------------
    # Step 3: Sequential Block Transcription Loop (DABB-driven)
    # -------------------------------------------------------------
    print(f"[3/5] Transcribing blocks (DABB adaptive, resuming from {next_source_index})...")
    checkpoint_mgr.update_status(CheckpointStatus.PROCESSING)

    block = None
    block_id = None
    last_file_id: str | None = None
    try:
        for adaptive_slice in orchestrator.iter_adaptive_blocks(start_from_seconds=resume_from_seconds):
            source_identity.assert_unchanged()
            block = adaptive_slice.source_block
            block_id = f"BLOCK_{block.source_index_str}"

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
            gemini_file = None

            # Transcribe & Validate with retry on FAIL
            validated = False
            last_val_result = None
            failure_summary = "Validation failed"
            block_result = None
            last_exc: Exception | None = None

            v_attempt, coverage_generation, total_attempts = 1, 1, 0
            size_retry_count, structural_retry_count, network_failure_count = 0, 0, 0
            # Independent budgets; the legacy validation setting supplies the
            # default size allowance, never its consumed counter.
            size_limit = config.size_retry_limit if config.size_retry_limit is not None else max(0, config.validator_max_retries - 1)
            orchestrator.current_block_id = block_id
            def select_model(model, reason):
                target = orchestrator.select_model(model, reason)
                gemini_client.dabb_target_input_tokens = orchestrator.dabb.current_target_tokens
                if adaptive_slice.current_duration_seconds > target + 1e-6:
                    raise ModelReplanRequired(target)
            gemini_client.on_model_selected = select_model
            while True:
                total_attempts += 1
                print(f" Transcribing...", end="", flush=True)
                try:
                    if gemini_file is None:
                        gemini_file = gemini_client.upload_audio(audio_path=block.file_path)
                    else:
                        gemini_file = gemini_client.refresh_audio_upload(gemini_file, block.file_path)
                    raw_name = getattr(gemini_file, "name", None)
                    last_file_id = str(raw_name) if raw_name is not None else str(gemini_file)
                    print(f" [block_id={block_id} attempt={total_attempts} generation={adaptive_slice.generation} "
                          f"validation_attempt={v_attempt} structural_retry_count={structural_retry_count} size_retry_count={size_retry_count} coverage_generation={coverage_generation} physical_generation={adaptive_slice.generation} "
                          f"start={block.start_time_seconds} end={block.end_time_seconds} "
                          f"duration={adaptive_slice.current_duration_seconds} slice={block.file_path} "
                          f"upload={last_file_id} action=TRANSCRIBE]", end="", flush=True)
                    gemini_client.physical_duration = adaptive_slice.current_duration_seconds
                    gemini_client.dabb_target_input_tokens = orchestrator.dabb.current_target_tokens
                    transcriber.speaker_context = list(dict.fromkeys(s.speaker for s in merger.get_merged_segments() if s.speaker))
                    outcome = transcriber.transcribe_block(
                        gemini_file=gemini_file,
                        job_id=current_job.job_id or "JOB_001",
                        session_id=current_job.session_id or "SESSION_001",
                        block_id=block_id,
                        first_source_index=next_source_index,
                        start_offset_seconds=block.start_time_seconds,
                        end_offset_seconds=block.end_time_seconds,
                    )
                    block_result = outcome.block_result
                    last_val_result = outcome.structural_validation
                    fidelity_result = outcome.fidelity_validation
                    last_exc = None
                except ModelReplanRequired as exc:
                    orchestrator.rebuild_audio_block(adaptive_slice, exc.target / adaptive_slice.current_duration_seconds, reason="MODEL_REPLAN")
                    block = adaptive_slice.source_block
                    gemini_file = None
                    continue
                except Exception as exc:
                    if (isinstance(exc, GeminiTranscribeError) or
                            isinstance(exc.__cause__, GeminiClientError) and classify_failure(exc) != FailureType.SIZE_FAILURE):
                        raise  # Client provider budget is already exhausted.
                    metadata = getattr(gemini_client, "last_response_metadata", {})
                    orchestrator.select_model(metadata.get("actual_model", orchestrator.active_model))
                    last_exc = exc
                    print(f" [Error: {exc}]", end="", flush=True)
                    # Classify and potentially shrink block for next attempt
                    failure_type = classify_failure(exc)
                    target_before = orchestrator.dabb.current_target_tokens
                    orchestrator.on_block_failure(failure_type)
                    print(f" [failure_class={failure_type.value} target_before={target_before} "
                          f"target_after={orchestrator.dabb.current_target_tokens}]", end="", flush=True)
                    if failure_type == FailureType.SIZE_FAILURE:
                        details = dict(metadata)
                        print(f" [OUTPUT_LIMIT] block_id={block_id} actual_model={details.get('actual_model')} physical_duration={adaptive_slice.current_duration_seconds} finish_reason={details.get('finish_reason')} prompt_tokens={details.get('input_tokens')} output_tokens={details.get('output_tokens')} max_output_tokens={details.get('configured_max_output_tokens')} dabb_target_input_tokens={target_before} size_retry_count={size_retry_count}", flush=True)
                        if size_retry_count >= size_limit:
                            print(f"[SIZE_RETRY_EXHAUSTED] block_id={block_id} size_retry_count={size_retry_count} size_retry_limit={size_limit} last_duration={adaptive_slice.current_duration_seconds} next_candidate_duration={max(config.min_duration_seconds, min(orchestrator.current_block_duration_seconds, adaptive_slice.current_duration_seconds * config.block_shrink_factor))}", flush=True)
                            raise TranscriptionError(f"SIZE_RETRIES_EXHAUSTED: block={block_id} size_failures={size_retry_count} size_limit={size_limit} total_attempts={total_attempts}: {exc}") from exc
                        orchestrator.rebuild_audio_block(adaptive_slice)
                        size_retry_count += 1
                        block = adaptive_slice.source_block
                        gemini_file = None
                        print(f" [action=SHRINK_REBUILD_REUPLOAD size_retry_count={size_retry_count}]", end="", flush=True)
                        time.sleep(config.retry_initial_delay_seconds)
                        continue
                    if failure_type == FailureType.NETWORK_FAILURE:
                        # Normally exhausted inside GeminiClient. This bounded
                        # path handles transport errors escaping an adapter.
                        network_failure_count += 1
                        if network_failure_count > config.max_transient_retries_per_model:
                            raise TranscriptionError("PROVIDER_RETRIES_EXHAUSTED") from exc
                        time.sleep(config.retry_initial_delay_seconds)
                        continue
                    structural_retry_count += 1
                    if v_attempt >= config.validator_max_retries:
                        raise TranscriptionError(f"VALIDATION_RETRIES_EXHAUSTED: block={block_id} validation_attempt={v_attempt}: {exc}") from exc
                    v_attempt += 1
                    time.sleep(config.retry_initial_delay_seconds)
                    continue

                metadata = getattr(gemini_client, "last_response_metadata", {})
                orchestrator.select_model(metadata.get("actual_model", orchestrator.active_model))
                actual_model = metadata.get("actual_model")
                if not last_val_result.is_valid:
                    structural_retry_count += 1
                if not last_val_result.is_valid and model_health.structural_failure(actual_model, last_val_result, block, block_result):
                    gemini_client.advance_after_output_failure()
                    continue
                if last_val_result.is_valid and fidelity_result.allows_confirmation:
                    # Provider STOP proves normal generation termination, not audio coverage.
                    if block_result.last_source_index != block_result.segments[-1].source_index:
                        print(f" [AUDIO_LAST_SEGMENT_ORDINAL_MISMATCH block_id={block_id} actual_model={metadata.get('actual_model')} received_last_source_index={block_result.last_source_index} final_segment_ordinal={block_result.segments[-1].source_index}]", flush=True)
                    coverage = coverage_validator.validate(block, block_result)
                    print(f" [coverage={coverage.decision.value} audio_end={block.end_time_seconds} "
                          f"last_timestamp={coverage.last_transcript_timestamp} tail_gap={coverage.tail_gap_seconds:.3f} "
                          f"tail_gap_ratio={coverage.tail_gap_ratio} physical_duration={coverage.physical_duration_seconds} tail_activity={coverage.tail_activity} active_seconds={coverage.active_tail_seconds}]", end="", flush=True)
                    if not coverage.allows_confirmation:
                        failure_summary = f"COVERAGE_FAILURE COVERAGE_UNRESOLVED: {coverage.to_dict()}"
                        if model_health.coverage_failure(actual_model, coverage):
                            gemini_client.advance_after_output_failure()
                            continue
                        if coverage.decision == CoverageDecision.RETRY:
                            orchestrator.on_coverage_failure(adaptive_slice.current_duration_seconds)
                        if coverage.decision == CoverageDecision.RETRY and coverage_generation < config.coverage_max_generations:
                            try:
                                orchestrator.rebuild_audio_block(adaptive_slice, config.coverage_shrink_factor, reason="COVERAGE_FAIL")
                            except BlockBuilderError as exc:
                                raise TranscriptionError(f"{failure_summary}; coverage rebuild exhausted: {exc}") from exc
                            block = adaptive_slice.source_block
                            gemini_file = None
                            print(" [COVERAGE_FAILURE action=RETRY_SMALLER_BLOCK]", end="", flush=True)
                            coverage_generation += 1
                            continue
                        break
                    renderer.validate_output_fields(block_result.segments)
                    validated = True
                    warn_tag = f" ({len(last_val_result.warnings)} warn)" if last_val_result.warnings else ""
                    orchestrator.on_coverage_success(adaptive_slice.current_duration_seconds, coverage)
                    next_dur = orchestrator.next_target_duration_seconds
                    acceptance = "COVERAGE_WARNING" if coverage.decision == CoverageDecision.WARNING else "PASS"
                    print(f" [{acceptance}{warn_tag}] Done. (next block ~{next_dur:.0f}s)")
                    if fidelity_result.decision == FidelityDecision.ACCEPT_WITH_WARNING:
                        print(f"  [ACCEPT_WITH_WARNING / NON-BLOCKING] {fidelity_result.summary()}")
                    break
                else:
                    # Classify validation failure for DABB feedback
                    failure_summary = (
                        f"Attempt {v_attempt}/{config.validator_max_retries}; Structural: {last_val_result.summary()}; "
                        f"Fidelity: {fidelity_result.summary()}"
                    )
                    failure_type = (
                        FailureType.FIDELITY_FAILURE
                        if not fidelity_result.allows_confirmation
                        else (FailureType.STRUCTURAL_FAILURE if last_val_result.reason_codes
                              else classify_failure(last_val_result.summary(), last_val_result))
                    )
                    orchestrator.on_block_failure(failure_type)
                    print(f" [{failure_summary}]", end="", flush=True)
                    if fidelity_result.decision == FidelityDecision.FAIL:
                        failure_summary = "DETERMINISTIC_FIDELITY_FAILURE: " + failure_summary
                        break
                    if v_attempt < config.validator_max_retries:
                        print(f" Retrying validation attempt {v_attempt+1}...", end="", flush=True)
                        time.sleep(config.retry_initial_delay_seconds)
                        v_attempt += 1
                    else:
                        print(f" Max validation retries reached.")
                        if fidelity_result.decision == FidelityDecision.RETRY_REVIEW:
                            failure_summary = "BLOCK_REVIEW_REQUIRED: " + failure_summary
                        break

            if not validated or block_result is None:
                # Do NOT commit checkpoint on FAIL
                fail_summary = failure_summary
                raise TranscriptionError(f"Block {block.source_index_str} failed validation: {fail_summary}")

            # Feed success back to DABB (actual token usage from response metadata)
            provider_metadata = getattr(gemini_client, "last_response_metadata", {})
            actual_in = provider_metadata.get("input_tokens") or adaptive_slice.estimated_input_tokens
            actual_out = provider_metadata.get("output_tokens") or 0
            orchestrator.on_block_success(
                actual_input_tokens=actual_in,
                actual_output_tokens=actual_out,
            )

            # Merge block segments idempotently
            # Do not merge an attempt before durable commit.
            # Keep declared last_source_index untouched in the response/diagnostics.
            # Compatibility counters describe stored segments; audio progress uses boundaries.
            final_segment_ordinal = block_result.segments[-1].source_index
            next_source_index = final_segment_ordinal + 1

            # Atomic checkpoint contains both transcript payload and physical progress.
            # Rendered files are rebuildable derivatives, never a resume authority.
            # Atomic Checkpoint Commit ONLY after validation & merge PASS
            # Persist quality signals in existing metadata using the same atomic commit.
            metric = {
                "contract_metadata": dict(current_job.contract_metadata),
                "source_mode": "AUDIO",
                "segment_count": len(block_result.segments),
                "received_last_source_index": block_result.last_source_index,
                "final_segment_ordinal": final_segment_ordinal,
                "structural_warnings": last_val_result.warnings,
                "endpoint_diagnostics": ([dict(reason_code="AUDIO_LAST_SEGMENT_ORDINAL_MISMATCH",
                    received_last_source_index=block_result.last_source_index,
                    final_segment_ordinal=final_segment_ordinal, block_id=block_id,
                    actual_model=provider_metadata.get("actual_model"))]
                    if block_result.last_source_index != final_segment_ordinal else []),
                "coverage": coverage.to_dict(),
                "block_id": block_id,
                "actual_start_offset": block.start_time_seconds,
                "actual_end_offset": block.end_time_seconds,
                "attempt_count": total_attempts,
                "provider_attempt_count": getattr(gemini_client, "_provider_attempt", 0),
                "validation_attempt_count": v_attempt,
                "structural_retry_count": structural_retry_count,
                "size_retry_count": size_retry_count,
                "coverage_generation_count": coverage_generation,
                "physical_generation_count": adaptive_slice.generation,
                "validation_attempt": v_attempt,
                "coverage_generation": coverage_generation,
                "physical_generation": adaptive_slice.generation,
                "generation": adaptive_slice.generation,
                "final_duration": adaptive_slice.current_duration_seconds,
                "provider_metadata": provider_metadata,
                "fidelity_status": fidelity_result.status.value,
                "fidelity_decision": fidelity_result.decision.value,
                "requires_quality_review": fidelity_result.needs_audio_review,
                "unknown_token_count": fidelity_result.unknown_token_count,
                "issues": [
                    {"source_index": i.source_index, "reason": i.indicator, "detail": i.detail, "evidence": i.evidence}
                    for i in fidelity_result.issues
                ],
            }
            source_identity.assert_unchanged()
            checkpoint_mgr.commit_block(
                block_id=block_id,
                last_source_index=final_segment_ordinal,
                new_segments=[s.to_dict() for s in block_result.segments],
                metric=metric, adaptive_state={},
                file_id=last_file_id,
            )
            merger.merge_block_result(block_result)  # Committed speaker context only.
            model_health.success(actual_model, adaptive_slice.current_duration_seconds)

    except CheckpointError as exc:
        print(f"[ERROR] Checkpoint commit failed; last persisted state retained: {exc}", file=sys.stderr)
        orchestrator.cleanup_slices()
        return 1
    except (GeminiClientError, TranscriptionError, BlockBuilderError, OSError, ValueError) as exc:
        print(f"[BLOCK_NOT_COMMITTED] block_id={block_id} start={getattr(block, 'start_time_seconds', None)} attempted_end={getattr(block, 'end_time_seconds', None)} checkpoint_end={checkpoint_mgr.current_data.next_audio_start_us / UNITS_PER_SECOND} reason={type(exc).__name__}")
        print(f"\n[ERROR] Pipeline failed during block transcription: {exc}", file=sys.stderr)
        checkpoint_mgr.update_status(
            CheckpointStatus.FAILED,
            file_id=last_file_id,
            error_message=str(exc),
        )
        orchestrator.cleanup_slices()
        return 1

    # -------------------------------------------------------------
    # Step 4: Render outputs (TXT, DOCX, SRT)
    # -------------------------------------------------------------
    print("[4/5] Rendering outputs (transcript.txt, transcript.docx, subtitle.srt)...")
    try:
        committed = checkpoint_mgr.load()
        merger = OutputMerger.from_checkpoint(committed, require_complete=True)
        renderer.provenance = {"source_fingerprint": source_identity.fingerprint, "checkpoint_version": committed.schema_version, "committed_block_count": len(committed.block_metrics)}
        merged_segments = merger.get_merged_segments()
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
