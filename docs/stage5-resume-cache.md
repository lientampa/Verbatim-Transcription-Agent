# Stage 5: exact resume and cache safety

## Startup and durable state

`main.run_pipeline` inspects audio with `DabbAudioOrchestrator.get_duration`, captures `SourceIdentity`, loads `CheckpointManager`, validates the checkpoint against the source, restores committed segments into `OutputMerger`, and invokes the ordinary adaptive block iterator at `next_audio_start_us / 1_000_000`. Source inspection and checkpoint verification precede Gemini construction/upload. Job/session names are operational metadata.

The authoritative AUDIO identity is `(source SHA-256, accepted start_us, accepted end_us, logical block_id)`. The source SHA-256 streams the complete file once per startup. Size and duration are checked too; path and mtime are diagnostic/supporting metadata. Renaming unchanged bytes preserves source identity. Replacing bytes with the same name, size and mtime is detected at the next startup. During a job, inexpensive stat checks detect ordinary modifications; deliberately changing bytes while restoring all metadata during an active job is outside this single-source snapshot guarantee.

Physical times are rounded to integer microseconds. Adjacent ranges must match exactly in that representation. Duration comparison permits one microsecond of probe/serialization variation; this is not a gap tolerance. Neither model timestamps nor model ordinals can select an AUDIO cursor.

## Checkpoint schema and compatibility

Schema v2 contains source identity, `next_audio_start_us`, committed block metrics and their transcript payloads in `confirmed_segments`. New records also explicitly identify AUDIO and store prompt/schema hashes as contract metadata. Each new block records its contract metadata, actual model provenance and physical accepted range. Existing v2 records without the optional mode/contract fields remain readable only after their physical ranges, identity, counts, order and EOF are verified; production startup additionally requires historical coverage evidence. Older v2 contiguous-ordinal payloads remain supported because their physical boundaries already exist. No ordinal-to-time conversion occurs.

Schema v1 remains available to the legacy TEXT checkpoint API, where SOURCE_INDEX is authoritative. It cannot resume AUDIO. Unknown versions, contradictory modes, gaps, overlaps, out-of-bounds cursors and ambiguous legacy AUDIO progress are rejected without modifying the checkpoint. The existing real checkpoint was audited read-only: v2, five blocks, cursor 3,590,634,667 microseconds, coverage evidence present.

## One durable manifest, derived renderings

Validation and commit eligibility precede `commit_block`. The checkpoint contains BOTH transcript payload and physical progress in one durable manifest. `save` writes a temporary JSON file, flushes, fsyncs and atomically replaces the checkpoint. There is no separately authoritative block output to reconcile. TXT/DOCX/SRT are rebuildable derivatives. A crash before replacement leaves the prior checkpoint; a crash after replacement resumes from the newly committed end. Failure during rendering cannot lose the committed payload. EOF restart reconstructs outputs from that payload without constructing a Gemini client, including when the last status was PROCESSING.

Repeated commit of the same physical/logical block with the same payload is a no-op (`IDEMPOTENT_ALREADY_COMMITTED`). A conflicting range or payload for an already committed logical block is rejected. Different model provenance does not create a new physical progress identity. Failures may update diagnostic status but never accepted boundaries.

`--force` intentionally restarts from zero and first copies the prior checkpoint to a uniquely named sibling archive. It does not delete shared slices. Normal resume does not use old unversioned adaptive hints; new-block sizing uses the existing policy, including MAX-FIRST and normal EOF/config caps. Hints never determine the cursor.

## Cache inventory

- Physical slices: `AudioBlockBuilder.slice_time_range`; SHA-256 key over source fingerprint, start/end in microseconds, extraction format/profile version and extension. Sidecar identity and artifact SHA-256 must match before a hit. Logical block id is intentionally excluded from the reusable media cache: the same source/range can supply different operational attempts. Changed source content creates a distinct key. Incomplete media/sidecar pairs rebuild. Old entries are not deleted.
- Upload references: only in-memory reuse within the current physical block. A rebuild resets the reference; restart uploads again. Checkpoint `file_id` is diagnostic and is never restored as an upload cache. Before orchestration reuses a reference, it checks expiry/provider existence; expired, missing (404) or failed files are re-uploaded from the same slice path. Other lookup errors fail without advancing progress.
- Parsed/model responses: no persistent generation cache. There is no stale response bypass into validation. Committed payloads are durable accepted work, not unvalidated response-cache hits. Prompt/schema hashes are provenance; changing a prompt does not silently re-transcribe committed work. Use explicit `--force` to reprocess it.
- Validation cache: none. Render cache: none; rendered files are derivatives. The transcriber's prompt cache is local to one instance/job.

## Verification and limits

Tests cover replacement with preserved metadata, renamed sources, wrong-source checkpoint rejection before provider work, interrupted MAX-FIRST/shrink runs, duplicate commits, corrupt boundaries/versions/modes, partial temp writes/fsync/replace failures, EOF recovery, TEXT indices, source-aware slice keys and expired/missing uploads. Existing validation and provider policy tests remain intact.

No paid live transcription is used. One writer per checkpoint is assumed; multi-process locking and cache garbage collection are not introduced. Atomic replace/fsync relies on the filesystem and storage honoring their durability guarantees. Source hashing is once per startup, not once per block. Stage 6 is not implemented.
