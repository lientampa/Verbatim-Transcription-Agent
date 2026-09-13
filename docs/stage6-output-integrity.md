# Stage 6: merger and renderer integrity

## Audited previous behavior

OutputMerger keyed AUDIO segments by `(block_id, ordinal)` in insertion order, overwriting conflicts; the pipeline merged accepted in-memory attempts before durable commit. Final outputs used that in-memory list. TXT had blank separators; DOCX used identical field formatting but no read-back parity. SRT used absolute starts where present, synthetic fixed starts otherwise, and a four-second maximum estimated duration. Its strict-mode text used the same bracketed timestamp/speaker line as TXT, an explicit earlier product convention.

## Authoritative reconstruction

Final production rendering now reloads the Stage 5 durable checkpoint. `OutputMerger.from_checkpoint` groups only its committed payloads, then checks accepted fidelity/coverage, canonical physical ranges, logical/physical compatibility and segment order. Ordering uses physical starts; identical canonical duplicates deduplicate, conflicting payloads or competing physical generations fail. Gaps and overlaps fail. Equal segment timestamps preserve original order. Transcript start timestamps do not determine physical continuity. Timestamp ranges retain the existing upstream one-second tolerance and enforce source EOF; global reversal is rejected, never repaired.

The canonical identity is the checkpoint source fingerprint plus accepted start/end in integer microseconds plus logical block id. Merger complexity is O(n log n) in records/segments, with no fuzzy text matching. The old TEXT_TIMESTAMP API remains ordered by SOURCE_INDEX. Accepted attempts enter the in-memory speaker context only after commit; final rendering never trusts that context as its payload authority.

## Render contract v2

Production strict-mode TXT has one exact `[hh:mm:ss - Speaker]: text` line per segment. DOCX has a title plus one matching paragraph per segment. No spelling, punctuation, fillers, repetitions or uncertainty markers are rewritten. Internal metadata never participates in field rendering. Literal technical words spoken in the text are preserved; technical speaker labels are rejected. Embedded newlines are rejected in strict mode rather than silently split or normalized. The existing explicitly disabled strict-format compatibility mode remains available for legacy callers.

All three outputs consume the same canonical list. RendererValidator reads staged TXT, DOCX and SRT back and checks complete content/order/count parity. SRT numbering and serialized positive/monotonic timings are checked independently. Renderers do not mutate canonical objects. TXT and SRT are byte-stable; DOCX is content-stable.

## SRT timing

The schema has no observed segment end. Every generated SRT end is rendering metadata, recorded as estimated in render_manifest.json. `SRT_END_POLICY=NEXT_START` is the default: use the next absolute segment start, applying `SRT_MIN_DURATION=0.1` seconds to equal/near-equal starts. Equal starts may produce overlapping cues without changing either source timestamp. The final cue ends at min(source end, start + SRT_ESTIMATED_DURATION), default four seconds. No hidden long-gap cap is introduced. ESTIMATED uses the configured estimated duration for each cue. MODEL_END is rejected because the schema cannot support it.

A cue with no positive millisecond duration inside source bounds fails. Production never synthesizes missing starts. Milliseconds use Python round(seconds*1000), including ties-to-even, followed by integer division so carry and hours beyond one hour remain correct. Strict-mode SRT retains the earlier product requirement of the bracketed transcript line inside each cue; this deliberate exception to the stage's preferred `Speaker: text` convention is tested through parity.

## Output set publication

Render into a temporary sibling directory, read back and validate all formats, then publish. A small render manifest records source fingerprint, checkpoint version, committed block count, segment count, render contract version and timing policy. It contains no transcript text.

Publication keeps fsynced old files and a pending journal, replaces the validated outputs, then removes the pending marker. Ordinary replacement failure rolls back; an interrupted publication is recovered before the next render. One writer per output directory is assumed. The filesystem cannot atomically replace four independent paths in one operation: concurrent readers may observe the short publication interval; after a handled failure rollback restores the previous set, and abrupt process loss is reconciled at the next invocation. Storage/permission failures that prevent rollback leave the journal for recovery rather than claim success.

Completed-source restarts and resumed completion rebuild from the checkpoint, not append to old outputs. Incomplete AUDIO cannot be presented as complete. Stage 5 progress/schema, DABB, coverage/fidelity, retry/model policy and prompts are unchanged. Tests establish output integrity only, not acoustic fidelity. Stage 7 is not implemented.
