# Transcribe temporal order and acoustic coverage endpoint

Baseline: 505 passed in 37.86 seconds using python -B -m pytest -q -p no:cacheprovider.

## Audit and policy

The official Transcribe guide (https://ai.google.dev/gemini-api/docs/transcribe) and
installed SDK WordInfo describe transcript byte positions and per-word relative audio
start/end offsets. Neither gives a global or per-speaker chronological guarantee.
The user supplied sanitized diagnostic pairs spk:2 638.3..638.4 -> spk:0 631..631.2,
and spk:1 96.1..96.3 -> spk:0 95.5..95.7. These demonstrate text-adjacent time reversal,
not proof that the whole response has invalid acoustic bounds. The actual private
response/audio was not available; fixture words are neutral placeholders.

Remove the unsupported global start cursor as a rejection rule. Same-speaker reversal
is also diagnostic rather than an invented per-speaker constraint. Do not sort text or
words by audio time. TRANSCRIBE_TEMPORAL_OVERLAP labels cross-speaker reversal and
TEXT_AUDIO_ORDER_DIFFERENCE for same-speaker reversal. These are ordering diagnostics,
not a claim that adjacent individual word intervals physically overlap.

Still reject invalid duration syntax, negative/nonfinite offsets, start>end, offsets
outside the slice (existing 1e-6 tolerance), malformed types, invalid UTF-8/text indexes,
and broken word/text correspondence. Explicit text byte spans are checked even when
word text is present. Missing end is not invented; missing start retains existing
controlled mapping failure. Response-semantic versus contract-failure taxonomy stays.

## Evidence and endpoint

The adapter computes max(valid word end), then adds physical start once. It emits
TRANSCRIBE_COVERAGE_ENDPOINT and evidence with MAX_WORD_END_OFFSET basis, exact physical
bounds, and payload/segment hashes. If no word end exists, endpoint is unavailable and
the previous segment-start fallback remains. Evidence is only attached after all
annotations and text mapping pass. Each client call clears prior evidence first.

Transcriber accepts this local evidence only if physical bounds and the exact canonical
payload match; it enables text-order timestamps only for that AUDIO response. Coverage
uses the same matched evidence for latest acoustic end. Flash continues to use maximum
segment start as before (the old implementation already used max, not list[-1]). No
coverage threshold, tail-activity test, DABB or recovery budget changes.

## Downstream integration without reordering

The same global-order assumption existed downstream. Narrow provenance-gated exceptions
were required in structural validation, committed-block merge, SRT timing and SRT
read-back validation; otherwise valid native text order would fail after checkpoint.
Committed metadata records provider adapter, temporal basis and segment hash. Merger
checks the hash against canonical segment fields (excluding backend block_id), retains
all physical/ordinal/bounds/conflict checks, and marks only verified native segments.
Renderers allow backwards cue starts only inside the same marked native block. All
other order checks and the existing cue-end policy remain. No sorting, timestamp
repair, speaker renaming, schema redesign, or checkpoint/resume redesign is introduced.
SRT cue numbers follow transcript order; some players may handle nonchronological
cues differently. No claim is made about universal subtitle-player compatibility.

## Callable state

provider_callable records a successful provider response even when subsequent native
semantic mapping fails. proven_callable remains a compatibility alias for callable
history. validated_success is separate and becomes true only after validated commit.
All three survive subsequent state transitions; quarantine/fallback policy is unchanged.

## Tests and production re-test

Tests cover both production offset patterns, same-speaker order differences, exact
max-end/absolute endpoints, invalid bounds/syntax/indexes, stale evidence rejection,
Flash segment-start isolation, semantic-error callable history, and a real orchestration
path using fake SDK/audio through validation, checkpoint, merge and TXT/DOCX/SRT export.
Older backwards-order retry tests now use genuine start-after-end corruption so they
continue testing bounded retry without imposing the removed invariant.

Resume the same source with the normal CLI and existing checkpoint, without force.
Valid reversals should emit temporal diagnostics with no structural retry increase.
Inspect MAX_WORD_END_OFFSET and absolute endpoint; verify physical end/checkpoint still
come from validated physical commits. True malformed annotations must still retry and
fallback; Flash coverage/model fallback remain strict. This change used no live provider
inference and did not modify private audio, production cache or checkpoint.
