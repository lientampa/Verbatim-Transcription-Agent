# Runtime model eligibility hotfix

Preferred order remains 2.5-flash, 3.5-transcribe, 3.6-flash, 3.5-flash,
3.8-flash (all gemini- prefixes). User configuration is not rewritten.
Discovery, preference, and runtime eligibility are separate. Metadata get calls
and local SDK contract validation precede audio generation. Metadata success
does not guarantee generation permission or quota; the first real call remains
authoritative. Transient metadata errors leave access explicitly unverified.

404/403 quarantine a model as UNAVAILABLE for this client/job. Unsupported SDK
contracts and adapter failures quarantine as ADAPTER_UNAVAILABLE. A new job
re-evaluates these decisions. 429 is QUOTA_LIMITED; 503 is TRANSIENT_FAILURE,
using existing bounded retry/backoff before exclusion. Neither is a permanent
adapter quarantine. Existing per-model health opens CIRCUIT_OPEN independently.
Selection skips blocked states; exhaustion and JOB_MODEL_SUMMARY retain causes
for every preferred model, plus actual used and quarantined models.

## Transcribe evidence and repair

The production log confirms TRANSCRIBE_ANNOTATION_INVALID but contains no failed
word annotation. Its exact failing predicate cannot be established retrospectively.
The old compound predicate incorrectly required speaker, text, and end_offset.
Installed SDK WordInfo permits these to be absent. The adapter now accepts optional
speaker/end_offset and can recover text from validated UTF-8 start_index/end_index.
Observed start_offset remains mandatory. Invalid types, nonmonotonic timestamps,
out-of-bounds offsets, and incomplete text coverage still fail closed.

Native Interactions requests use verbatim transcription, speaker diarization and
word timestamps only. No Flash thinking/tools/output-token options are sent.
Contract reference: https://ai.google.dev/gemini-api/docs/transcribe
Local SDK GenerationConfig validation checks payload roundtrip without an API call.
Errors identify REQUEST_BUILD, PROVIDER_REQUEST, PROVIDER_RESPONSE_PARSE,
ANNOTATION_PARSE, or CANONICAL_MAPPING with conditions and no transcript text.

SDK word offsets are relative to the uploaded audio. The adapter adds physical
block start once, explicitly using BLOCK_RELATIVE; it never infers timestamp basis
from values. Unknown speaker labels map to anonymous canonical roles; native IDs
never leak into output. Cross-block identity is not provable from request-local IDs;
the adapter does not invent names or merge speakers across requests without evidence.

## Verification and production re-test

Tests use fake SDK/audio only. They cover quarantine/fresh jobs, local preflight,
optional annotations, UTF-8 spans, timestamp offsets, private diagnostics, five-model
exhaustion, coverage rejection, unchanged physical assets across fallback, and
checkpoint retention at 120 seconds after a later block fails. The orchestration
fixture is a synthetic failure-class reproduction, not a byte-for-byte replay of
the production response. Original production annotations are unavailable.

Resume the same source with the existing normal CLI and checkpoint, without force.
Inspect MODEL_PREFLIGHT and RUNTIME_MODEL_CHAIN before generation. Metadata may
still permit 2.5 until its first runtime 404, after which it must not be retried in
that job. If Transcribe fails, retain TRANSCRIBE_ADAPTER_ERROR stage/reason/detail.
Check fallback reuses the same physical slice and upload until a size/coverage
rebuild; verify failed blocks never advance CHECKPOINT_COMMIT. If every model is
excluded, verify all causes and JOB_MODEL_SUMMARY appear. Provider availability,
quota and acoustic fidelity still require a real production re-test. No Stage 7B
or live generation was performed by this hotfix.
