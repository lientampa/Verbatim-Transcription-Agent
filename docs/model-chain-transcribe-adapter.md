# Model chain and transcription adapter

## Production selection

The fixed preference is gemini-2.5-flash -> gemini-3.5-transcribe ->
gemini-3.6-flash -> gemini-3.5-flash -> gemini-3.8-flash -> NO_ELIGIBLE_MODEL.
GEMINI_MODEL defaults to gemini-2.5-flash; FALLBACK_MODELS documents the same order.
The fixed product policy is authoritative for unlocked jobs, including when a stale
GEMINI_MODEL previously requested a newer model. Existing explicit MODEL_LOCK remains
an opt-in override. The client advances forward within a job and never restarts at the
primary during a coverage/size retry. A new client/job starts at the first eligible model.
No age/intelligence sorting and no Lite bypass.

Effective order is intersected with account Models.list discovery. Flash candidates
must advertise generateContent. The exact Transcribe model has a separately documented
Interactions contract and requires a callable SDK interactions.create endpoint; generic
Models.list action strings do not decide its request shape. Runtime authorization,
unsupported endpoint/capability and bounded provider errors follow the existing fallback
policy. Listing a model is not proof that all account runtime quotas/permissions will pass.
Startup prints primary, preferred_chain, effective_chain and lite_allowed=false.
No live account discovery or transcription was performed for this change.

## Adapter boundary

GeminiGenerateContentAdapter uses models.generate_content. TranscriptionModelAdapter
uses interactions.create with uploaded audio URI/MIME and only transcription_config:
verbatim mode, speaker diarization, word timestamps. It sends no system prompt, thinking,
AFC, functions, tools, code execution or generic output-token cap. Smart mode is never used.
Official contract checked against installed SDK types and:
https://ai.google.dev/gemini-api/docs/transcribe
https://ai.google.dev/gemini-api/docs/models/gemini-3.5-transcribe

Native completed model_output text content must have complete word_info annotations.
Whitespace-insensitive annotation text must cover the full content text. Optional speaker
and end offsets may be absent; missing word text requires valid UTF-8 byte spans.
Missing start offsets, invalid/unsorted/out-of-range offsets and unknown completion states
fail closed. Incomplete/budget-exceeded interactions remain OUTPUT_TRUNCATED size errors.
Native HTTP status_code and RetryInfo are normalized so 429/503 use the existing policy.
Usage maps total_input/output/thought tokens into existing diagnostics.

See runtime-model-eligibility-hotfix.md for job-local quarantine and preflight behavior.

Canonical schema 1.0 identity is copied from locally constructed block context; ordinals
are assigned in output order. Relative word start offsets add the physical block start,
then render as absolute HH:MM:SS at the project's existing whole-second precision.
Adjacent same-speaker words are grouped into spans shorter than 15 seconds to retain
near-tail start evidence. Word strings retain repetitions/fillers/false starts. Native
end offsets validate evidence only: they do not replace block end, fingerprint or SRT rules.
Only schema fields leave the adapter. Existing schema/structural/fidelity/coverage validators
still decide whether anything can be committed.

## Speaker limitations

Native spk_N IDs are request-local. The adapter maps them to Người nói N, reserving numbers
already in confirmed speaker context. It does not invent real names or assume spk_1 across
two blocks is the same voice. Mapping is consistent within each response, but cross-block
speaker reconciliation requires acoustic evidence and human review; it is not solved by
renaming provider IDs. This can fragment a person's anonymous labels across blocks. No
provider-specific speaker or timestamp fields reach TXT/DOCX/SRT, and renderer policy is
unchanged.

## Capabilities and retries

Profiles specify adapter, thinking support/levels/preference, function calling and native
structured-output support. Transcribe returns annotations rather than the project's JSON,
so canonical JSON is generated at its adapter boundary. Gemini 2.5 Flash uses supported
thinking_budget=0 (no Gemini-3 thinking_level). Gemini 3.6/3.5 Flash default minimal;
Gemini 3.8 Flash maps minimal to low. Gemini 3 never receives numeric thinking budgets.
https://ai.google.dev/gemini-api/docs/generate-content/thinking

Actual model/profile/adapter/config update on fallback. Flash discovered output caps and
generation-config retry remain unchanged. DABB keeps separate model profiles, output
trackers, learned durations and failure statistics. Provider fallback alone keeps the
same asset; a coverage failure first retries the same actual model with physical rebuild.
The prior MAX_TOKENS/config-recovery/emergency-floor hotfix is preserved. Failed native
outputs do not enter checkpoints or merger. MAX_FIRST, SPEED_FIRST, coverage thresholds,
Stage 5/6 and Stage 7A evaluator semantics are unchanged.

## Verification

Baseline before this task: 424 passed. New adapter tests use fake SDK endpoints, synthetic
native responses and temporary fake audio. They exercise default/ordered selection, all
fallbacks, no unsupported native config, canonical schema, absolute offsets, bad annotations,
HTTP RetryInfo, Lite exclusion, independent adaptive profiles and real orchestration through
coverage recovery and final output. No paid API calls or real audio/checkpoint changes.

```powershell
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_transcribe_adapter.py
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
git diff --check
```

Before a live production run, inspect effective_chain in MODEL_POLICY and actual_model /
provider_adapter in call diagnostics. Transcribe may be absent if account discovery or SDK
endpoint availability does not support it. Its word timestamps/diarization can affect
acoustic quality and limit native requests to 30 minutes; endpoint rejection follows the
existing failure policy. Review real output before claiming acoustic qualification.
