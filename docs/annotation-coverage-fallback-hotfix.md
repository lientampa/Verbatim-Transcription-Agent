# Annotation classification and coverage fallback hotfix

Baseline: 499 passed in 29.62 seconds (python -B -m pytest -q -p no:cacheprovider).

## Annotation taxonomy

Previously every ProviderAdapterError entered provider fallback and job quarantine,
including an otherwise interpretable response with NON_MONOTONIC_START. New
AnnotationSemanticError remains a ProviderAdapterError subtype for compatibility,
but escapes the provider retry loop into the existing model-specific structural
budget. A bad response retries the same model and physical upload. Exhaustion records
ANNOTATION_SEMANTIC_RETRY_EXHAUSTED, excludes it for that block recovery episode, and
advances forward. On a fresh block it may re-enter subject to existing health and
quarantine gates; proven_callable is retained. Unsupported contract/timestamp formats,
missing required timing and genuine SDK/request failures retain adapter policy.
Interpretable bounds, ordering, malformed individual fields and text correspondence
failures are semantic errors. No checkpoint is committed from these responses.

NON_MONOTONIC_START emits bounded timing/speaker metadata including previous/current
start/end and delta, without transcript text. No annotation sorting or repair occurs.

## Overlap audit

Official documentation https://ai.google.dev/gemini-api/docs/transcribe and installed
SDK WordInfo describe per-word offsets and speaker attribution, but do not explicitly
specify overlap/order guarantees. Do not claim a verified provider guarantee from
that absence. Existing adapter policy allows interval overlap (including nested ends)
when starts remain nondecreasing and individual bounds are valid. It does not require
previous_end <= current_start or monotonically increasing ends. Backwards starts still
fail, even for another speaker, because automatic reordering is not authorized and
canonical transcript order must remain intact. Real word-170 offset values were not
provided; no claim is made about whether that specific response reflected speech
ordering ambiguity or corruption. Bounded retry avoids treating it as adapter failure.

## Missing 120-second attempt

The old loop counted physical coverage generations globally. Default limit 4 was
reached after 1200, 600, 300, 150. on_coverage_failure printed the next target 120,
then the guard coverage_generation < coverage_max_generations failed and broke out.
This was coverage budget exhaustion, not size, structural, minimum or provider budget.
The observed ratios did not reach the existing .75 health strike threshold.

The coverage budget is now tracked per actual model. The physical coverage_generation
still increments only on a rebuild. At budget/floor/circuit exhaustion, the loop logs
COVERAGE_RETRY_EXHAUSTED with model, counters, limit and pending target, then tries the
next model on the same slice/upload. MODEL_COVERAGE_RETRY_EXHAUSTED persists per model
in exhaustion diagnostics. At the minimum duration no no-op rebuild occurs. Global
validation/size/provider counters do not decide this recovery. Existing high-ratio
health strikes remain unchanged; lower-ratio repeated failures are explicitly blocked
when their bounded model coverage budget runs out. Fresh block MAX_FIRST is unchanged.

## Verification and re-test

New synthetic tests cover annotation retry/success/exhaustion/fresh re-entry, preserved
success history, interval overlap versus backwards starts, exact production coverage
sequence, actual 120-second call with budget 5, fallback at 150 with budget 4, same
physical identity/generations, checkpoint commit at 150 and next start 150, and all
remaining models failing. Existing true adapter failure tests still require quarantine.
No real audio or private transcript fixture was added; no live inference was run.

Resume the source with the normal CLI and existing checkpoint, without force. Observe
TRANSCRIBE_ANNOTATION_FAILURE and bounded same-model retries. If coverage reaches the
limit, expect COVERAGE_RETRY_EXHAUSTED followed by COVERAGE_MODEL_FALLBACK with unchanged
start/end/upload/generation. With budget 5 the 150-to-120 retry should execute. Keep
coverage thresholds and preferred order unchanged. Failed output must never advance
checkpoint. Save bounded diagnostic lines if production behavior differs.
