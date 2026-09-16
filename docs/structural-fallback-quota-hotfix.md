# Structural fallback and transient quota recovery

Baseline: 492 passed in 33.04 seconds with python -B -m pytest -q -p no:cacheprovider.

## Root cause and recovery

The parse-exception path raised VALIDATION_RETRIES_EXHAUSTED, and the validator path
broke out at the global validation limit. Neither tried the remaining runtime models.
Structural failures now have a per-actual-model counter within each physical block
recovery episode. Each model receives validator_max_retries opportunities (default 3).
The existing severe timestamp circuit can terminate that model earlier. Deterministic
fidelity FAIL remains immediately blocking. Timestamp validation is not repaired,
sorted, shifted or relaxed.

MODEL_STRUCTURAL_RETRY_EXHAUSTED blocks the exhausted model in the job registry and
advances forward through the preferred runtime chain. The next model uses the same
slice, upload, start/end, physical generation, coverage generation and size counter.
Model-profile replanning cannot shrink that structural fallback. Independent genuine
size or coverage failures still use their existing rebuild policy.

JSON/schema exceptions increment structural_generation_failures. Validator structural
failures retain their existing structural counter; responses containing timestamp
order/range failures now also increment timestamp_semantic_failures once per response,
regardless of the number of bad segments. Existing severe-hour circuit thresholds
are retained. Exhaustion preserves per-model reasons and structural error context.

## Quota lifetime

Previously 429 set blocked=true in the runtime registry and provider_retry_exhausted
in ModelHealth, effectively excluding the model for the entire job despite
quarantined=false. Now 429 records a monotonic blocked_until deadline and does not
set permanent provider exhaustion. Provider retry duration is honored; absent or
nonpositive duration uses QUOTA_COOLDOWN_SECONDS (default 30, configurable 1..300).
Long provider delays are recorded rather than slept under SPEED_FIRST fallback.

Only the start of a new block checks expiry and clears quota exclusion. It may select
an earlier recovered preferred model, but never a quarantined one. The same block ID
cannot trigger re-entry, even after time advances. Forward-only fallback remains in
force throughout retries and physical rebuilds of that block. A proven_callable bit
survives quota failure and recovery. Existing provider 503 exhaustion policy remains.

## Tests and production re-test

Synthetic SDK tests run real orchestration/validators with seeded checkpoint 3900 and
slice 3900..5100. They exercise malformed JSON, 21 out-of-range timestamps, then an
order/range failure; next-model success; two exhausted models; full exhaustion; one
malformed response followed by same-model success; cooldown with/without RetryInfo;
and unavailable-model exclusion. No private audio or live provider call is used.

Resume the same source/checkpoint with the normal CLI, without force. Verify
MODEL_STRUCTURAL_FAILURE shows bounded per-model attempts and MODEL_HEALTH_SWITCH
shows MODEL_STRUCTURAL_RETRY_EXHAUSTED. Compare slice/upload and generation counters
before and after the switch. Checkpoint must remain 3900 until valid commit. Quota
state should include blocked_until and quarantined=false; future fresh blocks may
show QUOTA_COOLDOWN_EXPIRED. Provider availability/quota and final acoustic fidelity
still require production observation. DABB, coverage, schema, speaker policy,
checkpoint/resume, merger/renderers and evaluator implementations are unchanged.
