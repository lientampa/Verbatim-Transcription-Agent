# CODEX TRANSIENT CAPABILITY & FALLBACK REPORT

## 1. Executive Summary

Added explicit capability states and one bounded on-demand tiny re-probe for transient-unverified fallback paths. No live provider calls or Candidate rerun in this stage.

## 2. Root Cause of Latest Candidate Failure

The user reports transcription 429 with Retry-After 37s followed by fallback exhaustion. The newest runtime on disk remains fbb66207d79144eaae06acb2675220b0 (the earlier incomplete run); the newer 429 run is not independently available here. Code inspection confirms the reported failure mechanism: initial transient probe results were cached for the entire job with no re-verification route. The reported sequence is reproduced by mocked real-client regression tests, not claimed as a newly observed live trace.

## 3. Current Capability State Semantics

Previously all cached results were returned unchanged, whether verified incompatibility or transient inability to determine capabilities. Unknown failed closed correctly, but could never be re-verified in that job.

## 4. Problem With Transient Probe Results

A single preflight 503 could exclude a fallback for the whole job even after a verified primary became quota limited. This was a recovery limitation, not proof the fallback lacked capability.

## 5. New/Corrected Capability States

VERIFIED_ELIGIBLE, VERIFIED_INELIGIBLE, UNVERIFIED_TRANSIENT; additionally UNVERIFIED_UNKNOWN for failures without sufficient deterministic or transient evidence. Eligibility still requires the full existing contract. Unknown never permits transcription.

## 6. Deterministic Ineligibility Rules

Endpoint 404, local instruction-unsupported adapter, no-lite rejection and existing explicit incompatibility reasons remain excluded. No deterministic negative is re-probed by this mechanism. No model names/order changed.

## 7. Transient-Unverified Rules

408/429/500/502/503/504 and recognized timeout/network transport exceptions remain UNVERIFIED_TRANSIENT. Upload transport errors are kept separate from generation endpoint evidence. Unclassified errors remain UNVERIFIED_UNKNOWN.

## 8. On-Demand Re-Probe Behavior

Only when the existing policy would leave the current model, scan remaining candidates in existing order. A permitted transient-unverified candidate gets a tiny re-probe; only successful full-contract evidence permits transcription. Another transient result continues to the next candidate. Output-failure fallback uses the same eligibility check; validator/DABB decisions themselves are unchanged. Reading runtime_eligible_chain never triggers a re-probe.

## 9. Capability Probe Budget

Per effective path per job: maximum two probes total (initial plus one on-demand re-probe). Existing tiny synthetic audio, nonce, schema, timeout, disabled AFC and single SDK attempt retained. No extra sleep and no infinite probing. Failed uploads are included in this bounded probe-call accounting.

## 10. Transcription Provider Retry Budget

Existing retry_decision policy and transcription attempt accounting unchanged. CapabilityGate owns separate probe_counts and does not increment _provider_attempt.

## 11. Budget Isolation Proof

Regression executes the real GeminiClient routing against a mock SDK, distinguishes tiny probe assets from the full transcription asset, and checks provider attempt count equals only transcription calls. Size/coverage/structural/fidelity counters and checkpoint bytes remain unchanged by re-probing.

## 12. SPEED_FIRST Behavior

429 with RetryInfo 37s retains PROVIDER_DELAY_EXCEEDS_LIMIT and performs no long sleep. On-demand probing does not remove bounded latency: each generation probe retains its existing timeout.

## 13. Quota-Limited Behavior

Actual 429 retains the existing temporary QUOTA_LIMITED registry state. Previously verified capability remains VERIFIED_ELIGIBLE; temporary quota does not become a permanent incompatibility fact.

## 14. Fallback Selection Behavior

Tests cover successful 3.5 fallback, continued 503 on 3.5 followed by verified 3.8 fallback, and exhaustion when both bounded re-probes remain transient. No full audio goes to an unverified path, native Transcribe or lite model. No later/undiscovered model is invented.

## 15. Configured vs Effective Primary

MODEL_POLICY and JOB_MODEL_SUMMARY now label configured_primary and effective_primary separately. Effective primary means the initial selected eligible model; actual_model continues to identify each request, including later fallback. MODEL_CALL console diagnostics carry the same distinction without changing persisted response metadata.

## 16. Model Selection Reason Diagnostics

eligibility_reason=CAPABILITY_ELIGIBLE describes the gate; routing_reason describes why the preference/fallback router selected that model. DISCOVERED_PREFERRED_ORDER and capability eligibility are different concepts, not conflicting facts.

## 17. Capability Cache Behavior

Cache and budgets remain SDK-client/job local and keyed by model, adapter, API method and endpoint. Re-probe replaces only that path's transient evidence. A fresh job starts fresh; no global or persisted negative cache added.

## 18. Files Modified

src/provider_capability.py; src/gemini_client.py. Added tests/test_transient_capability.py and this report. Pre-existing workspace changes retained. No Candidate prompt, validator, DABB, checkpoint, evaluator, baseline or schema edits.

## 19. Tests Added/Modified

Nine new parametrized cases cover 503/504/429/timeouts, bounded re-probing, deterministic exclusions, ordered fallback recovery/exhaustion, quota semantics, counters/checkpoint and diagnostics. Existing tests/assertions unchanged.

## 20. Full Regression Results

581 passed in 20.04s. git diff --check passed; only existing Windows line-ending notices. No live API calls in this stage.

## 21. Checkpoint/DABB Impact

No changes. Failed/unconfirmed blocks do not advance checkpoint. Capability probing cannot trigger physical shrink or consume size/coverage budgets.

## 22. Candidate_001 Readiness

CANDIDATE_001_READY_FOR_CONTROLLED_RERUN — code/state semantics are regression-verified. This is not a fresh provider availability guarantee: live execution still requires successful capability gates and applicable permission. No Candidate rerun performed, no prompt promotion.
