# CODEX GOLDEN_001 CANDIDATE_001 A/B REPORT

Status: CANDIDATE_INVALID_PROMPT_NOT_APPLIED

## 1. Executive Summary

CANDIDATE_INVALID_PROMPT_NOT_APPLIED. No candidate transcript was generated; no A/B quality conclusion is available.

## 2. Baseline Prompt Version

BASELINE_PROMPT_PROVENANCE = UNKNOWN. Current production V5 is unchanged; its use for the observed baseline is not proven.

## 3. Candidate Prompt Version

VERBATIM_AUDIO_PROMPT_CANDIDATE_001. Effective instruction SHA-256: 135b3fbcd42e15776b29ee8e9d747fb6944140c76fee9c40f27bc50a6078799c. File hash differs because the production loader strips outer whitespace; both hashes are recorded.

## 4. Exact Prompt Changes

Operational evidence-based uncertainty replaces 100% certainty. Preserve dialect, fillers, repeats and unfinished speech; conservative entities; context cannot supply missing words. Existing JSON, timestamp and stable speaker contracts retained. No benchmark answers included.

## 5. Files Modified

Added experiments/candidate_001/{system_prompt.txt,run.py,README.md,.gitignore}, tests/test_candidate_001.py and these candidate reports. EXPERIMENT_INFRASTRUCTURE_CHANGE = isolated delivery guards. TRANSCRIPTION_PROMPT_CHANGE = separate candidate artifact. No provider capability/plumbing added to production.

## 6. Production Components Not Modified

Production V5, schema, validators, DABB, retry/fallback/model order/no-lite, checkpoint/resume, source identity, cache semantics, renderer and evaluator algorithms unchanged. Baseline fixtures, annotations and reports were not written. Human, baseline and audio hashes verified.

## 7. Candidate Runtime Model Provenance

Discovery selected gemini-2.5-flash. Preflight passed with exact system instruction attached; the actual outgoing Flash request also passed the guard, then returned HTTP 404. Existing fallback selected gemini-3.5-transcribe; its config contains no system instruction. Guard aborted BEFORE interactions.create. One transcription request sent, zero Transcribe requests. Production MODEL_CALL/actual_models_used logs include the intercepted selection; they do not prove a native API call occurred.

## 8. Model Confound Assessment

BASELINE_MODEL_PROVENANCE_UNKNOWN. This is BASELINE_001 observed output versus a proposed candidate, not a proven V5-versus-candidate experiment.

## 9. DABB/Physical Range Behavior

BLOCK_001 [0, 868.416] seconds; physical generation 1. Separate audio upload/cache/state under the recorded runtime directory. No block accepted.

## 10. Retry Behavior

Existing 404 model quarantine and ordered fallback preserved. Native path aborted, not skipped. No size/coverage retry executed.

## 11. Coverage Behavior

Not evaluated: no successful transcription response. No claim of coverage PASS.

## 12. Baseline Relaxed WER

50.27% from unchanged baseline report.

## 13. Candidate Relaxed WER

UNAVAILABLE: candidate not generated.

## 14. Baseline Strict WER

56.53% from unchanged baseline report.

## 15. Candidate Strict WER

UNAVAILABLE.

## 16. Substitution Comparison

Baseline relaxed S=324; candidate UNAVAILABLE.

## 17. Omission Comparison

Baseline relaxed D=110; candidate UNAVAILABLE.

## 18. Insertion Comparison

Baseline relaxed I=131; candidate UNAVAILABLE.

## 19. Alignment Comparison

Candidate UNAVAILABLE; no alignment tolerance/algorithm changes.

## 20. Repetition Preservation

Not measured for candidate.

## 21. Over-Normalization Comparison

Not adjudicated for candidate.

## 22. Unsupported Replacement Comparison

Not adjudicated for candidate; heuristic candidates must not be presented as confirmed semantic errors.

## 23. Entity Error Comparison

REQUIRES_CANDIDATE_ADJUDICATION after a valid future candidate. Baseline error strings are not required to appear in candidate.

## 24. Uncertainty Comparison

Candidate marker count, containing segments, delta and lexical span impact UNAVAILABLE. Future automatic marker counts are separate from acoustic appropriateness adjudication.

## 25. Speaker Comparison

Candidate UNAVAILABLE; speaker mapping methodology unchanged.

## 26. Critical Semantic Error Comparison

Candidate UNAVAILABLE, not zero. Existing baseline annotation history retained.

## 27. High-Risk Semantic Error Comparison

Candidate UNAVAILABLE, not zero.

## 28. Regression Test Results

542 passed in 12.08s; git diff --check PASS before live. Tests verify real Flash adapter config, forbidden-reference reads, missing/wrong instruction, no-lite rejection and stopping native fallback before SDK invocation.

## 29. Interpretation

Model discovery availability is insufficient to establish request eligibility: the listed Flash model returned 404. The safety guard worked, but prompt quality was not evaluated. The existing native Transcribe request cannot carry the candidate instruction through its current adapter. No live retries or silent rerouting will follow this abort.

## 30. Recommendation

INCONCLUSIVE_MODEL_CONFOUND. The experiment is invalid because the next provider path does not apply the prompt, with baseline model provenance also unknown. Do not promote or change FidelityValidator. Any new provider-path experiment requires an explicit scope decision; current approved attempt is stopped.

Runtime evidence: D:\ZEC\Verbatim-Transcription-Agent\experiments\candidate_001\runtime\51accc7f8cae4ee8a056291150c2a390
