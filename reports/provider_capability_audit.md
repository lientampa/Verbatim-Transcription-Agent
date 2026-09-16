# Fresh Provider Capability Audit

Job: 60012234e91142ddb4a48a0b8ccfd7bd. Started UTC: 2026-09-15T15:54:45.210678+00:00. Completed UTC: 2026-09-15T15:55:25.618676+00:00.

## 1. Scope Confirmation

No production code modified. Audit-only implementation; conditionally authorized Candidate run used existing unchanged runner. Existing CapabilityGate was invoked in the requested discovery-filtered order without modifying src/capability_audit.py.

## 2. Focused Regression

`.\.venv\Scripts\python.exe -B -m pytest tests/test_provider_capability.py tests/test_candidate_001.py -q -p no:cacheprovider`: 24 passed in 1.27s. No assertions changed.

## 3. Runtime Discovery

All three preferred Flash models were returned by discovery and intersected with existing effective_model_chain: gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash. Probed in that order. No native or 2.5 audit request.

## 4. Live Capability Results

| Model | Audio | System Instruction Delivery | Structured Output | Schema | Nonce Compliance | Result |
|---|---|---|---|---|---|---|
| gemini-3.8-flash | UNKNOWN | PASS | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| gemini-3.6-flash | PASS | PASS | PASS | PASS | PASS | VERIFIED |
| gemini-3.5-flash | UNKNOWN | PASS | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

System instruction delivery PASS proves attachment at the outgoing call only. For failed requests, support and semantic compliance remain UNKNOWN. The existing gate field system_instruction represents nonce compliance; the audit reports it separately. The nonce existed only in the system instruction and was matched exactly in a schema-valid segment text.

## 5. Provider Errors

3.8 Flash: HTTP 503, transient provider failure. 3.5 Flash: HTTP 504, server timeout; raw existing reason CAPABILITY_PROBE_FAILED, capability remains UNKNOWN. Neither establishes incompatibility or permanent exclusion. Upload succeeded; these were generation calls.

## 6. Eligible Paths

gemini-3.6-flash + generate_content + models.generate_content + configured Gemini v1beta endpoint, scoped to this audit client/job. All five PASS; finish reason STOP.

## 7. Candidate_001 Decision

CANDIDATE_001_PROVIDER_VERIFIED — CANDIDATE RUN PERMITTED

Candidate outcome (separate): the unchanged runner reverified 3.6 Flash and submitted Candidate requests with the correct candidate prompt hash. Coverage retries reduced physical ranges 868.416 → 434.208 → 217.104 → 120 seconds; a 503 triggered the existing same-model retry. The process later ended without result.json or output transcript. Current process check found no active candidate runner. Cause UNKNOWN; do not infer provider failure from missing final result. Zero confirmed blocks; next_audio_start_us=0. No WER or quality result, no promotion.

## 8. Files Changed

Updated only reports/provider_capability_audit.json and reports/provider_capability_audit.md. The authorized Candidate invocation additionally generated a new isolated runtime directory: experiments/candidate_001/runtime/fbb66207d79144eaae06acb2675220b0. All pre-existing protected files match their pre-audit SHA-256 snapshot.

## 9. Git Diff Safety

git status --short before/after identical: True. Full status is retained in JSON. Existing modified source/tests and untracked fixtures/cache/experiments/reports predate this task; no source/test change was introduced. git diff --check exit=0; only line-ending notices.

## 10. Remaining Unknowns

Flash 3.8/3.5 capability under successful service conditions; Candidate completion and transcription quality; cause of terminated Candidate process. Fresh probe evidence is historical to this job, not a perpetual availability guarantee.

`MODEL_DISCOVERED != MODEL_REQUESTABLE != MODEL_CONTRACT_VERIFIED != MODEL_QUALITY_VERIFIED`
