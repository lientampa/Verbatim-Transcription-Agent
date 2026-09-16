# Candidate 001 infrastructure

EXPERIMENT_INFRASTRUCTURE_CHANGE: an experiment-only client subclass checks the selected model before upload, and guarded adapters check the exact outgoing config on every attempt. A BaseException abort prevents production retry handlers from routing around invalid prompt delivery. Production modules are not edited. No provider capability is added or assumed.

TRANSCRIPTION_PROMPT_CHANGE: separately versioned system_prompt.txt. Production V5 remains unchanged.

Run tests first, then `python -B -m experiments.candidate_001.run --preflight`. This performs existing model discovery but no audio upload or transcription. Only after successful preflight run with `--live`. Both operations create a fresh isolated runtime directory and use production run_pipeline. The guard remains active across fallback. A selected native Transcribe path aborts with CANDIDATE_INVALID_PROMPT_NOT_APPLIED; it is never skipped for this experiment.

Credentials/configuration are inherited from the existing local environment without copying .env into artifacts. Runtime contains only the audio, candidate prompt, unchanged schema and generated artifacts; reference files are not read until successful transcription completes. Local runtime artifacts are ignored by git.

Baseline prompt and model provenance remain UNKNOWN. Candidate comparison uses unchanged automatic evaluator functions with no baseline machine-string annotations attached to candidate. All existing semantic annotations require separate candidate adjudication; absent baseline error strings are not success evidence. Marker span token impact is unavailable without acoustic adjudication.

## Experiment observability

EXPERIMENT_OBSERVABILITY_CHANGE: diagnostic_events.jsonl records REQUEST_START (ABOUT_TO_CALL_SDK), REQUEST_END (SDK_RETURNED), REQUEST_ERROR (SDK_RAISED), stage transitions and finalization. run_state.json is atomically replaced and records final state, last request, checkpoint end and candidate-artifact presence. inspect_state reports RUN_INCOMPLETE_NO_FINALIZATION for unfinished states; this alone does not prove a kill or timeout.

Exceptions propagate unchanged. Diagnostics redact arbitrary exception messages (which may contain credentials, prompts or reference text), retaining exception type, numeric provider code and traceback frame locations without source lines or locals. Diagnostic write failures are best effort and must not replace the original error. Production checkpoint is read only during diagnostic finalization. No request timeout or retry policy is added. Hard process termination cannot guarantee a final record.
