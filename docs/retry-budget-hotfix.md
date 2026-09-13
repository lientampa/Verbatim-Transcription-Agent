# Retry budgets and output-limit diagnostics

Counters are scoped to one logical block and do not reset on model fallback.

| Counter | Meaning / decision authority |
| --- | --- |
| provider_attempt_count | External generation calls reported by GeminiClient, including same-model retries and fallback; SDK retry budgets remain per model. |
| validation_attempt_count | One-based orchestration validation retry slot (legacy v_attempt); consumed by parsing/validation failures, not size/coverage/provider failures. It is cumulative for the logical block, not reset per slice. |
| structural_retry_count | Structural failures observed by orchestration; descriptive, not a size decision. An empty response handled by the client's existing fallback path carries EMPTY_RESPONSE provenance. |
| size_retry_count | Completed SIZE_FAILURE-driven physical rebuilds. SIZE_RETRY_LIMIT=2 permits initial slice plus two smaller retries. Exhaustion checks this counter only. If the setting is absent, the allowance defaults to VALIDATOR_MAX_RETRIES - 1 for compatibility; consumed counters are never shared. |
| coverage_generation_count | Starts at 1; increases only for a successful coverage-driven rebuild, retaining existing coverage limits. |
| physical_generation_count | Starts at 1; increments on an actual physical rebuild (size, coverage or existing model-profile replan). Provider calls do not change it. |
| attempt_count | Total orchestration iterations; diagnostic only. |

No global physical-generation limit is introduced. Exhaustion and minimum duration are distinct: SIZE_RETRIES_EXHAUSTED versus MIN_BLOCK_REACHED / legacy BLOCK_SIZE_EXHAUSTED. OUTPUT_LIMIT reports counters before the rebuild; SHRINK_REBUILD_REUPLOAD reports the completed transition count afterward. Checkpoint boundaries advance only after validation permits commit.

## Output cap audit

MODEL_MAX_OUTPUT_TOKENS (default 8192) is a shared DABB planning ceiling, not a field previously sent to Gemini. GenerateContentConfig previously omitted max_output_tokens entirely. Consequently the supplied production evidence does not establish an application-imposed low generation cap, nor does it establish the provider's effective default limit or the cause of output expansion.

Discovery uses the installed Google SDK Model.output_token_limit field. A fresh GenerateContentConfig is built for each actual model. Optional GEMINI_MAX_OUTPUT_TOKENS is clamped to that actual model's positive discovered output_token_limit. Without the option, max_output_tokens remains None (provider default); no automatic increase is introduced. If discovery omits a capability, diagnostics retain None, never a fabricated limit. Model chain, SPEED_FIRST and DABB planning/shrink formulas are unchanged.

Every returned response retains available usage_metadata fields using SDK names (prompt_token_count, candidates_token_count, total_token_count, thoughts_token_count and other provider fields), finish reason, configured cap, discovered limit, physical duration and DABB target. MAX_TOKENS diagnostics and OUTPUT_LIMIT logs expose the context without transcript content. Definitively diagnosing a past MAX_TOKENS event requires its usage/configuration evidence; the old logs supplied do not contain it.

DABB's existing minimal AdaptiveBlock proxy now receives the actual logical block id. Empty-response fallback uses EMPTY_RESPONSE rather than None_RETRY_POLICY. No JSON repair, validator relaxation, Stage 5 redesign, or Stage 6 implementation is included.
