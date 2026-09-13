# Gemini thinking / MAX_TOKENS hotfix

## Audit and scope

The only production SDK generate_content call is GeminiClient._call_model. Before
this change GenerateContentConfig set system_instruction, temperature=0 and JSON
response_mime_type, with no thinking_config or AFC disable. The per-model cap was
set only when GEMINI_MAX_OUTPUT_TOKENS was explicitly configured. Discovery limits
were used to clamp an override, not to supply a default. This explains None in the
old diagnostic. MODEL_MAX_OUTPUT_TOKENS (default 8192) is only the DABB prediction
budget, not the SDK output limit. This repo has no separate ModelCapabilities object:
SDK Model.output_token_limit populates GeminiClient.model_output_limits.

The user-provided 62915 thought / 1085 visible token observation is consistent with
reasoning pressure, not proof that 120 seconds of audio cannot fit. No live call was
made to reproduce or claim resolution of that acoustic run.

## Request policy

TRANSCRIPTION_THINKING_LEVEL defaults to minimal. Google's support table audited
2026-09-13 lists low as the minimum for 3.8/3.7 Flash and minimal for 3.6/3.5 Flash.
The wrapper maps accordingly; unknown Gemini 3 variants conservatively use low as
the default minimum. Non-Gemini-3 models receive no Gemini-3 thinking parameters.
No numeric thinking_budget is sent. Explicit low/medium/high settings are supported.

Source: https://ai.google.dev/gemini-api/docs/thinking

Each actual call rebuilds config from the actual model, including same-model retry,
fallback, config recovery, coverage retry and size rebuild. Explicit provider output
cap is clamped to a positive discovered model limit. Without override, that discovered
limit is used. Without reliable discovery, an explicit override is retained or the
SDK field is omitted and diagnostics say PROVIDER_DEFAULT / UNDISCOVERED. No numeric
limit is guessed, and the DABB prediction budget is unchanged. Provider output limits
include reasoning and visible output; reducing reasoning does not guarantee no truncation.

AFC is not required: the request has no function tools and JSON parsing/schema validation
happens in the existing transcription pipeline. Installed google.genai.models emits the
warning when its default AFC branch is entered; should_disable_afc returns early for
AutomaticFunctionCallingConfig(disable=True). We set that flag and leave tools absent.
No Chat migration or schema weakening.

## Recovery and counters

MAX_TOKENS with positive thought tokens >=4 times max(visible tokens,1) is diagnostically
REASONING_OUTPUT_PRESSURE. The factor is a diagnostic heuristic, not an acoustic validator.
Unknown usage remains output pressure with ratio unavailable. Partial generated text is
never returned for parsing/commit.

If the actual thinking level is above that model's minimum, one generation-config retry
per logical block lowers it and calls the same model with the exact same uploaded asset.
It does not re-enter DABB, coverage validation, upload or slicing. Provider attempt count
increments; generation_config_retry_count increments independently. The recovered model
stays at its minimum for the client lifetime. The logical-block counter resets on a new
block id. A second truncation propagates as SIZE_FAILURE. No model-health acoustic strike
is recorded for the config retry.

Size recovery still consumes only its existing size budget. At or below the normal
minimum, SIZE_FAILURE may use EMERGENCY_SIZE_MIN_BLOCK_SECONDS (default 60, clamped to
the normal floor). 60 seconds reuses the existing project default MIN_DURATION and avoids
introducing a new 30-second assumption. Normal configured minimum 120 recovers through
120 -> 72 -> 60 with the existing 0.6 shrink factor. At the emergency floor, failure is
EMERGENCY_SIZE_FLOOR_EXHAUSTED. Earlier exhaustion of SIZE_RETRY_LIMIT remains possible
and intentional. Coverage failures and model replans cannot use the emergency floor.
New-block MAX_FIRST and its existing guard remain unchanged.

Only a successful slice rebuild increments physical generation. Size rebuilds do not
consume coverage-generation budget. No failed result advances checkpoint or reaches the
merger. A simulated 868.416-second job preserves checkpoint 614.208 when its tail fails.
The exact BLOCK_004 interval test covers 254.208 -> 127.104 -> 120 -> 72. The checkpoint
integration test pins that fractional interval by bypassing unrelated token-rounding
model replans; other runtime tests use the real model-selection hook.

## Verification and production re-test

Baseline: 408 passed. Hotfix tests use fake SDK responses and temporary fake audio;
no paid calls, real checkpoint updates or Stage 7B execution occur during verification.
Run from the repository root:

```powershell
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_thinking_hotfix.py
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
git diff --check
```

For a later production re-test, preserve the existing source file and checkpoint and set
TRANSCRIPTION_THINKING_LEVEL=minimal plus EMERGENCY_SIZE_MIN_BLOCK_SECONDS=60 in the
production environment. Keep MIN_DURATION=120 if that is the job's existing normal floor.
Retain an intentional GEMINI_MAX_OUTPUT_TOKENS override; leaving it unset now uses the
discovered cap. Run the normal existing entry point (`python -m src.main`) to resume.
Confirm source fingerprint b8ae1d822af7e40e0753aaa56aa30786bca6873ab60b738be5648c0ca2951a45
and NEXT_START=614.208 before expecting the supplied failed run to continue.

Inspect GENERATION_CONFIG, MAX_TOKENS_CLASSIFICATION, OUTPUT_LIMIT and, if needed,
GENERATION_CONFIG_RETRY / EMERGENCY_SIZE_SHRINK. Commit must occur only after the
unchanged structural, fidelity and coverage checks pass. Minimal thinking is a cost
control, not a proof of acoustic fidelity. Keep live qualification unavailable until
real results have been reviewed. Do not start Stage 7B as part of this hotfix.
