# Stage 7A: offline golden evaluation

Status: framework implemented; reviewed corpus PARTIAL; live NOT_RUN;
acoustic qualification UNAVAILABLE; release readiness PARTIAL.
Synthetic scores validate evaluator arithmetic only. Production stages 1-6 are unchanged.
No Gemini client, upload, transcription, cache mutation or live execution switch exists here.

## Run

From the repository root in PowerShell:

```powershell
.venv\Scripts\python.exe -B -m src.evaluate_golden --fixture-dir tests/golden_audio --output-dir evaluation/stage7a-framework
.venv\Scripts\python.exe -B -m src.evaluate_golden --fixture-dir D:/private/golden --fixture my_fixture --threshold-profile release-candidate --thresholds docs/stage7a-thresholds.json --output-dir evaluation/private-run
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_stage7a_evaluator.py
```

`GOLDEN_FIXTURE_DIR` supplies the default fixture directory. Each fixture directory contains
`reference.json` and `model_output.json`; optional notes and local audio remain private.
Empty directories succeed with zero fixtures, null metrics and PARTIAL release readiness.
Malformed fixtures or missing model outputs produce framework FAIL and report errors.
`EXAMPLE_ONLY` references are skipped. No live environment variable enables paid calls.
Reports contain transcript excerpts: keep private output directories private too.

## Schema and source identity

The versioned schema is `schemas/golden_reference.schema.json`; a complete synthetic
example is `tests/golden_audio/synthetic_verbatim/reference.json`.
Required fields: schema_version, fixture_id, language=vi, source_mode=AUDIO, source,
review, segments. Source has fingerprint_sha256 and positive duration_seconds.
Each segment has absolute start_seconds, nonempty speaker/text and uncertain.
Optional segment events contain type (filler/repetition/false_start) and literal text;
proper_names contains exact names. Optional regions have start/end/type and authoritative.
Timestamps must be finite, ordered and bounded by source duration. Same-time order is retained.
Uncertainty metadata must agree with the literal `[không rõ]` marker.

Review states: UNREVIEWED, PENDING_MANUAL_REVIEW, HUMAN_REVIEWED, DOUBLE_REVIEWED,
SYNTHETIC_TEST and EXAMPLE_ONLY. The evaluator never changes review status or reference text.
Only HUMAN_REVIEWED and DOUBLE_REVIEWED enter reviewed-reference aggregates.
Synthetic and pending references cannot qualify acoustic quality.

For a private reviewed fixture, a human supplies the transcript and explicit review status,
and sets source.audio_path to an absolute local path or a path relative to reference.json.
Compute the actual SHA256 of that audio, record it in source.fingerprint_sha256 and
model_output.source_fingerprint. The loader rehashes local audio and fails closed on mismatch
(GOLDEN_SOURCE_MISMATCH). Reviewed fixtures require locally verifiable audio.
Schema/metadata errors raise GOLDEN_FIXTURE_INVALID. No references are generated from models.

Stored model output is an object containing source_fingerprint, segments and optional
provenance. Segments accept start_seconds or canonical absolute timestamp (MM:SS/HH:MM:SS),
speaker and text; estimated SRT ends are ignored. Provenance can preserve per-block runtime
diagnostics and actual_models_used, a list checked against the existing hard ban on -lite.
Stored provenance is descriptive, never proof that a live run completed.

## Normalization and metrics

Strict normalization uses Unicode NFC and collapses whitespace only. Strict tokens split
on whitespace, retaining attached punctuation and case; `[không rõ]` remains one token.
Relaxed tokens casefold and remove punctuation but retain Vietnamese diacritics, lexical
fillers, repetitions and uncertainty markers. This is deterministic whitespace/syllable
WER, not a claim of linguistic Vietnamese word segmentation. CER uses NFC codepoints,
including normalized spaces and punctuation. No synonym matching or text beautification.

Levenshtein reports substitutions, deletions, insertions and reference_count independently
of segment alignment. Ties prefer diagonal, then deletion, then insertion. WER/CER use
(S+D+I)/N; insertions may produce rates above one. Ratios include numerator, denominator,
value and availability; zero denominators produce null, never a fabricated perfect score.
Exact edit distance uses linear memory and quadratic worst-case time. Segment alignment
uses a quadratic DP table, so corpus fixtures should be bounded clips.

Reference event annotations take precedence, including an explicit empty events list.
Without annotations: fillers use configurable EvaluationConfig.fillers (default ờ, ừ, ừm,
ờm, à); repetitions are adjacent equal relaxed tokens; false starts are phrases ending in
ellipsis. These heuristics are imperfect and do not establish acoustic truth.
Model event counts come from text only, ignoring model-supplied annotations. Annotated
reference phrases are also searched in model text. Matches use aligned-group multiset
intersection; precision and recall expose match/reference/model counts separately.
No composite fidelity score hides these individual components.

Deletions flag LIKELY_OMISSION, insertions POSSIBLE_HALLUCINATION, substitutions
TEXT_DISAGREEMENT. These are human-review candidates, not definitive acoustic judgments.

## Alignment, time and speakers

Monotonic dynamic programming supports 1:1, 1:N and N:1 groups (default maximum three,
configurable up to five). Candidate groups require relaxed text similarity >=0.35,
anchor delta <=15 seconds and internal spans <=15 seconds. Parameters are configurable
through EvaluationConfig; CLI exposes --alignment-max-time-delta and
--alignment-min-text-similarity. No candidate meeting bounds means ALIGNMENT_UNRESOLVED;
no forced matching, reordering or invented timing. Full-text WER still includes all errors.

Timestamp metrics use absolute first-start anchors of resolved groups only. Grouped interior
word timing is unavailable. Report count, raw errors, median, nearest-rank p90, max and
within 1/2/5 seconds plus configured tolerance. Set --timestamp-tolerance or
GOLDEN_TIMESTAMP_TOLERANCE_SECONDS (default 1). SRT durations never become acoustic timing.

Anonymous `Người nói N` labels use deterministic optimal one-to-one count-weighted Hungarian
mapping across the entire fixture. Explicit named labels retain identity, reserving correct
names. Named swaps cannot be hidden by remapping. Grouped speaker attribution counts paired
reference/model labels, not inferred word durations. Reports include accuracy, swap,
fragmentation, under-identification and unsupported named labels. Unknown explicit labels
produce review flags; generic roles may also require human disambiguation. A name flag is
not a determination that a model invented a real person. Swap/fragmentation review flags
are fixture-wide and can mark otherwise correct groups for contextual review.

## Additional diagnostics

Uncertainty precision/recall matches marker counts inside resolved groups. Missing markers
with specific model text flag UNCERTAIN_OVERGUESS; extra markers flag FALSE_UNCERTAINTY.
Golden speech coverage is retained relaxed reference tokens within resolved groups divided
by all reference tokens. It is lexical coverage, independent of production committed/audio
coverage, and cannot establish that the full physical audio was transcribed.
Proper names require NFC exact, case-sensitive, diacritic-preserving phrase matches with
word boundaries. Authoritative silence/noise/music/non_speech regions flag model segment
starts inside those regions; no segment-end inference is made. Overlap regions receive
review flags rather than invented speaker separation.

## Reports, thresholds and gates

Outputs: summary.json, metrics.json, fixtures.json, review_queue.jsonl and report.md.
Each queue item contains fixture id, time range, golden/model excerpts, flags and severity.
The end is a review-window boundary (next start or source duration), not an acoustic end.
Per-fixture results include config, source identity, provenance and raw counts.
Aggregates sum edit operations and denominators, never average fixture WERs. Timing pools
observations; event, name, speaker and coverage ratios pool their own denominators.
Synthetic aggregates remain separate from reviewed-reference comparisons.

Both evaluation-only and release-candidate profiles use explicit thresholds. The example
threshold file leaves all limits null (DISABLED). No release-quality limit is assumed.
Unavailable denominators remain UNAVAILABLE. All mandatory limits must be configured and
available before the threshold engine can PASS. Evaluation settings never modify runtime
retry, fallback, coverage or validation policies.

The release gate also requires reviewed fixtures, completed live production-path executions
for the mandatory set, passing thresholds and no outstanding critical reviews. Its positive
unit test is a simulation only. Stage 7A always supplies zero live executions; even reviewed
stored outputs cannot yield release PASS. A future live orchestrator must supply verified
execution evidence and human review disposition, not trust arbitrary stored provenance.

## Stage 7B boundary

The pure evaluate(reference, recognized, source_fingerprint, config, provenance) interface
accepts canonical pipeline segments for a future opt-in runner. Stage 7A implements no such
runner. Stage 7B still needs human-reviewed private audio/reference fixtures, selected quality
thresholds, real production-path run evidence, long-audio/resume scenarios and review of
flagged disagreements. Do not auto-start Stage 7B or use synthetic tests as acoustic evidence.
