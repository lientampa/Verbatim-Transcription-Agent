# GOLDEN_001 benchmark implementation and fidelity audit

## Inputs and integrity

The user populated human.txt, machine_baseline.txt and local/audio.m4a. The audio was
inspected with ffprobe: 868.416 seconds. Manifest SHA-256 fingerprints bind the exact
bytes of all three inputs. The importer never writes them. Human lexical authority is
PRIMARY by user designation; timestamp authority is PARTIAL. No independent listening
review or new transcription was performed. Audio stays local/Git-ignored.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m src.golden_benchmark --fixture-dir tests/golden/golden_001 --output-dir reports
```

Inputs with a changed fingerprint fail closed. Future candidates must be separate
files/fixtures with their own manifest; never overwrite the original baseline. This
command is offline and has no live inference option. Existing Stage 7A remains intact.

## Architecture and scope

New src/golden_benchmark imports existing Stage 7A edit-distance/tokenization and
Hungarian assignment primitives. It provides a separate partial-metadata text importer,
alignment, evidence annotations and report CLI. It does not weaken the existing Stage
7A reviewed-reference schema or production validation. Baseline regression: 519 passed
in 23.76 seconds. Source files for production orchestration, prompt, validators,
checkpoint, cache, speaker rendering and provider selection were not modified here.

## Import and alignment

Original text is immutable. The evaluation view parses multiple utterances on one line,
records missing delimiters, marks backwards or spaced timestamp strings unreliable,
and retains raw display names while trimming label whitespace only for internal ID
association. Manifest excludes the reference's 'chú thích' display label from speaker
scoring; its lexical uncertainty evidence remains. Canonical production HH:MM:SS rules
are untouched. Invalid minutes/seconds and unparseable dialogue headers fail explicitly.

Deterministic monotonic DP supports 1:1, 1:N, N:1 (maximum group 3) and unmatched segments.
Cost combines relaxed token edit similarity, grouping penalty, and bounded timing
penalty. Default alignment tolerance is 15 seconds, configurable in the manifest.
Reliable timing gates weak matches; strong matches of at least five tokens may cross
the window, exposing a timing disagreement for review. Unreliable reference timing
never forces a lexical mismatch. No source timestamps are rewritten. Worst-case
segment alignment is quadratic in segment count with bounded group sizes; intended
for benchmark clips, not unbounded corpora.

## Taxonomy and metrics

ErrorType supports all requested lexical/phrase, entity/number, uncertainty, speaker,
timestamp and coverage categories. Severity INFO/MINOR/MAJOR/CRITICAL and risk
LOW/MEDIUM/HIGH are separate enums. Automatic groups are conservative lexical review
candidates, not acoustic proof. Hallucination/unsupported semantic invention is never
inferred from Levenshtein substitutions alone. Entity cases and normalization examples
are explicit reference-grounded annotations rather than a production dictionary.

Example D has primary WORD_INSERTION (the inserted 'sự'); formalization is an
interpretation, not a second counted error. Example E has primary PHRASE_SUBSTITUTION,
MAJOR/HIGH; unsupported-replacement interpretation does not add another event.
Location substitution has CRITICAL/HIGH and triggers FAIL_CRITICAL_REFERENCE_ERROR
in the offline report, regardless of average WER. This is not a production decision.

Strict WER uses NFC and whitespace tokens; relaxed WER additionally casefolds and uses
word/punctuation tokenization. Both retain repetitions, fillers, lexical differences,
entities and numbers. [không rõ] is one token. WER=(S+D+I)/N; Vietnamese units are
syllable/whitespace units, not linguistic word segmentation. Counts of automatic groups,
lexical edits and annotated examples are separate overlapping views, never summed.
No opaque aggregate score is introduced. Annotated errors are not exhaustive totals.

Speaker mapping optimizes one-to-one agreement for all display labels, including Anh
and Chị, rather than comparing raw labels. Mixed-speaker grouped alignments are excluded
from attribution. Fragmentation/merge counts are candidates requiring audio review, not
DER or confirmed identity errors. A consistent name permutation has zero error.
Timestamp deltas are observed disagreements, not proof the machine is wrong. Acoustic
coverage and silence adjudication remain unavailable; unmatched lexical segments are
not converted into physical gaps or checkpoint authority.

## Online/offline audit and evidence-based decision

OFFLINE_ONLY: reference lexical diff, annotated factual/terminology substitutions,
speaker permutation, human timestamp anomaly analysis, benchmark critical override.
ONLINE_HEURISTIC: suspicious phrase loops, polished language and uncertainty density.
ONLINE_SAFE: schema/identity/bounds checks (they prove those constraints, not fidelity).

All ten selected machine examples, including the exact repetition and critical location
substitution, return ACCEPT in reference-free FidelityValidator audit. This demonstrates
that heuristic acceptance is not acoustic verification. It does not justify a rule
that rejects every occurrence of either place name or any plausible phrase.

No production validator change is justified by these text comparisons alone. Preserve
ACCEPT/ACCEPT_WITH_WARNING/RETRY_REVIEW/FAIL and independent retry domains. Ordinary
uncertainty remains warning-capable; no reward is assigned to excessive uncertainty.
Existing prompt V5 already prohibits contextual invention, preserves fillers and
repetitions, requests structured JSON and absolute timestamps, and uses [không rõ].
Its 100% certainty language could overproduce unknowns; calibration requires acoustic
review, so it is flagged rather than blindly rewritten. Native Transcribe does not
receive the Flash system instruction; its verbatim API configuration remains separate.

Future optional context_terms should come from an explicit caller vocabulary, accompany
an instruction that hints never replace acoustic evidence, and be capability-gated.
Do not add benchmark entities globally. Native Transcribe vocabulary cannot simply be
combined with the current diarization/word-timestamp contract. No production schema or
speaker-ID migration is necessary for the offline benchmark identity representation.

## Deliverables and next step

reports/golden_001_baseline.md and .json contain fingerprints, alignment, input segments,
metrics, anomalies, evidence examples and online-validator audit. They contain private
transcript excerpts and should be shared only as intended by the user. Unit tests use
small synthetic text and do not require local audio or API credentials.

Next: adjudicate unresolved alignments, critical/terminology substitutions and speaker
identity against audio; add independently reviewed clips before changing production
heuristics or running a separately stored candidate. Coverage PASS cannot erase a
fidelity failure. Physical audio commits remain the sole resume authority.
