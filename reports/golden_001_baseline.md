# GOLDEN_001 baseline evaluation

Source duration: 868.416 seconds. Human lexical authority: PRIMARY; timestamp authority: PARTIAL.
Human segments: 135; machine segments: 126.
Aligned groups: 88; unmatched human: 18; unmatched machine: 15.

## Lexical metrics

Relaxed syllable/whitespace WER: 50.2669%. S=324, D=110, I=131, N=1124.
Strict WER: 56.5256%.
NFC; strict whitespace tokens; relaxed casefold/punctuation. Fillers/repetitions/entities/numbers retained; [không rõ] is one token.

WER edits, automatic group classifications and evidence-annotated examples are separate views and must not be summed. One primary label per example; no composite score.

## Critical and verbatim examples

Offline fidelity decision: FAIL_CRITICAL_REFERENCE_ERROR.

Explicit annotated CRITICAL errors: 1; HIGH semantic risk: 6. These are not exhaustive corpus totals.

| Example | Human | Machine | Primary label | Severity |
|---|---|---|---|---|
| G001-E001 | Nên nên nên đầu tư. | Nên nên nên đầu tư. | EXACT_MATCH | INFO |
| G001-E002 | Nó có bọt rồi đó. | Nó có bột rồi đó. | WORD_SUBSTITUTION | MINOR |
| G001-E003 | qua coi thực tế | qua xem thực tế | OVER_NORMALIZATION | MINOR |
| G001-E004 | Khách họ cần thực tế hơn. | Khách họ cần sự thực tế hơn. | WORD_INSERTION | MINOR |
| G001-E005 | Hình như phổ nhĩ quýt thơm. | Em uống phô mai rứa thôi. | PHRASE_SUBSTITUTION | MAJOR |
| G001-E006 | Dạ em ở Quảng Nam. | Dạ em Đà Nẵng luôn. | PHRASE_SUBSTITUTION | CRITICAL |
| G001-E007 | Phổ Nhĩ quýt | phổ nhị quất | ENTITY_ERROR | MAJOR |
| G001-E008 | thiềm thừ | thiểm thử | ENTITY_ERROR | MAJOR |
| G001-E009 | Dạ Tử Đào. | Dạ từ đào. | ENTITY_ERROR | MAJOR |
| G001-E010 | Nghê với lại | Nghe bây giờ | ENTITY_ERROR | MAJOR |

## Speaker, uncertainty and timestamp observations

Speaker metrics: `{"mapping": {"SPEAKER_002": "SPEAKER_002", "SPEAKER_003": "SPEAKER_003", "SPEAKER_005": "SPEAKER_001", "SPEAKER_006": "SPEAKER_004"}, "eligible_groups": 55, "errors": 31, "fragmentation_candidates": 5, "merge_candidates": 5, "unmatched_machine_identities": ["SPEAKER_001", "SPEAKER_004", "SPEAKER_007"], "human_labels": {"SPEAKER_001": "Người nói 3", "SPEAKER_002": "Người nói 2", "SPEAKER_003": "Người nói 1", "SPEAKER_004": "Người nói 4", "SPEAKER_005": "chú thích", "SPEAKER_006": "Người nói 5"}, "machine_labels": {"SPEAKER_001": "Chị", "SPEAKER_002": "Anh", "SPEAKER_003": "Người nói 3", "SPEAKER_004": "Người nói 4", "SPEAKER_005": "Người nói 5", "SPEAKER_006": "Người nói 6", "SPEAKER_007": "Người nói 7"}, "limitation": "Group-count weighted label permutation; mixed-speaker groups excluded. Candidates require acoustic adjudication; not DER."}`.
Uncertainty units: human=29; machine=6. No accuracy reward for unknown markers.
Human import anomalies: `[{"reason": "NON_DIALOGUE_PREFIX", "text": "BẢN PHIÊN ÂM NGUYÊN VĂN"}, {"reason": "MISSING_DELIMITER", "index": 12, "line": 13}, {"reason": "MISSING_DELIMITER", "index": 24, "line": 25}, {"reason": "MISSING_DELIMITER", "index": 65, "line": 66}, {"reason": "TIMESTAMP_REFERENCE_ANOMALY", "index": 85, "line": 86, "timestamp": "00:07: 23", "previous_seconds": 439}, {"reason": "MISSING_DELIMITER", "index": 89, "line": 90}, {"reason": "MISSING_DELIMITER", "index": 90, "line": 91}, {"reason": "TIMESTAMP_REFERENCE_ANOMALY", "index": 94, "line": 95, "timestamp": "00:08:19", "previous_seconds": 502}, {"reason": "MISSING_DELIMITER", "index": 103, "line": 104}, {"reason": "MISSING_DELIMITER", "index": 104, "line": 105}, {"reason": "MISSING_DELIMITER", "index": 118, "line": 119}, {"reason": "TIMESTAMP_REFERENCE_ANOMALY", "index": 123, "line": 124, "timestamp": "00:10:50", "previous_seconds": 705}]`.
Timestamp disagreements do not establish machine error. Acoustic coverage gaps are UNAVAILABLE without audio adjudication.

## Validator implications

Structural PASS and coverage PASS do not establish fidelity. Reference-free heuristic results below cannot detect arbitrary factual or entity substitutions.
`[{"annotation_id": "G001-E001", "online_without_reference": "ACCEPT", "issues": [], "offline_primary_error": "EXACT_MATCH"}, {"annotation_id": "G001-E002", "online_without_reference": "ACCEPT", "issues": [], "offline_primary_error": "WORD_SUBSTITUTION"}, {"annotation_id": "G001-E003", "online_without_reference": "ACCEPT", "issues": [], "offline_primary_error": "OVER_NORMALIZATION"}, {"annotation_id": "G001-E004", "online_without_reference": "ACCEPT", "issues": [], "offline_primary_error": "WORD_INSERTION"}, {"annotation_id": "G001-E005", "online_without_reference": "ACCEPT", "issues": [], "offline_primary_error": "PHRASE_SUBSTITUTION"}, {"annotation_id": "G001-E006", "online_without_reference": "ACCEPT", "issues": [], "offline_primary_error": "PHRASE_SUBSTITUTION"}, {"annotation_id": "G001-E007", "online_without_reference": "ACCEPT", "issues": [], "offline_primary_error": "ENTITY_ERROR"}, {"annotation_id": "G001-E008", "online_without_reference": "ACCEPT", "issues": [], "offline_primary_error": "ENTITY_ERROR"}, {"annotation_id": "G001-E009", "online_without_reference": "ACCEPT", "issues": [], "offline_primary_error": "ENTITY_ERROR"}, {"annotation_id": "G001-E010", "online_without_reference": "ACCEPT", "issues": [], "offline_primary_error": "ENTITY_ERROR"}]`

Human-vs-machine lexical/semantic comparison is OFFLINE_ONLY. Repetition/normalization heuristics are ONLINE_HEURISTIC. Schema and physical identity checks are ONLINE_SAFE.
Do not hard-code benchmark entities into production. Existing native Transcribe uses provider verbatim configuration and does not receive the Flash system instruction.

## Recommended changes and limitations

Keep production acceptance/retry/checkpoint/coverage policy unchanged: these examples do not justify guessing semantic correctness from machine text alone.
The Flash prompt already forbids contextual invention. Its absolute 100% certainty wording should be calibrated using acoustic review before changing it; excessive uncertainty is not automatically correct.
Optional future context_terms should be user-supplied hints only, never forced replacement, and capability-gated (native vocabulary cannot be combined with current diarization/word timestamps).
Review unresolved alignments, speaker fragmentation candidates and human timing anomalies against audio. No live inference or listening adjudication performed.
Detailed alignment, normalized metrics, classifications, fingerprints and anomalies are in golden_001_baseline.json.
