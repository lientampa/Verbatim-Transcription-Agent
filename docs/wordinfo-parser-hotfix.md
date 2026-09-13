# WordInfo parser and preflight semantics hotfix

Baseline: 456 tests passed using Python -B / pytest -p no:cacheprovider.

Audit found no decimal-unit conversion or byte-index/time confusion in the old
parser. END_OFFSET_OUT_OF_BOUNDS combined end-before-start with end-after-duration.
The supplied production log lacks raw offsets, so the exact word-4 cause remains
unconfirmed; this patch must not be described as a verified live production repair.

word_timing explicitly parses start_offset/end_offset as finite decimal seconds,
validates relative bounds against slice duration, then computes separate absolute
start/end values. start_index/end_index remain text byte positions only. The existing
1e-6 second technical tolerance is retained: 120.000001s is accepted for 120 seconds;
120.001s, 120.003s and larger overruns are rejected. Nothing is clamped or inferred.
Missing start returns a controlled timestamp mapping error; missing end stays None.

TRANSCRIBE_WORDINFO_PARSE_ERROR logs only timing/index fields (bounded to 80 chars),
word index, slice duration and physical start. No word/transcript text is dumped.
The production-like synthetic fixture accepts word 4 within 868.416 seconds, but
is not a captured provider response and does not prove the actual failing offsets.

Canonical text is now sliced from provider content text around verified ordered
word correspondences, preserving punctuation and internal whitespace instead of
joining annotation words with synthetic spaces. Missing lexical correspondence
still fails closed. Model-output text content remains authoritative. Unrelated
annotation types are ignored. Native verbatim request and speaker policy unchanged.

Metadata-only preflight now says DISCOVERED_UNVERIFIED and permits an optimistic
first real call. A successful call sets ELIGIBLE; runtime 404 still quarantines for
the job. No inference preflight, configuration rewrite, provider order, retry budget,
coverage, checkpoint, merger/renderer or evaluator changes.

Production re-test: use the normal command and existing checkpoint without force.
Capture TRANSCRIBE_WORDINFO_PARSE_ERROR if failure recurs: compare raw start/end,
parsed seconds and duration, including whether end precedes start. Do not increase
bounds tolerance to bypass malformed annotations. Verify metadata state remains
unverified until a successful call, unavailable models are quarantined, and failed
blocks do not advance checkpoint. No live provider call was made for this patch.
