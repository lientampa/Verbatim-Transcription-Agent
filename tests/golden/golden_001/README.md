# GOLDEN_001 — populated reference and immutable machine baseline

The user supplied all three inputs. `manifest.json` records exact SHA-256 hashes and
ffprobe duration. Keep `human.txt` and `machine_baseline.txt` unchanged. Local audio
remains Git-ignored. Do not replace the baseline with an improved candidate.

Run offline from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m src.golden_benchmark --fixture-dir tests/golden/golden_001 --output-dir reports
```

Reference timestamps have PARTIAL authority. Import anomalies are recorded separately;
no spelling, dialect, fillers, repetitions or timestamps are corrected in the source.
`annotations.json` contains selected reference-grounded examples, not exhaustive error
labels or independent acoustic adjudication. Label `chú thích` is excluded from speaker
metrics by manifest configuration. No human reference enters production validation.

Alignment, detailed evaluation and the report are under `reports/`; see
`docs/golden-001-benchmark.md` for the algorithm, counting policy and fidelity audit.
This partial-metadata fixture uses the new offline importer, while the existing Stage
7A `reference.json` format and its strict metadata checks remain unchanged.
