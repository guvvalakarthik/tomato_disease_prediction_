# Evidence reporting

Generate the public results report only from machine-produced artifacts:

```bash
python scripts/generate_results_report.py \
  --metadata model/release/metadata.json \
  --clean-metrics model/release/metrics.json \
  --field-metrics model/release/field_metrics.json \
  --comparison reports/generated/comparison.json \
  --latency reports/generated/latency.json \
  --user-study reports/generated/user-study.json \
  --output reports/generated/EVIDENCE_REPORT.md
```

Locked evaluation also writes `evidence_manifest.json`, which binds both confusion
matrices, the reliability diagram, risk/coverage plot, metrics, and metadata by SHA-256
and byte size. `metrics.json` includes a seeded 2,000-iteration class-stratified
bootstrap 95% interval for macro F1 and every class's precision, recall, and F1. The
report renders missing intervals as `Pending` rather than inventing bounds.

Missing inputs remain `Pending`. The generator never inserts zero, guessed, notebook,
or tutorial results. It records SHA-256 for every supplied evidence file.

A numeric before/after comparison is permitted only when `comparison.json` declares
`comparable=true`, a shared locked `evaluation_id`, one metric definition, and baseline
and candidate values. Otherwise the report explicitly suppresses both numbers. Clean,
OOD, field, latency, and user-study evidence remain separate sections.

Before publishing, review representative failures and confirm the model card, public
status pages, release version, dataset versions, and report hashes all refer to the same
immutable release. Do not use the report to claim definitive diagnosis, treatment
efficacy, or population-wide performance.
