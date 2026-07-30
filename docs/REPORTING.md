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
