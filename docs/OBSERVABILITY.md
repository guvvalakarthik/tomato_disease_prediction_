# Privacy-safe observability

`GET /metrics` exposes bounded Prometheus counters and histograms for HTTP status,
route latency, prediction/uncertain outcomes, and canonical top-class distribution.
It never stores or exports image bytes, filenames, request IDs, IP addresses, user
agents, precise locations, free text, or confidence values tied to an individual.

Recommended alerts:

- readiness failure for five consecutive minutes;
- 5xx rate above 1% over ten minutes;
- warm `/v1/predict` p95 above two seconds without explanations;
- uncertainty rate changing by more than 15 percentage points after 100 predictions;
- Jensen-Shannon class-distribution divergence above 0.10 after 100 predictions.

Export only aggregate windows to `scripts/check_drift.py`:

```json
{
  "sample_count": 500,
  "uncertain_rate": 0.22,
  "class_distribution": {
    "tomato_healthy": 180,
    "tomato_early_blight": 120
  }
}
```

```bash
python scripts/check_drift.py \
  --baseline monitoring/baseline.json \
  --current monitoring/current-24h.json \
  --output reports/generated/drift-24h.json
```

A drift alert opens an investigation. It never triggers automatic retraining or changes
thresholds. Review data quality, seasonality, device mix, field failures, and consented
samples before creating a new versioned training dataset.
