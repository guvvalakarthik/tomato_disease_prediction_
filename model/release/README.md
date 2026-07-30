# Validated model release placeholder

The lightweight deployment image expects this directory to contain files created by:

```text
python -m training.tomato_guard_ml.export_onnx \
  --model artifacts/<run>/model.keras \
  --metadata reports/generated/<run>/metadata.json \
  --metrics reports/generated/<run>/metrics.json \
  --field-metrics reports/generated/<run>/field_metrics.json \
  --output model/release
```

Required release files are `model.onnx`, `classifier_weights.npy`, `metrics.json`,
`field_metrics.json`, and `metadata.json`. They are absent until a model passes the clean,
field, calibration, and OOD gates in `docs/RELEASE_GATES.md`. The legacy HDF5
model remains available for local compatibility but is not a validated release.

After a candidate passes `scripts/validate_release.py`, promote it atomically:

```text
python scripts/promote_release.py \
  --candidate model/release \
  --releases-root model/releases \
  --version 1.0.0 \
  --source-commit <full-40-character-git-sha>
```

Promotion refuses mutable version names, mismatched metadata versions, failed gates,
and overwrites. It produces `release-manifest.json` with source provenance, sizes,
and SHA-256 attestations. Deployment must consume a versioned directory, never an
unversioned candidate.
