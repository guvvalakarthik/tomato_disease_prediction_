# Release and resume gates

## Engineering gates

- Backend tests pass with at least 80% branch-aware coverage.
- Frontend component tests, production build, and high-severity dependency audit pass.
- Docker Compose validates; containers run as non-root where supported.
- Upload validation covers wrong MIME, corrupt, empty, oversized, grayscale, and RGBA.
- TensorFlow and ONNX probabilities meet declared parity tolerance on a fixed sample.
- Artifact checksums, class order, input shape, outputs, and CAM weights validate.
- Production CORS is a precise allowlist and readiness returns the loaded model identity.

## ML gates

- Deterministic leaf-group manifest has no exact/group leakage.
- Three seeds are reported for baseline and candidate; selection uses validation only.
- Clean locked-test macro F1 is at least 0.90 and every class recall is at least 0.80.
- Clean 15-bin ECE is at most 0.05.
- Locked OOD false acceptance is at most 10% at the frozen threshold.
- At least 80% of clean test samples remain accepted.
- Confusion matrices, per-class report, calibration diagram, risk/coverage, model size,
  and warm CPU latency are published.

## Field gates

- Consent and expert-label protocol is documented and validated.
- Target 300 field photos and at least 20 per represented class.
- Field macro F1 is at least 0.70 and field ECE is at most 0.10.
- Missing classes are labeled `not externally validated` in the model card and UI docs.
- Field photos were not used to fit weights, temperature, or threshold.
- Representative failures and CAMs are reviewed for background shortcuts.

## Product gates

- Warm CPU p95 is at most 2 seconds without CAM and 4 seconds with CAM.
- At least ten farmers, agriculture students, or domain reviewers attempt the flow.
- At least 80% complete upload-to-interpretation without help.
- Findings and resulting UI/model changes are recorded without personal data.
- The live demo, OpenAPI schema, model card, data card, and reports are public.

## Current status

| Gate | Status | Evidence |
| --- | --- | --- |
| Hardened API and schema | Implemented | Automated API tests |
| Frontend uncertainty flow | Implemented | Component tests and production build |
| Reproducible ML commands | Implemented | Manifest/train/evaluate/export modules |
| Validated ONNX release | Pending | `model/release/` intentionally has no artifact |
| Clean release metrics | Pending | Requires dataset and GPU run |
| OOD evaluation | Pending | Requires curated calibration and locked sets |
| Expert field evaluation | Pending | Requires consented collection and review |
| Public deployment | Pending | Requires validated artifact and account connection |
| User study | Pending | Requires deployed validated demo |

The project should remain off the resume as a field-ready system while any pending
evidence gate remains. Engineering work may be described as work in progress, but the
legacy 83.85% notebook result must not be used as the headline metric.
