# TomatoGuard

TomatoGuard is an uncertainty-aware tomato leaf disease screening system. It
combines a reproducible TensorFlow training pipeline, calibrated rejection,
ONNX CPU inference, a validated FastAPI upload boundary, class-activation maps,
and a React interface that can decline to diagnose an image.

> **Current evidence status:** engineering rebuild complete; validated model
> release pending. The tracked `tomatoes.h5` is a legacy model trained on clean
> PlantVillage-style images. Its notebook recorded approximately 83.85% test
> accuracy using a flawed batch-level split. It is retained for compatibility,
> not presented as field-ready evidence. See [release gates](docs/RELEASE_GATES.md).

**Live portfolio deployment:** [TomatoGuard web app](https://tomatoguard-lake.vercel.app)
and [FastAPI documentation](https://tomatoguard-api.vercel.app/docs). The live app
serves the explicitly unvalidated `0.1.0-legacy` compatibility model.

## Why this is not a generic classifier demo

- Leaf-group-aware, deterministic manifests prevent duplicate leakage.
- Clean test, OOD test, and expert-reviewed local field evidence are separated.
- Temperature scaling and a locked rejection threshold replace raw softmax claims.
- Low-confidence uploads return `uncertain`, not a forced diagnosis.
- Obvious unusable inputs (flat/synthetic, blank, extreme exposure, or monochrome)
  are rejected before inference; semantic OOD performance remains a locked test gate.
- The API validates content bytes, dimensions, type, size, and decode safety.
- Versioned artifacts bind preprocessing, class order, metrics, thresholds, and hashes.
- ONNX inference and a linear CAM-compatible head support low-cost CPU deployment.
- CI tests the API contract, frontend behavior, coverage, containers, and release bundle.

## Architecture

```text
PlantVillage source ??> manifest + group split ??> TensorFlow training
                                                    ?
OOD calibration set ??> temperature + threshold ????
                                                    v
Field test set ????????> locked reports      model release bundle
                                                    ?
                                                    v
React ?? HTTPS upload ??> FastAPI validation ??> ONNX Runtime
                                ?                       ?
                                ??? uncertainty <?? probabilities + CAM
```

Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Local application

Prerequisites: Python 3.12, Node.js 22, and optionally Docker Desktop.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,legacy]"
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. Copy `.env.example` when overriding the model,
origin allowlist, or upload limits. The current legacy artifact requires the
`legacy` dependency group. A validated ONNX release needs only the base group.

Run the current compatibility stack with:

```powershell
docker compose up --build
```

## Reproducible ML workflow

Install the training environment on Colab, Kaggle, or a Python 3.12 GPU machine:

```bash
python -m pip install -r training/requirements.txt
```

1. Obtain the original color PlantVillage data and record its immutable source
   revision. The project supports the original ten folder names listed in
   `training/tomato_guard_ml/constants.py`.
2. If available, provide the source repository's leaf-group mapping. Otherwise,
   the manifest generator uses perceptual groups and reports that limitation.
3. Build the deterministic manifest before any shuffle, batching, or augmentation:

```bash
python -m training.tomato_guard_ml.manifest prepare \
  --dataset-root data/raw/PlantVillage \
  --output data/processed/plantvillage_manifest.csv \
  --leaf-groups data/raw/leaf_groups.csv \
  --source https://github.com/spMohanty/PlantVillage-Dataset \
  --revision <immutable-commit-sha>
```

4. Train the legacy-style baseline and the primary transfer-learning model:

```bash
python -m training.tomato_guard_ml.train \
  --architecture baseline \
  --manifest data/processed/plantvillage_manifest.csv \
  --dataset-root data/raw/PlantVillage \
  --output artifacts/baseline-seed-42

python -m training.tomato_guard_ml.train \
  --architecture mobilenetv3 \
  --manifest data/processed/plantvillage_manifest.csv \
  --dataset-root data/raw/PlantVillage \
  --output artifacts/mobilenetv3-seed-42
```

Repeat both architectures with three declared seeds. Select the final run using
validation evidence only.

5. Fit calibration, freeze the rejection threshold using a separate OOD
   calibration set, then open the clean and locked OOD test sets once:

```bash
python -m training.tomato_guard_ml.evaluate \
  --model artifacts/mobilenetv3-seed-42/model.keras \
  --manifest data/processed/plantvillage_manifest.csv \
  --dataset-root data/raw/PlantVillage \
  --ood-manifest data/processed/ood_calibration.csv \
  --ood-root data/raw/ood-calibration \
  --ood-test-manifest data/processed/ood_test.csv \
  --ood-test-root data/raw/ood-test \
  --output reports/generated/1.0.0
```

6. Validate the expert-reviewed field manifest and run external evaluation
   without refitting calibration or thresholds:

```bash
python -m training.tomato_guard_ml.manifest validate-field \
  --manifest data/field_manifest.csv

python -m training.tomato_guard_ml.evaluate_field \
  --model artifacts/mobilenetv3-seed-42/model.keras \
  --metadata reports/generated/1.0.0/metadata.json \
  --manifest data/field_manifest.csv \
  --field-root data/field \
  --output reports/generated/1.0.0/field_metrics.json
```

7. Export and validate the release bundle. Validation refuses artifacts that do
   not meet the declared clean, OOD, calibration, and field gates:

```bash
python -m training.tomato_guard_ml.export_onnx \
  --model artifacts/mobilenetv3-seed-42/model.keras \
  --metadata reports/generated/1.0.0/metadata.json \
  --metrics reports/generated/1.0.0/metrics.json \
  --field-metrics reports/generated/1.0.0/field_metrics.json \
  --output model/release

python scripts/validate_release.py model/release
```

## API contract

- `GET /health/live`: process liveness.
- `GET /health/ready`: loaded artifact version, SHA-256, and class count.
- `POST /v1/predict?explain=true`: multipart JPEG/PNG screening request.
- Errors use stable `code`, `message`, and `request_id` fields.
- Predictions return `predicted` or `uncertain`, top-three calibrated
  probabilities, model identity, an optional CAM, and a safety disclaimer.

Interactive OpenAPI documentation is available at `/docs` while the API runs.

## Tests and security checks

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=api --cov-report=term-missing
cd frontend
npm test
npm run build
npm audit --audit-level=high
```

CI also validates Compose, builds both current application containers, and only
enables the lightweight ONNX release job when the required artifacts exist.
Uploaded image bytes are processed in memory and are not logged or persisted.

## Evidence and safety

- [Model card](MODEL_CARD.md)
- [Data and field collection](data/README.md)
- [Field evaluation protocol](docs/FIELD_EVALUATION_PROTOCOL.md)
- [Deployment guide](docs/DEPLOYMENT.md)
- [Release and resume gates](docs/RELEASE_GATES.md)

TomatoGuard is an educational screening tool, not a definitive diagnosis. It
does not prescribe pesticides or replace laboratory testing, extension advice,
or a qualified plant-health professional.
