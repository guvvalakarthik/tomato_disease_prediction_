# Architecture

## Training and evidence flow

```mermaid
flowchart LR
  PV[Versioned PlantVillage source] --> M[Hash and leaf-group manifest]
  M --> TR[Train partition]
  M --> VA[Validation partition]
  M --> CT[Locked clean test]
  TR --> B[Baseline CNN]
  TR --> MV3[MobileNetV3-Small]
  VA --> SEL[Model selection]
  SEL --> CAL[Temperature scaling]
  OODC[OOD calibration] --> TH[Rejection threshold]
  CAL --> TH
  CT --> REP[Clean evaluation]
  TH --> REP
  FIELD[Expert-reviewed field test] --> FREP[Separate field evaluation]
  TH --> FREP
  REP --> GATE[Release gate]
  FREP --> GATE
  GATE --> ONNX[Versioned ONNX bundle]
```

The test and field branches have no path back into training, selection, calibration,
or threshold fitting.

## Serving flow

```mermaid
sequenceDiagram
  participant User
  participant React
  participant API as FastAPI
  participant Runtime as ONNX Runtime
  User->>React: Select JPEG/PNG
  React->>React: Check MIME and 10 MB limit
  React->>API: POST /v1/predict?explain=true
  API->>API: Limit bytes, verify format, dimensions, decode, RGB, EXIF
  API->>Runtime: 224x224 float32 batch
  Runtime-->>API: logits and convolutional feature map
  API->>API: temperature, top-3, threshold, CAM
  API-->>React: predicted or uncertain + model identity
  React-->>User: Evidence, disclaimer, and safe next step
```

## Boundaries

- The browser provides convenience validation; FastAPI is the security boundary.
- Model metadata is the single class/preprocessing/threshold contract.
- ONNX Runtime is the deployment runtime; TensorFlow is training-only after migration.
- Images are in-memory request data and are never persistent application state.
- Readiness fails when artifact, metadata, checksum, or runtime loading fails; liveness
  remains available for orchestration diagnostics.
- The legacy HDF5 adapter exists only until release 1.0 passes evidence gates.
