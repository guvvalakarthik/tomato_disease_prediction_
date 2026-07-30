# Deployment

## Local compatibility deployment

The default Compose stack serves the tracked legacy HDF5 artifact so the existing app
remains runnable while new evidence is collected:

```bash
docker compose up --build
```

Frontend: `http://localhost:3000`; API docs: `http://localhost:8000/docs`.
The legacy image includes TensorFlow and is intentionally not the final free-tier image.

## Validated ONNX deployment

1. Complete the clean, OOD, field, and artifact gates.
2. Export `model.onnx`, `classifier_weights.npy`, and `metadata.json` into
   `model/release/`.
3. Run `python scripts/validate_release.py model/release`.
4. Build the lightweight image:

```bash
docker build -f api/Dockerfile.onnx -t tomatoguard-api:1.0.0 .
docker run --rm -p 8000:8000 \
  -e TOMATOGUARD_ALLOWED_ORIGINS=https://your-pages-domain.example \
  tomatoguard-api:1.0.0
```

Both `/health/live` and `/health/ready` must pass before deployment.

## Render API

`render.yaml` describes the current compatibility service. Connect the repository in
Render and set `TOMATOGUARD_ALLOWED_ORIGINS` to the exact HTTPS frontend origin. After
ONNX validation, switch its Dockerfile to `api/Dockerfile.onnx` before promoting the
field-ready release.

Free Render services sleep after inactivity and can take roughly one minute to wake.
This is acceptable for a portfolio demonstration, not a production SLA. The service
has no persistent disk and does not store uploads.

## Cloudflare Pages frontend

Create a Pages project with:

- root directory: `frontend`;
- build command: `npm ci && npm run build`;
- output directory: `dist`;
- build variable `VITE_API_URL`: the Render HTTPS API URL.

The included `wrangler.toml` supports CLI deployment after authentication. Add the
resulting Pages origin to the API's CORS allowlist and rerun the CORS smoke test.

## Post-deploy smoke tests

```bash
curl -fsS https://<api>/health/live
curl -fsS https://<api>/health/ready
curl -fsS -X POST "https://<api>/v1/predict?explain=true" \
  -F "file=@consented-smoke-test.jpg;type=image/jpeg"
```

Also verify an uncertain/OOD image, corrupt upload, wrong MIME, oversized upload,
mobile keyboard flow, model version, CAM display, and cold-start message.

## Rollback and monitoring

- Roll back the immutable container/model version together; never mix new metadata
  with old weights.
- Log request ID, status, latency, response class/uncertain state, and model version;
  never log image bytes, filename, or inferred personal data.
- Track readiness failures, 4xx validation counts, 5xx rate, uncertainty rate, and
  warm p50/p95 latency using host logs.
- An unexpected confidence or class-distribution shift triggers investigation, not
  automatic retraining.
