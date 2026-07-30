# TomatoGuard edge inference

The validated Keras candidate can be exported as dynamic-range, float16, or full-int8
TFLite. Int8 export requires representative samples from the locked training partition;
field, validation, test, and OOD images are never calibration inputs.

```bash
python -m training.tomato_guard_ml.export_tflite \
  --model artifacts/runs/<run>/model.keras \
  --metadata model/release/metadata.json \
  --output artifacts/edge/tomatoguard-int8.tflite \
  --quantization int8 \
  --manifest artifacts/manifests/plantvillage.csv \
  --dataset-root /data/plantvillage
```

The adjacent `.metadata.json` contains the model version, artifact hashes, input and
output quantization, class contract, frozen calibration temperature, and rejection
threshold. Run TensorFlow/TFLite parity on a fixed sample before device testing.

## Raspberry Pi demonstration

On a 64-bit Raspberry Pi, install a compatible `tflite-runtime`, NumPy, and Pillow,
then copy the `.tflite` and `.metadata.json` files without changing their hashes:

```bash
python edge/raspberry_pi/infer.py \
  --model tomatoguard-int8.tflite \
  --metadata tomatoguard-int8.metadata.json \
  --image consented-test-leaf.jpg \
  --warmup 10 --runs 100 \
  --benchmark-output raspberry-pi-benchmark.json
```

The CLI performs EXIF correction, RGB conversion, quantization, inference,
temperature-scaled softmax, unknown-image rejection, and top-three output. The benchmark
reports warm p50/p95, model size, platform, processor, and whether decode/resize time is
included.

## Release rule

Do not claim mobile or Raspberry Pi performance until a real-device benchmark and
TensorFlow/TFLite parity report are attached to the same immutable model version.
`device_benchmark_completed` therefore defaults to `false`. Android LiteRT can consume
the same artifact and metadata contract, but no Android performance claim is made here.
