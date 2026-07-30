# TomatoGuard training

The notebook is a repository-relative walkthrough with no stored outputs. The
command-line workflow is the source of truth and must be used to produce a
release model.

Use the command-line modules in `tomato_guard_ml/`:

- `manifest.py`: scan, hash, group, split, and version the ten tomato classes.
- `train.py`: run the CNN baseline or MobileNetV3-Small transfer learning.
- `evaluate.py`: fit calibration on validation data and evaluate the locked test.
- `evaluate_field.py`: evaluate expert-reviewed field photos without refitting.
- `export_onnx.py`: produce the CPU release and explanation weights.

All paths are command-line parameters. Configuration lives in `config.json`,
and exact dependencies are pinned in `requirements.txt`.

From the repository root:

```bash
python -m pip install -r training/requirements.txt
python -m training.tomato_guard_ml.manifest --help

Install export dependencies after the training environment. TensorFlow 2.20 requires
modern protobuf while tf2onnx 1.16 declares an older protobuf range, so the converter
is intentionally installed without replacing TensorFlow's working dependency set:

```bash
python -m pip install -r training/requirements-export.txt
python -m pip install --no-deps tf2onnx==1.16.1
```

python -m training.tomato_guard_ml.train --help
python -m training.tomato_guard_ml.evaluate --help
```

The dataset itself is not committed. Each run must preserve its manifest hash,
source revision, seed, configuration, history, model, and reports. Never use the
clean test or field partitions for model selection.

## Reproducible experiment run

Use a new output directory for every experiment; training refuses to overwrite an
existing non-empty run. A publishable run must be produced from a clean commit.

```bash
python -m training.tomato_guard_ml.train \
  --manifest artifacts/manifests/plantvillage.csv \
  --dataset-root /data/plantvillage \
  --config training/config.json \
  --output artifacts/runs/mobilenetv3-seed42 \
  --run-name mobilenetv3-seed42
python -m training.tomato_guard_ml.validate_run \
  artifacts/runs/mobilenetv3-seed42 \
  --manifest artifacts/manifests/plantvillage.csv
```

Each run records the Git revision and dirty state, Python and dependency versions,
dataset/config hashes, timing, seed, and artifact hashes.

## Release 1.0 experiment matrix

A release candidate must be selected using validation loss across at least three
MobileNetV3 seeds. The baseline is trained on the same manifest for comparison but
cannot win release selection merely through clean-test performance.

```bash
python scripts/run_training_matrix.py \
  --manifest artifacts/manifests/plantvillage.csv \
  --dataset-root /data/plantvillage \
  --config training/config.json \
  --output artifacts/runs/release-1 \
  --seeds 17 42 73
```

The script refuses mixed manifests, fewer than three seeds, reused output folders,
or runs that cannot prove the test partition remained locked. Actual training is
pending until the external dataset is supplied; no model or metric is fabricated.
