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
python -m training.tomato_guard_ml.train --help
python -m training.tomato_guard_ml.evaluate --help
```

The dataset itself is not committed. Each run must preserve its manifest hash,
source revision, seed, configuration, history, model, and reports. Never use the
clean test or field partitions for model selection.
