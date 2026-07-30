# TomatoGuard model card

## Model status

**Release status: not yet validated.** The repository currently serves a legacy
Keras HDF5 artifact (`0.1.0-legacy`) for compatibility. It must not be described
as calibrated, field-ready, or production-grade. `model/release/` remains gated
until a new artifact passes all criteria in `docs/RELEASE_GATES.md`.

## Intended use

- Educational screening of a clear photograph containing one tomato leaf.
- Demonstration of reproducible computer vision, uncertainty handling,
  explainability, API safety, and ML deployment.
- Support for nine named tomato disease/pest classes plus healthy.

The model is not intended for definitive diagnosis, pesticide prescription,
yield prediction, unsupported crops, laboratory replacement, or autonomous farm
decisions. An `uncertain` response is an expected safety behavior.

## Supported labels

Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider
Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, and Healthy. Stable IDs
and order are versioned in each release's `metadata.json`.

## Training design for release 1.0

- ImageNet-pretrained MobileNetV3-Small, 224?224 RGB input.
- Field-like rotation, crop, translation, contrast, and horizontal-flip augmentation.
- Global average pooling and a linear logits head for CAM-compatible explanations.
- Initial frozen-backbone phase followed by partial fine-tuning.
- Early stopping and learning-rate reduction based only on validation loss.
- Three declared seeds and comparison against the reconstructed small-CNN baseline.
- Temperature scaling fitted on validation logits.
- Maximum-softmax rejection threshold selected using validation and OOD calibration data.
- TensorFlow training artifact exported to ONNX; parity and checksums are release gates.

## Data

The clean source is the color tomato subset of PlantVillage. It contains controlled
backgrounds and is not representative of field conditions. Splits are deterministic,
stratified, and leaf-group-aware. Exact hashes, perceptual groups, source revision,
class counts, and split membership are stored in the generated manifest.

The external field set must contain consented local photographs with expert-confirmed
labels. It is never used for training, model selection, temperature fitting, or
threshold selection in release 1.0. See `data/README.md`.

## Metrics to publish

No placeholder number is a release metric. The evaluation command generates:

- accuracy and balanced accuracy;
- macro and weighted precision, recall, and F1;
- per-class precision, recall, F1, support, and two confusion matrices;
- NLL, Brier score, 15-bin ECE, and a reliability diagram;
- validation precision/coverage and OOD acceptance at the frozen threshold;
- separate clean and field reports, with bootstrap confidence intervals for field F1;
- latency, model size, artifact parity, and represented/missing field classes.

## Known risks and mitigations

- **Background shortcut:** field tests and CAM review reveal non-leaf attention.
- **Domain shift:** clean and field results are never merged into one score.
- **Overconfidence:** temperature scaling plus refusal below a locked threshold.
- **Novel conditions:** OOD tests include other leaves, objects, low-quality images,
  and invalid files; rejection does not guarantee novelty detection.
- **Label ambiguity:** field labels require expert confirmation and disagreements are
  excluded or marked unresolved.
- **Treatment harm:** the UI provides only cautious general next steps and directs
  users to qualified local advice.
- **Class coverage:** a missing field class is explicitly "not externally validated."

## Privacy and monitoring

Uploaded bytes are processed in memory and are not persisted or logged. Structured
logs contain request ID, path, status, latency, and model version only. Any future
feedback-image collection requires separate consent, retention policy, and expert
relabeling.

## Ownership and versioning

Every release bundle contains `model.onnx`, `classifier_weights.npy`,
`metadata.json`, `metrics.json`, hashes, training commit, data-manifest hash, class
order, preprocessing, calibration, threshold, dependency versions, and creation date.
Semantic versions change whenever weights, preprocessing, labels, or thresholds change.
