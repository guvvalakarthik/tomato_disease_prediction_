# Field evaluation protocol

## Objective

Measure domain shift on real local tomato-leaf photographs without contaminating
training, hyperparameter selection, calibration, or rejection thresholds.

## Before collection

1. Freeze collection consent, privacy, labeling, and withdrawal procedures.
2. Recruit a qualified expert reviewer and assign an anonymous reviewer ID.
3. Freeze model weights, class order, temperature, threshold, and clean-test report.
4. Register the desired sample count, class coverage, and analysis in the run record.

## Sampling

- Target 300 images with at least 20 confirmed examples for every represented class.
- Vary farm, phone family, lighting, leaf distance, background, angle, occlusion,
  growth stage, and severity.
- Include consecutive and hard cases; do not keep only visually obvious examples.
- Record unsupported, mixed-condition, and low-quality images for OOD/refusal analysis.
- Do not publish exact farm location or identifying content.

## Review

The expert reviews full-resolution originals without seeing model predictions. Each
sample becomes `confirmed`, `unresolved`, `unsupported`, or `excluded-quality`.
Disagreements are adjudicated by a second reviewer when available. Only confirmed
supported-class images enter class metrics.

## Locked analysis

Run `evaluate_field.py` once with the frozen model and metadata. Publish:

- count and acquisition coverage by class and capture condition;
- per-class precision, recall, F1 and support;
- macro F1 with a seeded 2,000-sample bootstrap 95% interval;
- balanced accuracy, confusion matrix, ECE, Brier score, and NLL;
- acceptance coverage, accepted-case accuracy, and uncertain rate;
- error slices by device, lighting, background, and severity when support permits;
- unsupported/OOD false acceptance separately;
- represented and not-externally-validated classes;
- representative CAMs and failure analysis with consented, de-identified images.

Never merge clean and field samples into one headline accuracy.

## Decision rule

Field-ready wording is permitted only when all field gates in `RELEASE_GATES.md` pass.
Otherwise publish the failure, retain the educational-prototype label, and create a
new training-data iteration. The locked field set remains a test set; do not move its
images into training. Collect a new future field test after retraining.
