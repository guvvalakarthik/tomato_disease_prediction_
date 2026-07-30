# Data card and collection guide

No image dataset is committed to this repository. Generated manifests are safe to
version only after removing personal or precise-location information.

## Clean benchmark

- **Source:** original PlantVillage color images, filtered to ten tomato classes.
- **Provenance:** record a public source URL, immutable commit/revision, citation,
  license, download date, and archive checksum in the generated summary.
- **Known limitation:** controlled backgrounds and centered leaves can inflate
  performance and encourage background shortcuts.
- **Split policy:** 70% train, 15% validation, 15% locked clean test, stratified by
  class and grouped by original leaf. If official leaf groups are unavailable,
  perceptual groups are a disclosed fallback.
- **Leakage controls:** exact hash conflicts fail preparation; a group crossing splits
  fails both preparation and training.

The source folders map to stable IDs in `training/tomato_guard_ml/constants.py`.
Do not rename classes manually in the API or frontend.

## OOD evidence

Create two disjoint provenance-tracked sets:

1. **OOD calibration:** used only to select the rejection threshold.
2. **Locked OOD test:** used once to report false acceptance and AUROC/risk coverage.

Include non-tomato leaves, non-leaf objects, multiple leaves, blank frames, extreme
blur/exposure, partial leaves, screenshots, and unsupported file types. Respect each
source's license. Do not duplicate an OOD image across the two sets.

## Local field photographs

Target at least 300 expert-reviewed images and at least 20 per represented class.
Collect across phones, farms, backgrounds, lighting, distance, orientation, occlusion,
growth stage, and symptom severity. Avoid collecting faces, vehicle plates, addresses,
GPS coordinates, or other unnecessary personal data.

Required consent statement:

> I understand that this photograph and non-identifying capture metadata may be used
> to evaluate and improve an educational tomato leaf screening research project. I
> can request removal before the published dataset version is frozen.

Use anonymous reviewer IDs. Keep the private consent record outside Git; the manifest
contains only `consent_recorded=true`. Bucket location at a broad region level.

Required manifest columns are demonstrated in `field_manifest.example.csv`:

- sample ID and relative path;
- canonical class ID;
- anonymous expert reviewer and `confirmed` status;
- consent flag and capture date;
- device family, lighting, background, broad location bucket, and notes.

Run:

```bash
python -m training.tomato_guard_ml.manifest validate-field \
  --manifest data/field_manifest.csv
```

## Label protocol

1. Collector records symptoms and conditions without suggesting a class to the reviewer.
2. A qualified agriculture lecturer, extension worker, or plant pathologist reviews
   the original-resolution image and metadata.
3. The reviewer selects one supported class, `unsupported`, or `unresolved`.
4. Low-quality, unresolved, multi-condition, or unsupported samples remain valuable
   OOD evidence but are excluded from ten-class accuracy calculations.
5. A second reviewer adjudicates disagreements where possible; never force consensus.

## Versioning and privacy

Version manifests and reports by semantic dataset version and SHA-256. Keep images in
access-controlled storage. Define retention and withdrawal dates before collection.
Never upload field photos to application logs, analytics, issue trackers, or the
anonymous usability form.
