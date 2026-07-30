# Usability study protocol

Run this only after a validated model and public demo are available.

## Participants and tasks

Recruit at least ten farmers, agriculture students, extension workers, or plant-health
reviewers. Do not claim representativeness from this convenience sample.

Ask each participant to:

1. Explain what the tool can and cannot diagnose from the landing page.
2. Upload a valid leaf photograph and interpret the top classes and confidence.
3. Explain what an `uncertain` result means and choose an appropriate next step.
4. Locate model version, explanation limitation, and expert-consultation guidance.
5. Recover from one invalid or oversized upload.

## Measurements

- unassisted completion for each task;
- completion time and observed errors;
- whether confidence is mistaken for certainty;
- whether CAM is mistaken for proof;
- whether the participant would seek expert confirmation;
- five-point usefulness, clarity, trust, and refusal-understanding ratings;
- bounded issue tags for upload, result, confidence, uncertainty, attention-map, or accessibility problems.

Success requires at least 80% unassisted end-to-end completion. Do not optimize for
trust alone; appropriate skepticism and correct uncertainty interpretation matter.

## Privacy

Use participant codes, not names. Do not collect participant images, faces, farm
addresses, phone numbers, or uploaded leaf files. Obtain study consent and allow
withdrawal before aggregation.

## Report template


## Collection and aggregate export

The result screen includes an optional consent checkbox and categorical form submitted
to `POST /v1/feedback`. The API does not store images, filenames, network identifiers,
contact information, or free text. Use a protected PostgreSQL URL with
`TOMATOGUARD_REQUIRE_DURABLE_FEEDBACK=true` in production. Keep any local SQLite file
on encrypted, access-controlled storage and never commit it.

```bash
python scripts/export_user_study.py \
  --database "$TOMATOGUARD_FEEDBACK_DATABASE_URL" \
  --output reports/generated/user-study.json \
  --minimum-participants 20 \
  --baseline-version 0.1.0-legacy \
  --candidate-version 1.0.0 \
  --minimum-per-version 10 \
  --primary-metric task_completed \
  --bootstrap-iterations 2000 \
  --seed 42
```

Preregister one primary outcome before recruitment. Improvement is reported only when
its candidate-minus-baseline bootstrap 95% interval is entirely above zero. Counts are
explicitly consented responses, not unique people, because the privacy-preserving form
does not store participant identifiers. Publish null or negative findings; do not switch
the primary metric after seeing results.
Publish participant mix, dates, tasks, completion rates, rating distributions, major
misunderstandings, accessibility issues, changes made, and unresolved findings. Keep
the status `pending` in `RELEASE_GATES.md` until the anonymized report exists.
