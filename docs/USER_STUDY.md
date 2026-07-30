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
- short free-text improvement suggestion.

Success requires at least 80% unassisted end-to-end completion. Do not optimize for
trust alone; appropriate skepticism and correct uncertainty interpretation matter.

## Privacy

Use participant codes, not names. Do not collect participant images, faces, farm
addresses, phone numbers, or uploaded leaf files. Obtain study consent and allow
withdrawal before aggregation.

## Report template

Publish participant mix, dates, tasks, completion rates, rating distributions, major
misunderstandings, accessibility issues, changes made, and unresolved findings. Keep
the status `pending` in `RELEASE_GATES.md` until the anonymized report exists.
