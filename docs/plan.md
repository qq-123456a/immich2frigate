# Implementation plan

## A. Fix the upstream baseline

- Pin `if-curator` v0.4.0 to the reviewed commit and preserve its MIT notice.
- Reproduce upstream behavior before adding compatibility changes.
- Record dependency and model provenance; do not claim model weights may be redistributed until their licenses are checked.

## B. Read-only Frigate compatibility

- Target Frigate 0.18.0 with the `large` model first.
- Validate preprocessing, crop selection, feature extraction, class-center aggregation, blur penalty, score rounding, and unknown handling against the exact target version.
- Revalidate the exact final image bytes that would be uploaded. Keep selection and any future Frigate write client separate.
- Emit a reviewable dry-run plan. Dry-run may not upload, delete, rename, or record a successful sync.

## C. Controlled additive sync

- Persist each intended operation before sending it, with instance identity, person/asset IDs, target, and image SHA-256.
- Upload sequentially. If the remote result is ambiguous, record `UNKNOWN_RESULT` / `NEEDS_REVIEW`; do not blindly retry or delete.
- Back up with SQLite's online backup API and verify the backup. Never auto-restore over later manual changes.
- Refuse to take ownership of manual Frigate files. Do not rename, replace, or delete in the first writable version.

## D. Scheduled operation

- Reconcile named people and asset ownership every run; handle missing people, merges, reassignments, and manual deletions conservatively.
- Add locking, bounded work, auditable status, and a scheduler only after isolated integration and recovery behavior are verified.

## Release gates

Before any production face-library trial, demonstrate in an isolated environment that repeated plans are stable, uploaded bytes match validated bytes, ambiguous outcomes stop safely, manual files are untouched, and backup restoration works. Public CI must use synthetic data only.
