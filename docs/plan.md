# Implementation plan

## A. Fix the upstream baseline

- Pin `if-curator` v0.4.0 to the reviewed commit and preserve its MIT notice.
- Reproduce upstream behavior before adding compatibility changes.
- Record dependency and model provenance; do not claim model weights may be redistributed until their licenses are checked.

## B. Read-only Frigate compatibility

- Target Frigate 0.18.0 with the `large` model first.
- Validate preprocessing, crop selection, feature extraction, class-center aggregation, blur penalty, score rounding, and unknown handling against the exact target version.
- Source-derived primitives now cover ArcFace preprocessing, float32 class-center aggregation, profile version gating, confidence/blur scoring, Frigate's 1080px YuNet input scaling/largest-face crop/WebP-100 registration transform, LBF eye alignment, and Laplacian blur measurement. Synthetic regression tests cover these transforms. The detector and landmark models must still be the exact verified Frigate assets before image-equivalence can be claimed.
- A GET-only Frigate adapter checks the exact version/model, verifies the internal API's administrator profile and enabled face recognition, reads existing face names/filenames, and retrieves registered face images using validated, URL-quoted path segments. It rejects redirects, unsafe path components, oversized responses, and unexpected schema.
- A read-only Immich adapter uses only the pinned upstream people/search/face-box/image methods. It returns minimal metadata and in-memory BGR previews; it does not use the upstream CLI, persistent face cache, or export functions.
- An in-memory selector adapter reuses the pinned if-curator diversity/identity selection while validating Frigate embedding shapes and ignoring its legacy 0.17 score.
- Revalidate the exact final image bytes that would be uploaded. Keep selection and any future Frigate write client separate.
- Emit a reviewable dry-run plan. Dry-run may not upload, delete, rename, or record a successful sync.

**Current gate:** there is not yet a usable sync plan. Candidate retrieval, service inventory, model-injected registration transforms, and the selector wrapper are implemented, but ArcFace embedding generation, exact runtime model assets/OpenCV build, end-to-end image equivalence, and a persisted/exportable review plan remain open. Integration/recovery verification remains open. No live Immich or Frigate instance has been contacted by this work.

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
