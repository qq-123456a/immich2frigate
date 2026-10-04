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
- Frigate 0.18 exposes labels and registered image filenames, not a stable person UUID. `identity_registry.py` keeps a local Immich-person UUID -> project UUID -> current Frigate-label binding and reports safe reconciliation states. Because the face library is exclusively managed by this integration, the first sync automatically binds a normalized same-name label with registered faces; subsequent renames follow the Immich UUID. Successful enrollments should persist the binding immediately after Frigate confirms the first face. The fixed write client supports Frigate's native rename endpoint, which moves the label folder and retains its face images. A label changed before any binding exists cannot be safely attributed from Frigate state alone.
- An in-memory selector adapter reuses the pinned if-curator diversity/identity selection, checks vector shape/finiteness/non-zero norm, and ignores its legacy 0.17 score. It does not prove the vectors came from Frigate's verified 0.18 model.
- Revalidate the exact final image bytes that would be uploaded. Keep selection and the fixed-route Frigate write client separate.
- `dry_run.py` builds a deterministic in-memory JSON review plan from selected upload bytes, records source/upload/simulated registered-image hashes, and never uploads, deletes, renames, or records successful sync. Entries remain explicitly marked as model-unverified. It skips labels that already exist; name reconciliation is handled separately by `identity_sync.py`.

**Current gate:** the dry-run builder, fixed-route Frigate write client, in-memory preview crop encoder, Immich v3 person-thumbnail reader, and person-name reconciliation planner/executor are implemented. Name-only sync remains a separate explicit API; new enrollment is not yet orchestrated, and this API does not run on a schedule. ArcFace embedding generation, exact runtime model assets/OpenCV build, end-to-end image equivalence, and isolated integration/recovery verification remain open. No credentials, person identifiers, or image data are included in the repository.

## C. Controlled enrollment trial and additive sync

- Persist each intended operation before sending it, with instance identity, person/asset IDs, target, and image SHA-256.
- Upload sequentially. If the remote result is ambiguous, record `UNKNOWN_RESULT` / `NEEDS_REVIEW`; do not blindly retry or delete.
- Back up with SQLite's online backup API and verify the backup. Never auto-restore over later manual changes.
- A one-time, user-authorized destructive trial may back up the live Frigate database and face library, remove registered face images by explicit inventory IDs, and enroll one explicitly named Immich person. This does not authorize routine deletion behavior.
- Routine name sync automatically binds a normalized same-name Frigate label with registered faces on the first run, under the assumption that the library is exclusively integration-managed. Once bound, rename only when the destination label is absent and the operation is journaled and verified. Do not replace or delete labels that do not have an unambiguous same-name match; a label changed before the first binding needs review.
- Before a rename run, take and verify a database/face-library backup using the caller-provided callback. On ambiguous outcomes, reconcile against a fresh Frigate inventory and the operation journal; do not issue a second PUT without review.
- If a multi-person apply stops partway through, create a new plan from fresh Immich, registry, and Frigate state. The old plan is intentionally stale after any remote change.

## D. Scheduled operation

- Reconcile named people and asset ownership every run; handle missing people, merges, reassignments, and manual deletions conservatively.
- Add a scheduler and full enrollment-operation orchestration only after isolated integration and recovery behavior are verified. Name-only sync already has a per-registry lock, JSONL journal, and recovery handling.

## Release gates

Before any production face-library trial, demonstrate in an isolated environment that repeated plans are stable, uploaded bytes match validated bytes, ambiguous outcomes stop safely, manual files are untouched, and backup restoration works. Public CI must use synthetic data only.
