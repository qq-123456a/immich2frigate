# immich2frigate

Build a verified Frigate face library from Immich-assigned faces. Immich remains the independent identity teacher; Frigate remains the classifier. Production operations use frozen plans and private state, never Git-tracked face data.

## Fixed foundation and strict face profile

The library is a locked three-person roster with exactly five images per person after a rebuild. `prepare` scans the bounded Immich candidate pool (up to 700 rows per person) and applies Immich face-identity safety before preview fetches; scene embeddings do not affect eligibility or ranking. A candidate is accepted only after the actual Frigate upload and stored-WebP readback simulate the Frigate 0.18 large/ArcFace, YuNet, and LBF path. Profile `frigate-0.18.0-strict-face-v1` pins model hashes and requires a target-face IoU of at least 0.5, stored dimensions of at least 80 px, sharpness at least 250, usable color/exposure, eDifFIQA at least 0.30, and absolute yaw/pitch/roll no greater than 15 degrees. Ambiguous, invalid, or out-of-bounds detections fail closed.

Frigate embeddings are ranked by robust class center, local density, and quality. Same-source/checksum and cosine-at-least-0.98 duplicates are suppressed. Each proposed face must retain a raw-cosine margin of at least 0.10 over other identities, and Immich independently confirms it with the source asset/checksum excluded. No person is added or reranked; a rebuild never proceeds with fewer than five eligible images per person.

The quality models are external, read-only assets. Mount the root of Frigate's `model_cache` at `/models/frigate` (it contains `facedet/`) and the two quality ONNX files under `/models/quality`; the runtime compose mounts both read-only. Their exact SHA-256 values are locked in the profile and listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Install OpenCV 4.11 and onnxruntime 1.24.4 through the `vision` extra; runtime rejects version or model-hash mismatches.

## Independent validation and gated expansion

`validate --watch` archives Frigate train crops privately, groups them by event, and asks Immich to confirm clear samples independently. Calibration events and holdout events are deterministically separated; holdout images never train or score proposals. The watcher reloads the current roster/profile as it runs. Each person needs at least 20 confirmed calibration events, zero conflicts, and at least 0.90 agreement before an expansion can be proposed.

An expansion proposes at most two Immich source images per person per batch, with a 30-image library cap. Candidate sources must pass the same strict face profile, deduplication, cross-person margin, and independent Teacher check. The complete three-person Frigate prototype update is simulated against calibration events; the batch is frozen only if coverage strictly improves and conflicts do not increase. Empty proposals are safe no-ops. After apply, added images remain pending until all three people pass a fresh holdout gate of at least 20 confirmed events, zero conflicts, and at least 0.90 agreement. A new conflict withdraws that batch's added Frigate files, restores the prior source ledger and profile, and starts a new validation epoch at the withdrawal time; collect fresh events before proposing or accepting another batch.

## Deployment sequence

From the immich2frigate repository root, build the image first; this does not touch production services or data. Then stop the existing scheduled sync before operating on the library. Keep the face models mounted read-only and provide secrets/state only through runtime environment and persistent private directories.

```sh
docker compose -f compose.runtime.yaml build
docker compose -f compose.runtime.yaml stop immich2frigate-sync
docker compose -f compose.runtime.yaml run --rm --no-deps immich2frigate-sync prepare \
  --registry /var/lib/immich2frigate/identity-registry.json \
  --state /var/lib/immich2frigate/sync-state.json \
  --plan-dir /var/lib/immich2frigate/plans/foundation
docker compose -f compose.runtime.yaml run --rm --no-deps immich2frigate-sync rebuild \
  --registry /var/lib/immich2frigate/identity-registry.json \
  --state /var/lib/immich2frigate/sync-state.json \
  --plan-dir /var/lib/immich2frigate/plans/foundation \
  --backup-dir /backups/foundation-unique-id --confirm-reset
```

Review the frozen plan and fresh backup before `rebuild`; `prepare` creates its private plan directory. In Frigate's existing runtime directory, set only `face_recognition.recognition_threshold: 0.95` and `min_faces: 3`, then run `docker compose restart frigate` once. From the immich2frigate repository root, run `docker compose -f compose.runtime.yaml up -d`. Do not use the UI to register these faces. The validation watcher starts with the runtime; wait for all three calibration gates. Generate a proposal with `docker compose -f compose.runtime.yaml run --rm --no-deps immich2frigate-sync sync --registry /var/lib/immich2frigate/identity-registry.json --state /var/lib/immich2frigate/sync-state.json --propose-only --plan-dir /var/lib/immich2frigate/proposals`, review it, then apply it with `docker compose -f compose.runtime.yaml run --rm --no-deps immich2frigate-sync sync --registry /var/lib/immich2frigate/identity-registry.json --state /var/lib/immich2frigate/sync-state.json --apply-plan /var/lib/immich2frigate/proposals/replace-with-plan-id`. Keep scheduled sync proposal-only unless deliberately applying a reviewed plan.

For rollback, stop Frigate, restore the raw face-library files plus the matching sync state, identity registry, and Frigate config from the same backup, then start Frigate. Never rollback by registering images through the API.

## Development

Python 3.12+. Install with `python -m pip install -e ".[compat,vision,database,curator,dev]"` and run `python -m pytest -q`. CI uses synthetic data only. Never commit credentials, names or IDs, face images, embeddings, model weights, validation archives/reports, frozen plans, or backups.

See [docs/plan.md](docs/plan.md) for the operating gates and [SECURITY.md](SECURITY.md) for data-handling requirements.
