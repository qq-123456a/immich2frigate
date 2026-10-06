# Operating plan

## Locked foundation

- Keep the existing three-person Immich/Frigate roster fixed; do not select new people.
- `prepare` scans up to 700 Immich face vectors per roster member. Apply Immich identity-safe filtering to the full pool before preview, quality scoring, or Frigate ranking. Ignore scene vectors.
- Require five strict, Teacher-confirmed candidates per person before producing a frozen rebuild plan. Strict profile: pinned Frigate 0.18 large/ArcFace, YuNet and LBF readback; target-face IoU ≥0.5; stored side ≥80 px; sharpness ≥250; color/exposure gates; eDifFIQA ≥0.30; absolute yaw/pitch/roll ≤15°; cross-person raw-cosine margin ≥0.10; same-source/checksum and cosine ≥0.98 duplicates rejected.
- Freeze upload bytes, stored-WebP bytes, embeddings, metrics, roster, service target, source hashes, model profile, and inventory. A changed source or runtime invalidates the plan.

## Deployment and rollback

1. From the immich2frigate repository root, build with `docker compose -f compose.runtime.yaml build`, then stop the scheduled sync with `docker compose -f compose.runtime.yaml stop immich2frigate-sync`. Building does not touch production data. Keep model assets read-only; mount the root of Frigate `model_cache` at `/models/frigate` with its `facedet/` child intact, and verify pinned hashes through profile startup.
2. Run `docker compose -f compose.runtime.yaml run --rm --no-deps immich2frigate-sync prepare --registry /var/lib/immich2frigate/identity-registry.json --state /var/lib/immich2frigate/sync-state.json --plan-dir /var/lib/immich2frigate/plans/foundation`; `prepare` creates the private plan directory. Review the full frozen plan and ensure all three identities have five eligible images.
3. Run `docker compose -f compose.runtime.yaml run --rm --no-deps immich2frigate-sync rebuild --registry /var/lib/immich2frigate/identity-registry.json --state /var/lib/immich2frigate/sync-state.json --plan-dir /var/lib/immich2frigate/plans/foundation --backup-dir /backups/foundation-unique-id --confirm-reset`. Rebuild writes sequentially and validates the exact stored crop and Teacher result after every registration.
4. In Frigate's existing runtime directory, change only `recognition_threshold` to `0.95` and `min_faces` to `3`, then run `docker compose restart frigate` once. From the immich2frigate repository root, run `docker compose -f compose.runtime.yaml up -d`.
5. Start the event validation watcher. Wait for at least 20 confirmed calibration events per person, zero conflicts, and agreement coverage ≥0.90. Only then generate an expansion proposal.

Rollback stops Frigate and restores its raw face-library files, sync state, identity registry, and Frigate config from one matching backup. Start Frigate after restore. Do not use registration API calls to rollback.

## Validation and expansion

- The Teacher is Immich's own face-recognition endpoint and indexed embeddings with source asset/checksum exclusion. Its neutral-border PNG preprocessing is versioned at 50% padding and value 127. Keep the Immich detector/recognizer threshold at 0.7; never lower it to force confirmation.
- Archive face crops privately by event. Calibration rows are used to simulate Frigate prototypes; holdout rows are never read for candidate scoring and never become Immich candidate sources.
- Propose at most two new Immich images per person per batch, never exceeding 30 registered images. Apply source uniqueness, strict quality, the 0.10 cross-person margin, Frigate class-center/density rank, and independent Teacher confirmation.
- Simulate all three prototypes together. Freeze only a batch that strictly improves calibration coverage without increasing conflicts. An empty proposal is a no-op.
- After apply, mark the batch pending. It becomes accepted only when all three people pass a fresh holdout gate with ≥20 confirmed events, zero conflicts, and ≥0.90 agreement. Any new conflict withdraws only the batch's added Frigate images, restores the previous source ledger and profile, and sets a new validation epoch at the withdrawal time; collect fresh events before proposing or accepting another batch.

## Release and privacy gates

- Python 3.12 CI runs synthetic tests with `.[compat,vision,database,curator,dev]`.
- No production names/IDs, face images, vectors, archives, reports, frozen plans, model weights, credentials, or backups enter Git.
- See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for pinned model sources and hashes. Weight license terms must be verified upstream before any redistribution.
