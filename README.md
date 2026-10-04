# immich2frigate

Safely prepare and synchronize a small set of useful face training images from Immich to Frigate's native face recognition library.

The project plans to reuse the image selection work in [`if-curator`](https://github.com/ds-sebastian/if_curator), add a compatibility layer for a narrowly verified Frigate version, and keep any write operations behind a separate synchronization layer.

## Status

The upstream baseline is pinned in [`UPSTREAM.md`](UPSTREAM.md). Frigate **0.18.0 with the `large` face model** is the first target. [`frigate018.py`](src/immich2frigate/frigate018.py) and [`frigate_registration.py`](src/immich2frigate/frigate_registration.py) contain source-derived preprocessing, detection/crop/re-encode, alignment, and scoring primitives. [`selection.py`](src/immich2frigate/selection.py) wraps if-curator's in-memory diversity selector while ignoring its Frigate 0.17 scoring. These still require exact Frigate model assets and end-to-end verification before use. [`immich_client.py`](src/immich2frigate/immich_client.py) wraps the pinned upstream candidate APIs; [`immich_face_reader.py`](src/immich2frigate/immich_face_reader.py) reads Immich v3's paginated people response and a person's feature-face thumbnail without requiring full-asset access. [`upload_image.py`](src/immich2frigate/upload_image.py) scales an Immich face box to its in-memory preview, checks aspect ratio, and encodes a bounded crop. [`frigate_client.py`](src/immich2frigate/frigate_client.py) exposes bounded reads; [`frigate_write.py`](src/immich2frigate/frigate_write.py) adds fixed Frigate 0.18 create/register/delete/recognize routes. Deletes require explicit image IDs and reject `train`. The write adapter is not a sync workflow by itself: callers must verify target identity, back up the live database and face files, persist an operation record, and confirm the resulting inventory after every operation.

The first usable release is intended to be conservative:

- Read Immich people, candidate face boxes, and previews without invoking upstream cache or export paths.
- Read Frigate version, administrator profile, face-recognition settings, existing face-library filenames, and registered face images through the allow-listed client.
- Prepare and validate the exact image bytes intended for Frigate.
- Produce a deterministic JSON dry-run plan with upload and simulated registered-image hashes before any write. Until model assets are verified, proposed entries are explicitly marked unverified and the plan cannot enable writes. The plan is returned in memory; it contains person/asset IDs and names, so keep any exported copy private.
- Treat ambiguous remote results as requiring review; never blindly retry or delete. A one-time destructive reset for local project acceptance has been exercised against a local instance, with a verified private backup and post-registration recognition check. This is not routine sync behavior.
- Keep credentials in environment variables or a runtime secret mount. The application must not save them to a config file, log, manifest, or report.

Do not use this project with a production face library until the compatibility and recovery requirements in [`docs/plan.md`](docs/plan.md) have been verified in an isolated environment.

## Development setup

Python 3.12 or later is required. The optional `curator` extra installs the pinned upstream selector and its inference dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[curator]"
```

To work only on the compatibility primitives and their regression suite:

```powershell
python -m pip install -e ".[compat,vision,dev]"
python -m pytest
```

The `vision` extra provides OpenCV for local image-transform simulation. Frigate's exact YuNet and LBF model assets are not bundled or downloaded by this project; callers must supply verified models explicitly.

For local development, copy `.env.example` to `.env` and fill in credentials locally. `.env`, if-curator's `.immich_config.json`, `.if_cache/`, `frigate_train/`, and other runtime data are ignored by Git. Prefer Docker secrets or environment injection for deployments. Keep HTTP traffic confined to an isolated trusted network; use HTTPS across untrusted networks.

The application reads the Immich API key and both service URLs from the process environment only. Its Frigate read-only client uses the internal API port and does not support username/password login. It does not call if-curator's interactive CLI, which has its own plaintext local connection-file behavior.

## Security and privacy

Never commit API keys, tokens, real names-to-person-ID mappings, face images, embeddings, production API responses, local manifests, databases, or backups. Public CI and checked-in examples must use synthetic data only. Open a private security report for a suspected vulnerability; see [`SECURITY.md`](SECURITY.md).

The selector and synchronization layers are code boundaries, not OS-level privilege isolation. Frigate's internal API grants administrator-equivalent access; keep port 5000 on a trusted private network. The Frigate adapter exposes only bounded GET operations, confirms the admin role, and does not log in to or support the externally authenticated port.

## License

This project's original code is MIT licensed. Upstream and model licenses are separate; see [`UPSTREAM.md`](UPSTREAM.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
