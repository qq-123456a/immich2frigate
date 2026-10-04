# immich2frigate

Safely prepare and synchronize a small set of useful face training images from Immich to Frigate's native face recognition library.

The project reuses the image selection work in [`if-curator`](https://github.com/ds-sebastian/if_curator), adds a compatibility layer for a narrowly verified Frigate version, and keeps any write operations behind a separate synchronization layer.

## Status

This repository is at the initial implementation stage. The upstream baseline is pinned in [`UPSTREAM.md`](UPSTREAM.md). The first supported target is Frigate **0.18.0 with the `large` face model**. No Frigate write or delete operation is implemented yet.

The first usable release is intended to be additive and conservative:

- Read Immich people and candidate assets.
- Prepare and validate the exact image bytes intended for Frigate.
- Produce a reviewable plan before any write.
- Treat ambiguous remote results as requiring review; never blindly retry or delete.
- Keep credentials in environment variables or a runtime secret mount. The application must not save them to a config file, log, manifest, or report.

Do not use this project with a production face library until the compatibility and recovery requirements in [`docs/plan.md`](docs/plan.md) have been verified in an isolated environment.

## Development setup

Python 3.12 or later is required. The optional `curator` extra installs the pinned upstream selector and its inference dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[curator]"
```

For local development, copy `.env.example` to `.env` and fill in credentials locally. `.env` and runtime data are ignored by Git. Prefer Docker secrets or environment injection for deployments.

The application reads Immich and Frigate credentials from the process environment only. It does not call if-curator's interactive CLI, which has its own plaintext local connection-file behavior.

## Security and privacy

Never commit API keys, tokens, real names-to-person-ID mappings, face images, embeddings, production API responses, local manifests, databases, or backups. Public CI and checked-in examples must use synthetic data only. Open a private security report for a suspected vulnerability; see [`SECURITY.md`](SECURITY.md).

The selector and synchronization layers are code boundaries, not OS-level privilege isolation. Frigate's internal API may grant administrator-equivalent access, so only the synchronization entry point may ever receive write capability.

## License

This project's original code is MIT licensed. Upstream and model licenses are separate; see [`UPSTREAM.md`](UPSTREAM.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
