# immich2frigate

Build a compact Frigate face-training library from faces that Immich has already assigned to known people.

## Core design

Immich is the teacher. Frigate is the student. The project does not classify pose, lighting, expression, scene labels, quality, or invent another face-confidence score.

For each named Immich person, the selector reuses Immich face_search and smart_search embeddings and chooses representatives with deterministic coverage maximisation across both vector spaces. The production target is exactly 30 successfully registered Frigate training images per person.

## Rebuild flow

Before any deletion, the program requires at least 30 vector-backed candidates for every named person, builds the full plan, verifies Frigate 0.18.0/large, and backs up the registered face library. It then clears registered faces, registers exactly 30 per person, verifies counts, and persists the private identity registry.

Run the destructive rebuild with the explicit --confirm-reset flag. Runtime settings come only from environment variables: IMMICH_URL, IMMICH_API_KEY, IMMICH_DATABASE_URL (dedicated read-only PostgreSQL user), and FRIGATE_URL.

Production data, API keys, DB credentials, person mappings, face images, embeddings, reports, manifests, and backups must never be committed.

## Development

Python 3.12+. Install .[compat,vision,database,dev] and run python -m pytest -q. Public CI uses synthetic data only.

## Closed-loop direction

The feedback stage uses real Frigate CCTV tracks as holdout observations and Immich as the independent teacher. Useful samples are those where Immich resolves a known person but Frigate misses or disagrees. Benchmark samples stay isolated from training. The feedback layer must reuse Immich recognition configuration rather than creating another project confidence model.

See docs/plan.md for release gates and SECURITY.md for security requirements.
