# immich2frigate

Build a compact Frigate face-training library from faces that Immich has already assigned to known people.

## Core design

Immich is the teacher. Frigate is the student. The project does not add its own pose, expression, scene-label, or identity-confidence model.

The first goal is a small, strong foundation rather than a fixed image quota. Each named person must have at least five foundation-quality images before a destructive rebuild is allowed. Foundation candidates must be clear, color, reasonably exposed, and large enough to be useful. The selector then prefers the most typical faces in Immich's own face-vector space.

After the first five, extra images are optional. They are added only when the combined Immich face and smart-search vectors show meaningfully new coverage. Similar images from the same situation are intentionally skipped. Thirty images is an upper bound for the initial import, not a target that must be filled.

This follows Frigate's guidance to start with a few clear, front-facing photos and expand slowly with useful variation. Frigate also warns that diversity matters more than volume and that low-quality or overly similar training images can reduce accuracy.

## Rebuild flow

Before any deletion, the program builds the complete adaptive plan for every named person, verifies that each person has at least five foundation-quality images, verifies Frigate 0.18.0/large, and backs up the registered face library. It then clears registered faces, uploads each person's selected adaptive set, verifies the exact final count for each person, and persists the private identity registry.

Runtime settings come only from environment variables: IMMICH_URL, IMMICH_API_KEY, IMMICH_DATABASE_URL using a dedicated read-only PostgreSQL user, and FRIGATE_URL.

Production data, API keys, DB credentials, person mappings, face images, embeddings, reports, manifests, and backups must never be committed.

## Development

Python 3.12+. Install .[compat,vision,database,dev] and run python -m pytest -q. Public CI uses synthetic data only.

## Closed-loop direction

After the foundation is live, real Frigate CCTV tracks become the feedback source. Immich remains the independent teacher. Useful additions are clear samples where Frigate misses or disagrees and the sample adds new conditions. Benchmark samples remain isolated from training.

See docs/plan.md for release gates and SECURITY.md for security requirements.
