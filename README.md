# immich2frigate

Build a compact Frigate face-training library from faces that Immich has already assigned to known people.

## Core design

Immich is the teacher. Frigate is the student. The project does not add its own pose, expression, scene-label, or identity-confidence model.

The first goal is a small, strong foundation rather than a fixed image quota. Each named person must have at least five foundation-quality images before a destructive rebuild is allowed. Foundation candidates must be clear, color, reasonably exposed, large enough to be useful.

The selector uses Immich's persisted face embeddings to build a conservative identity core before rewarding diversity. Isolated or clearly off-cluster faces are excluded from automatic training selection. Within that safe core, the first five prefer medoid-central, locally dense faces while avoiding near-duplicate foundation images when alternatives exist.

After the first five, extra images are optional. They are added only when the Immich face vector, plus the Smart Search scene vector when available, shows meaningfully new coverage. Smart Search embeddings are helpful but not required: candidates without one fall back to face-vector novelty rather than disappearing from the pool. Thirty images is an upper bound for the initial import, not a target that must be filled.

For bounded memory use, selection evaluates at most 2,000 candidates per person, sampled across the sorted timeline while retaining foundation-quality candidates first.

This follows Frigate's guidance to start with a few clear, front-facing photos and expand slowly with useful variation. Frigate also warns that diversity matters more than volume and that low-quality or overly similar training images can reduce accuracy.

## Rebuild flow

Before any deletion, the program builds the complete adaptive plan for every named person, verifies that each person has at least five safe foundation-quality images, verifies Frigate 0.18.0/large, and backs up the registered face library. It then clears registered faces, uploads each person's selected adaptive set, verifies the exact final count for each person, and persists the private identity registry.

Runtime settings come only from environment variables: IMMICH_URL, IMMICH_API_KEY, IMMICH_DATABASE_URL using a dedicated read-only PostgreSQL user, and FRIGATE_URL.

Production data, API keys, DB credentials, person mappings, face images, embeddings, reports, manifests, and backups must never be committed.

## Development

Python 3.12+. Install .[compat,vision,database,dev] and run python -m pytest -q. Public CI uses synthetic data only.

## Closed-loop direction

After the foundation is live, real Frigate CCTV tracks become the feedback source. Immich remains the independent teacher. Useful additions are clear samples where Frigate misses or disagrees and the sample adds new conditions. Benchmark samples remain isolated from training.

See docs/plan.md for release gates and SECURITY.md for security requirements.
