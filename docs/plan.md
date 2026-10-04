# Implementation plan

## 1. Immich source of truth

- Use Immich person assignment as the candidate boundary.
- Read persisted face_search and smart_search embeddings through a dedicated read-only PostgreSQL account.
- Do not add project-owned pose, lighting, expression, scene-label, quality, or identity-confidence models.
- Force PostgreSQL transactions read-only and keep credentials runtime-only.

## 2. Representative selection

- Target exactly 30 successfully registered images per named person.
- Normalize face and scene spaces independently.
- Use deterministic greedy facility-location coverage across the combined vector space.
- Keep an ordered spare pool for local crop failures without lowering the 30-image target.

## 3. Destructive rebuild gate

Before deleting any registered Frigate face data, enumerate all named Immich people, require at least 30 vector-backed candidates each, build the complete selection plan, verify Frigate 0.18.0/large/admin, and back up the registered library.

After the gate passes, delete registered faces from a fresh inventory, verify the library is empty, upload sequentially, stop immediately on ambiguous remote writes, and verify exactly 30 images for every completed person. The Frigate train staging directory is cleared separately during the one-time acceptance reset.

## 4. Closed-loop validation

- Group observations by Frigate person track rather than every frame.
- Keep a holdout benchmark that never participates in training.
- Send selected CCTV faces through Immich's own recognition path.
- Classify only agreement, Frigate miss, and disagreement; surface disagreements for review.
- Feed useful non-benchmark misses into a later representative-selection round.

## 5. Release gates

- Public tests use synthetic data only.
- CI passes on Python 3.12.
- No secrets, real face data, embeddings, mappings, manifests, reports, or backups enter Git.
- Production is a one-shot Docker job on the existing trusted Windows + Docker Desktop host.
- Production reset requires runtime access to Immich API, read-only Immich PostgreSQL, and Frigate internal API.
