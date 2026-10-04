# Implementation plan

## 1. Immich source of truth

- Use Immich person assignment as the candidate boundary.
- Read persisted face_search and smart_search embeddings through a dedicated read-only PostgreSQL account.
- Do not add project-owned pose, expression, scene-label, or identity-confidence models.
- Force PostgreSQL transactions read-only and keep credentials runtime-only.

## 2. Foundation-first selection

- Require at least 5 foundation-quality images per named person before destructive reset.
- Apply Frigate-aligned non-ML quality gates: clear enough by Laplacian sharpness, color rather than effectively grayscale, reasonable exposure, and sufficient face area.
- Among foundation-quality candidates, prefer faces closest to the Immich face-vector centroid as a conservative proxy for typical, front-facing identity views.
- Do not claim that centroid distance is a new identity confidence score.

## 3. Adaptive expansion

- Five images are the baseline, not a quota floor for every available photo.
- After the first five, add an image only when its combined Immich face and smart-search vectors are sufficiently novel relative to the already selected set.
- Stop naturally when remaining images are too similar, even if the person has 20, 30, or hundreds of source photos.
- Cap the initial import at 30 images per person.
- A person with many near-duplicate photos may therefore receive only 5-7 initial training images.

## 4. Destructive rebuild gate

Before deleting any registered Frigate face data, build the complete adaptive plan for all named people, verify every person has the five-image foundation, verify Frigate 0.18.0/large/admin, and back up the registered library.

After the gate passes, delete registered faces from a fresh inventory, verify the library is empty, upload sequentially, stop immediately on ambiguous remote writes, and verify each person's exact adaptive target count.

## 5. Closed-loop validation

- Group observations by Frigate person track rather than every frame.
- Keep a holdout benchmark that never participates in training.
- Send selected CCTV faces through Immich's own recognition path.
- Add clear lower-performing or disagreement samples only when they add new useful conditions.
- Keep expansion gradual to avoid over-fitting.

## 6. Release gates

- Public tests use synthetic data only.
- CI passes on Python 3.12.
- No secrets, real face data, embeddings, mappings, manifests, reports, or backups enter Git.
- Production reset still requires runtime access to Immich API, read-only Immich PostgreSQL, and Frigate internal API.
