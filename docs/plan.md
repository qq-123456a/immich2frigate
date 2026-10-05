# Implementation plan

## 1. Immich source of truth

- Use Immich person assignment as the candidate boundary.
- Read persisted `face_search` embeddings through a dedicated read-only PostgreSQL account.
- Read `smart_search` embeddings when present, but do not require them for a face to remain eligible.
- Do not add project-owned pose, expression, scene-label, or identity-confidence models.
- Force PostgreSQL transactions read-only and keep credentials runtime-only.

## 2. Foundation-first selection

- Require at least 5 foundation-quality images per named person before destructive reset.
- Apply Frigate-aligned non-ML quality gates: clear enough by Laplacian sharpness, color rather than effectively grayscale, reasonable exposure, sufficient face area, and enough surrounding crop context for a stable seed image.
- Build a conservative identity-safe pool from Immich's persisted face embeddings before rewarding novelty. Use a robust medoid envelope and nearest-neighbor isolation check only to reject clear intra-person outliers; do not claim this is a second identity classifier.
- Among safe foundation-quality candidates, prefer medoid-central and locally dense faces.
- Avoid near-duplicate foundation faces when safe alternatives exist, but keep the five-image minimum by falling back to the most central safe candidates when the source library is genuinely repetitive.

## 3. Adaptive expansion

- Five images are the baseline, not a quota floor for every available photo.
- After the first five, consider only candidates that remain inside the conservative Immich face-vector identity core.
- Add an image only when it contributes sufficient novelty relative to the selected set.
- Use combined face + Smart Search scene distance when both candidates have scene embeddings; otherwise fall back to face-vector distance.
- Stop naturally when remaining images are too similar, even if the person has 20, 30, or hundreds of source photos.
- Cap the initial import at 30 images per person.
- A person with many near-duplicate photos may therefore receive only 5-7 initial training images.

## 4. Destructive rebuild gate

Before deleting any registered Frigate face data, build the complete adaptive plan for all named people, verify every person has five safe foundation-quality images, verify Frigate 0.18.0/large/admin, and back up the registered library.

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
