# Upstream baseline

The selector baseline is [`ds-sebastian/if_curator`](https://github.com/ds-sebastian/if_curator), tag `v0.4.0`, resolved to commit:

```text
7bf745c18e68282e1257dc4a99e0ea1c93511d44
```

The tag and commit were read directly from the upstream Git remote on 2026-10-04. The project is marked Alpha upstream and is licensed MIT. This repository does not copy or modify upstream source; the pinned optional dependency is declared in `pyproject.toml`.

The upstream dependency graph includes inference and image-processing packages. Keep it optional until the selector integration is needed, and preserve the upstream OpenCV/ONNX Runtime constraints when resolving it. Do not infer model-weight redistribution rights from the source-code license; audit model provenance and licenses before packaging or redistributing weights.

The integration must not call upstream credential-saving paths. Supply Immich credentials only at runtime and ensure neither configuration nor logs persist them.
