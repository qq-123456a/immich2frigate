# Third-party notices

## if-curator

- Project: https://github.com/ds-sebastian/if_curator
- Version: 0.4.0
- Commit: `7bf745c18e68282e1257dc4a99e0ea1c93511d44`
- License: MIT
- Source: https://github.com/ds-sebastian/if_curator/tree/7bf745c18e68282e1257dc4a99e0ea1c93511d44

The upstream source is installed as an optional pinned dependency and is not vendored in this repository. Its license and third-party notices apply to that dependency separately.

The upstream downloads face-model assets at runtime from [`NickM-27/facenet-onnx`](https://github.com/NickM-27/facenet-onnx). The repository declares Apache-2.0, but the release assets have not been independently confirmed to carry that license. This project does not redistribute those model files; verify the applicable license before mirroring or packaging any weights.

## Strict face-profile model assets

`FaceProfile` pins these external, read-only files by SHA-256. A mismatch stops preparation. These hashes identify the approved bytes; they do not establish the assets' copyright or license terms.

| Role | File | SHA-256 | Upstream reference |
| --- | --- | --- | --- |
| Frigate ArcFace | `arcface.onnx` | `ec639a0429b4819130d1405a2d3b38beaa4cc4a6c5bd9cf48b94fdf65461de83` | [Frigate 0.18.0 face-recognition source](https://github.com/blakeblackshear/frigate/tree/v0.18.0) and [facenet-onnx v1.0 assets](https://github.com/NickM-27/facenet-onnx/releases/tag/v1.0) |
| Frigate YuNet | `facedet.onnx` | `321aa5a6afabf7ecc46a3d06bfab2b579dc96eb5c3be7edd365fa04502ad9294` | [Frigate 0.18.0 face-recognition source](https://github.com/blakeblackshear/frigate/tree/v0.18.0) and [facenet-onnx v1.0 assets](https://github.com/NickM-27/facenet-onnx/releases/tag/v1.0) |
| Frigate LBF landmarks | `landmarkdet.yaml` | `70dd8b1657c42d1595d6bd13d97d932877b3bed54a95d3c4733a0f740d1fd66b` | [Frigate 0.18.0 face-recognition source](https://github.com/blakeblackshear/frigate/tree/v0.18.0) and [facenet-onnx v1.0 assets](https://github.com/NickM-27/facenet-onnx/releases/tag/v1.0) |
| Head pose | `mobilenetv2.onnx` | `1e902872868e483bd0e4f8f4a8ff2a4d61c2ccbca9dadf748e5479b5cc86a9e9` | [yakhyo/head-pose-estimation at commit `3ce191454012ffe146348e434fe287d1d6b6e708`](https://github.com/yakhyo/head-pose-estimation/tree/3ce191454012ffe146348e434fe287d1d6b6e708), using its [`onnx_inference.py`](https://github.com/yakhyo/head-pose-estimation/blob/3ce191454012ffe146348e434fe287d1d6b6e708/onnx_inference.py) preprocessing/inference reference |
| Face image quality (eDifFIQA) | `ediffiqa_tiny_jun2024.onnx` | `9426c899cc0f01665240cb7d9e7f98e18e24e456c178326c771a43da289bfc6a` | [opencv/face_image_quality_assessment_ediffiqa at revision `4fce5f6a5438b55a9f996a4e02164b63b849c587`](https://huggingface.co/opencv/face_image_quality_assessment_ediffiqa/tree/4fce5f6a5438b55a9f996a4e02164b63b849c587) |

Model weights are not vendored or redistributed here. Do not infer a weight license from its source-code repository or from Frigate's code license; review the upstream asset-specific terms before mirroring or packaging any file. The pose and eDifFIQA references are pinned to the source revisions used for this profile.

## Frigate

The source-derived compatibility routines in `src/immich2frigate/frigate018.py` and `src/immich2frigate/frigate_registration.py` are based on Frigate v0.18.0, commit `77a66e75c61862b048a07c1295877f4b31343504` (https://github.com/blakeblackshear/frigate/tree/v0.18.0). Copyright (c) 2026 Frigate, Inc. (Frigate™). Licensed under the MIT License:

> The MIT License
>
> Copyright (c) 2026 Frigate, Inc. (Frigate™)
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
