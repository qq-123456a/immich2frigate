# Third-party notices

## if-curator

- Project: https://github.com/ds-sebastian/if_curator
- Version: 0.4.0
- Commit: `7bf745c18e68282e1257dc4a99e0ea1c93511d44`
- License: MIT
- Source: https://github.com/ds-sebastian/if_curator/tree/7bf745c18e68282e1257dc4a99e0ea1c93511d44

The upstream source is installed as an optional pinned dependency and is not vendored in this repository. Its license and third-party notices apply to that dependency separately.

The upstream downloads face-model assets at runtime from [`NickM-27/facenet-onnx`](https://github.com/NickM-27/facenet-onnx). The repository declares Apache-2.0, but the release assets have not been independently confirmed to carry that license. This project does not redistribute those model files; verify the applicable license before mirroring or packaging any weights.

## Frigate

The initial compatibility primitives in `src/immich2frigate/frigate018.py` are derived from Frigate v0.18.0, commit `77a66e75c61862b048a07c1295877f4b31343504` (https://github.com/blakeblackshear/frigate/tree/v0.18.0). Copyright (c) 2026 Frigate, Inc. (Frigate™). Licensed under the MIT License:

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
