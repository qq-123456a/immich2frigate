FROM python:3.12-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir \
    "numpy>=2.2.6" \
    "pillow>=12.1.0" \
    "requests>=2.32.5" \
    "psycopg[binary]>=3.2" \
    "opencv-contrib-python-headless==4.11.0.86" \
    "onnxruntime==1.24.4"

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-deps --no-cache-dir .
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["immich2frigate"]
