FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN python -m pip install --no-cache-dir ".[curator,database,vision]"

ENTRYPOINT ["immich2frigate"]
