"""Runtime-only application settings; credentials are never written to disk."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


def _required_url(name: str, environ: dict[str, str]) -> str:
    value = environ.get(name, "").strip().rstrip("/")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be an absolute http(s) URL")
    return value


@dataclass(frozen=True, slots=True)
class ImmichSettings:
    """Immich read credentials loaded only by the client that needs them."""

    immich_url: str
    immich_api_key: str = field(repr=False)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> ImmichSettings:
        env = os.environ if environ is None else environ
        immich_key = env.get("IMMICH_API_KEY", "").strip()
        if not immich_key:
            raise ValueError("IMMICH_API_KEY is required")
        return cls(
            immich_url=_required_url("IMMICH_URL", env),
            immich_api_key=immich_key,
        )


@dataclass(frozen=True, slots=True)
class FrigateSettings:
    """Frigate API URL. This client is read-only and uses the internal API port."""

    frigate_url: str

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> FrigateSettings:
        env = os.environ if environ is None else environ
        return cls(
            frigate_url=_required_url("FRIGATE_URL", env),
        )
