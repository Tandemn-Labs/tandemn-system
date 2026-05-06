from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class StorageUri:
    scheme: str
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"{self.scheme}://{self.bucket}/{self.key}" if self.key else f"{self.scheme}://{self.bucket}"


def parse_storage_uri(uri: str, allowed_schemes: set[str] | None = None) -> StorageUri:
    parsed = urlparse(uri)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"storage URI must include scheme and bucket/container: {uri}")
    if allowed_schemes is not None and parsed.scheme not in allowed_schemes:
        allowed = ", ".join(sorted(allowed_schemes))
        raise ValueError(f"storage URI scheme must be one of {allowed}. Got: {uri}")
    return StorageUri(
        scheme=parsed.scheme,
        bucket=parsed.netloc,
        key=parsed.path.lstrip("/"),
    )


def is_storage_uri(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)
