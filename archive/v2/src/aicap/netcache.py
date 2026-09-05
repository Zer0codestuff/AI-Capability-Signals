"""Content-addressed download cache with recorded provenance.

Design notes
------------
Every byte the pipeline reads from the network is written to ``data/raw/`` and recorded in a
sidecar ``.meta.json`` holding the URL, the fetch timestamp, the HTTP status, the byte count and
a SHA-256 digest. Analyses therefore always run against a hashed artifact, and a reviewer can
verify that two runs saw the same input.

Failure behaviour is deliberately loud. The previous version caught broad exceptions per source
and continued, so a run could "succeed" while silently missing a source. Here a fetch failure
raises :class:`FetchError` unless the caller has explicitly opted into a cached-only mode.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from .config import RAW

USER_AGENT = "ai-capability-signals/2.0 (reproducible public-data research)"

#: Transient HTTP statuses worth retrying. Anything else is a real answer from the server and
#: should surface immediately rather than being retried into a timeout.
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    """Raised when a required artifact is neither cached nor fetchable."""


@dataclass
class Artifact:
    """A cached byte stream plus its provenance record."""

    path: Path
    url: str
    sha256: str
    bytes: int
    fetched_at: str
    from_cache: bool
    source_id: str

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def read_json(self) -> Any:
        return json.loads(self.read_text())

    def as_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "url": self.url,
            "path": str(self.path.relative_to(RAW.parents[1])),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "fetched_at": self.fetched_at,
            "served_from_cache": self.from_cache,
        }


@dataclass
class Fetcher:
    """Fetches URLs into a content-addressed cache.

    Parameters
    ----------
    offline:
        Never touch the network. Missing artifacts raise :class:`FetchError`.
    refresh:
        Ignore cached copies and re-download.
    """

    offline: bool = False
    refresh: bool = False
    timeout: int = 120
    max_attempts: int = 4
    artifacts: list[Artifact] = field(default_factory=list)
    _session: requests.Session | None = None

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": USER_AGENT})
        return self._session

    def fetch(self, source_id: str, url: str, relative_path: str) -> Artifact:
        target = RAW / relative_path
        meta_path = target.with_suffix(target.suffix + ".meta.json")

        if target.exists() and not self.refresh:
            artifact = self._from_cache(source_id, url, target, meta_path)
            self.artifacts.append(artifact)
            return artifact

        if self.offline:
            raise FetchError(
                f"{source_id}: {relative_path} is not cached and offline mode is enabled. "
                f"Run without --offline to populate the cache from {url}."
            )

        payload = self._download(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(payload)
        tmp.replace(target)

        artifact = Artifact(
            path=target,
            url=url,
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
            fetched_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
            from_cache=False,
            source_id=source_id,
        )
        meta_path.write_text(json.dumps(artifact.as_record(), indent=2), encoding="utf-8")
        self.artifacts.append(artifact)
        return artifact

    def _from_cache(self, source_id: str, url: str, target: Path, meta_path: Path) -> Artifact:
        digest = sha256_file(target)
        fetched_at = ""
        if meta_path.exists():
            try:
                fetched_at = str(json.loads(meta_path.read_text(encoding="utf-8")).get("fetched_at", ""))
            except json.JSONDecodeError:
                fetched_at = ""
        return Artifact(
            path=target,
            url=url,
            sha256=digest,
            bytes=target.stat().st_size,
            fetched_at=fetched_at,
            from_cache=True,
            source_id=source_id,
        )

    def _download(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:  # network-level failure
                last_error = exc
            else:
                if response.status_code == 200:
                    return response.content
                last_error = FetchError(f"HTTP {response.status_code} for {url}")
                if response.status_code not in RETRY_STATUSES:
                    break
            if attempt < self.max_attempts:
                time.sleep(2**attempt)
        raise FetchError(f"Failed to fetch {url} after {self.max_attempts} attempts: {last_error}")

    def provenance_records(self) -> Iterator[dict[str, Any]]:
        seen: set[tuple[str, str]] = set()
        for artifact in self.artifacts:
            key = (artifact.source_id, artifact.url)
            if key in seen:
                continue
            seen.add(key)
            yield artifact.as_record()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
