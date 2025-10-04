"""Utilities for discovering and downloading MODIS granules via NASA Earthdata."""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import earthaccess  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    earthaccess = None


@dataclass(frozen=True)
class BoundingBox:
    """Geospatial bounding box in WGS84 coordinates (lat/lon)."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def to_list(self) -> List[float]:
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


MODIS_COLLECTIONS = {
    "mod06": "MOD06_L2",
    "myd06": "MYD06_L2",
}


class EarthdataUnavailable(RuntimeError):
    """Raised when optional earthaccess dependency is missing."""


class EarthdataClient:
    """Thin wrapper around ``earthaccess`` to simplify authentication and download."""

    def __init__(self) -> None:
        if earthaccess is None:  # pragma: no cover - import guard
            raise EarthdataUnavailable(
                "earthaccess is not installed. Install it with `pip install earthaccess`."
            )
        self._client = earthaccess

    def login(
        self,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        netrc: Optional[Path] = None,
    ) -> None:
        """Authenticate against NASA Earthdata.

        Strategy selection priority:
        1. Explicit username/password arguments.
        2. Environment variables (EARTHDATA_USERNAME / EARTHDATA_PASSWORD).
        3. ``~/.netrc`` file or supplied ``netrc`` path.
        """

        strategy = "environment"
        kwargs = {}
        if username and password:
            strategy = "credentials"
            kwargs.update({"username": username, "password": password})
        elif netrc:
            strategy = "netrc"
            kwargs["netrc_path"] = str(netrc)
        self._client.login(strategy=strategy, **kwargs)

    def search(
        self,
        *,
        collection: str,
        start_date: _dt.datetime,
        end_date: _dt.datetime,
        bbox: BoundingBox,
        limit: Optional[int] = None,
    ) -> Iterable[dict]:
        """Yield metadata dictionaries matching the MODIS query."""

        granules = self._client.search_data(
            short_name=collection,
            temporal=(start_date, end_date),
            bounding_box=bbox.to_list(),
        )
        for idx, granule in enumerate(granules):
            if limit is not None and idx >= limit:
                break
            yield granule

    def download(self, granules: Iterable[dict], target_dir: Path) -> List[Path]:
        """Download each granule into ``target_dir``; returns downloaded file paths."""

        target_dir.mkdir(parents=True, exist_ok=True)
        return list(self._client.download(granules, target_dir=target_dir, threads=4))


def discover_and_download(
    *,
    output_dir: Path,
    bbox: BoundingBox,
    start_date: _dt.datetime,
    end_date: _dt.datetime,
    collection_key: str = "mod06",
    limit: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    netrc: Optional[Path] = None,
) -> List[Path]:
    """Authenticate, search, and download MODIS granules for the specified period."""

    collection = MODIS_COLLECTIONS.get(collection_key.lower())
    if collection is None:
        raise ValueError(f"Unknown MODIS collection key '{collection_key}'.")

    client = EarthdataClient()
    client.login(username=username, password=password, netrc=netrc)
    granules = client.search(
        collection=collection,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        limit=limit,
    )
    return client.download(granules, output_dir)


def parse_iso_date(date_str: str) -> _dt.datetime:
    """Parse ISO date strings like ``2023-07-01`` to ``datetime``."""

    return _dt.datetime.fromisoformat(date_str)


def parse_bbox(coords: Tuple[float, float, float, float]) -> BoundingBox:
    """Helper to quickly build a ``BoundingBox`` from a tuple."""

    return BoundingBox(*coords)


__all__ = [
    "BoundingBox",
    "EarthdataClient",
    "EarthdataUnavailable",
    "discover_and_download",
    "parse_bbox",
    "parse_iso_date",
]
