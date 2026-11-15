from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astroquery.ipac.irsa import Irsa

# Per IRSA's SPHEREx Data Access guide: QR2 is no longer searchable via SIA2.
# DR1 and QR3 are the searchable releases; deep fields have separate collections.
COLLECTIONS = {
    "QR3 — Wide survey": "spherex_qr3",
    "QR3 — Deep survey": "spherex_qr3_deep",
    "DR1 — Wide survey": "spherex_dr1",
    "DR1 — Deep survey": "spherex_dr1_deep",
}
CACHE_DIR = Path(__file__).resolve().parent / "data" / "spherex_cache"


def _python_value(value: Any) -> Any:
    """Convert Astropy table cells to plain Python values."""
    if np.ma.is_masked(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def search_observations(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    collections: Sequence[str],
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Query IRSA SIA2 for SPHEREx image MEFs at a sky position."""
    ra_deg, dec_deg, radius_deg = float(ra_deg), float(dec_deg), float(radius_deg)
    if not 0.0 <= ra_deg <= 360.0:
        raise ValueError("RA must be between 0 and 360 degrees.")
    if not -90.0 <= dec_deg <= 90.0:
        raise ValueError("Dec must be between -90 and 90 degrees.")
    if not 0.0 < radius_deg <= 180.0:
        raise ValueError("Search radius must be greater than 0 and at most 180 degrees.")
    if not collections:
        raise ValueError("Select at least one SPHEREx collection.")
    if any(collection not in COLLECTIONS.values() for collection in collections):
        raise ValueError("Unsupported collection. QR2 is not searchable via SIA2.")

    position = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    observations: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    errors: list[str] = []

    for collection in collections:
        try:
            # Astroquery maps this to IRSA's SIA2 endpoint and COLLECTION parameter.
            table = Irsa.query_sia(
                pos=(position, radius_deg * u.deg), collection=collection
            )
        except Exception as exc:
            errors.append(f"{collection}: {exc}")
            continue

        columns = {str(name).lower(): name for name in table.colnames}
        url_column = columns.get("access_url")
        if url_column is None:
            errors.append(f"{collection}: IRSA response has no access_url column")
            continue

        for row in table:
            record = {
                str(name): _python_value(row[name])
                for name in table.colnames
            }
            url = record.get(str(url_column))
            if not isinstance(url, str) or urlparse(url).scheme.lower() not in {"http", "https"}:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            record["access_url"] = url
            record["collection"] = collection
            # Use SIA metadata when present; don't assume one fixed column spelling.
            lowered = {str(key).lower(): value for key, value in record.items()}
            record["epoch"] = next(
                (lowered[key] for key in ("mjd", "mjd_obs", "t_min", "date_obs", "obs_date")
                 if lowered.get(key) not in (None, "")),
                "Epoch not provided",
            )
            observations.append(record)
            if len(observations) >= int(limit):
                break
        if len(observations) >= int(limit):
            break

    if not observations and errors:
        raise RuntimeError("IRSA SIA2 query failed. " + " | ".join(errors))
    observations.sort(key=lambda item: str(item.get("epoch", "")))
    return observations


def download_observation(observation: dict[str, Any]) -> Path:
    """Download and validate an observation FITS file into a URL-keyed cache."""
    url = observation.get("access_url")
    if not isinstance(url, str) or urlparse(url).scheme.lower() not in {"http", "https"}:
        raise ValueError("The IRSA result does not contain a valid access_url.")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (hashlib.sha256(url.encode("utf-8")).hexdigest() + ".fits")
    if path.is_file() and path.stat().st_size:
        try:
            with fits.open(path, memmap=False) as hdus:
                hdus.verify("exception")
                if any(hdu.data is not None for hdu in hdus):
                    return path
        except Exception:
            path.unlink(missing_ok=True)

    temporary: Path | None = None
    try:
        with requests.get(
            url, stream=True, timeout=(15, 180),
            headers={"User-Agent": "SPHEREx-Explorer/1.0"},
        ) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                dir=CACHE_DIR, suffix=".download", delete=False
            ) as handle:
                temporary = Path(handle.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        with fits.open(temporary, memmap=False) as hdus:
            hdus.verify("exception")
            if not any(hdu.data is not None for hdu in hdus):
                raise ValueError("Downloaded FITS file contains no data.")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def read_fits_image(path: str | Path) -> tuple[np.ndarray, WCS]:
    """Read the first numeric 2-D celestial image from a SPHEREx FITS MEF."""
    with fits.open(path, memmap=False) as hdus:
        for hdu in hdus:
            if hdu.data is None or np.ndim(hdu.data) != 2:
                continue
            if not np.issubdtype(hdu.data.dtype, np.number):
                continue
            wcs = WCS(hdu.header).celestial
            if not wcs.has_celestial:
                continue
            image = np.array(hdu.data, dtype=np.float64, copy=True)
            image[~np.isfinite(image)] = np.nan
            return image, wcs
    raise ValueError(
        "No 2-D image extension with celestial WCS was found in this FITS MEF."
    )
