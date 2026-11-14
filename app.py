from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import requests
import streamlit as st

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from reproject import reproject_interp


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEFAULT_COLLECTIONS = ("spherex_dr1", "spherex_qr3")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="SPHEREx Explorer",
    page_icon="🌌",
    layout="wide"
)


# ============================================================
# 2. INITIALISATION DES VARIABLES DE SESSION
# ============================================================

if "search_done" not in st.session_state:
    st.session_state.search_done = False

if "observations" not in st.session_state:
    st.session_state.observations = []

if "search_error" not in st.session_state:
    st.session_state.search_error = None


# ============================================================
# 3. FONCTIONS DE TRAITEMENT DES DONNÉES
# ============================================================

def _plain(value: Any) -> Any:
    """Convertit les valeurs NumPy et Astropy en valeurs Python."""

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


def _get(record: Mapping[str, Any], *names: str) -> Any:
    """Recherche une information dans les métadonnées."""

    fields = {
        str(key).lower(): value
        for key, value in record.items()
    }

    for name in names:

        value = fields.get(name.lower())

        if value is not None and str(value).strip().lower() not in {
            "", "--", "nan", "none"
        }:
            return value

    return None


def _sort_key(record: Mapping[str, Any]) -> tuple:
    """Trie les observations selon leur date."""

    mjd = _get(record, "mjd", "mjd_obs", "t_min")

    try:
        if mjd is not None:
            return (
                0,
                float(mjd),
                str(record.get("observation_id", ""))
            )
    except (TypeError, ValueError):
        pass

    date = _get(
        record,
        "time",
        "date_obs",
        "obs_date",
        "t_min"
    )

    return (
        1,
        str(date or ""),
        str(record.get("observation_id", ""))
    )


# ============================================================
# 4. RECHERCHE DANS LES ARCHIVES SPHEREx
# ============================================================

def search_observations(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = 1.0,
    collections: Sequence[str] = DEFAULT_COLLECTIONS,
    band: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Recherche les observations dans les collections IRSA."""

    ra_deg = float(ra_deg)
    dec_deg = float(dec_deg)
    radius_deg = float(radius_deg)

    if not 0 <= ra_deg <= 360:
        raise ValueError("RA must be between 0 and 360 degrees.")

    if not -90 <= dec_deg <= 90:
        raise ValueError("Dec must be between -90 and 90 degrees.")

    if not 0 < radius_deg <= 180:
        raise ValueError("Invalid search radius.")

    if int(limit) < 1:
        raise ValueError("Limit must be at least 1.")

    if not collections:
        raise ValueError("At least one collection is required.")

    position = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        frame="icrs"
    )

    records = []
    seen_urls = set()

    for collection in collections:

        table = Irsa.query_sia(
            pos=(position, radius_deg * u.deg),
            collection=collection
        )

        url_column = next(
            (
                name for name in table.colnames
                if name.lower() == "access_url"
            ),
            None
        )

        if url_column is None:
            continue

        for row in table:

            record = {
                str(name): _plain(row[name])
                for name in table.colnames
            }

            raw_url = record.get(url_column)

            if not isinstance(raw_url, str):
                continue

            url = raw_url.strip()

            if urlparse(url).scheme.lower() not in {"http", "https"}:
                continue

            if url in seen_urls:
                continue

            band_value = _get(
                record,
                "band",
                "filter",
                "filter_name"
            )

            if (
                band
                and band_value is not None
                and band.casefold() not in str(band_value).casefold()
            ):
                continue

            seen_urls.add(url)

            record["access_url"] = url
            record["collection"] = collection
            record["band_value"] = band_value

            record["epoch"] = _get(
                record,
                "mjd",
                "mjd_obs",
                "time",
                "date_obs",
                "obs_date",
                "t_min"
            )

            record["observation_id"] = str(
                _get(
                    record,
                    "obs_id",
                    "obsid",
                    "dp_id",
                    "obs_publisher_did"
                )
                or hashlib.sha256(
                    url.encode("utf-8")
                ).hexdigest()
            )

            records.append(record)

            if len(records) >= int(limit):
                break

        if len(records) >= int(limit):
            break

    records.sort(key=_sort_key)

    return records


# ============================================================
# 5. TÉLÉCHARGEMENT DES OBSERVATIONS
# ============================================================

def download_observation(
    observation: Mapping[str, Any] | str,
    destination: str | Path,
    timeout: tuple[float, float] = (15.0, 180.0),
) -> Path:
    """Télécharge une observation et vérifie son fichier FITS."""

    url = (
        observation
        if isinstance(observation, str)
        else observation.get("access_url")
    )

    if (
        not isinstance(url, str)
        or urlparse(url).scheme.lower() not in {"http", "https"}
    ):
        raise ValueError("Invalid observation URL.")

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = None

    try:

        with requests.get(
            url,
            stream=True,
            timeout=timeout,
            headers={"User-Agent": "SPHEREx-Explorer/1.0"}
        ) as response:

            response.raise_for_status()

            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                suffix=".download",
                delete=False
            ) as handle:

                temporary = Path(handle.name)

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        handle.write(chunk)

        with fits.open(temporary, memmap=False) as hdus:

            hdus.verify("exception")

            if not any(hdu.data is not None for hdu in hdus):
                raise ValueError("Downloaded FITS file has no data.")

        os.replace(temporary, path)
        temporary = None

    finally:

        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return path


def cached_download(
    observation: Mapping[str, Any],
    cache_dir: str | Path = DEFAULT_CACHE_DIR
) -> Path:
    """Évite de télécharger plusieurs fois la même observation."""

    url = observation.get("access_url")

    if not isinstance(url, str) or not url:
        raise ValueError("Observation is missing access_url.")

    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)

    path = cache / (
        hashlib.sha256(url.encode("utf-8")).hexdigest()
        + ".fits"
    )

    if path.is_file() and path.stat().st_size:

        try:

            with fits.open(path, memmap=False) as hdus:

                hdus.verify("exception")

                if any(hdu.data is not None for hdu in hdus):
                    return path

        except Exception:
            path.unlink(missing_ok=True)

    return download_observation(observation, path)


# ============================================================
# 6. LECTURE DES IMAGES FITS
# ============================================================

def read_fits_image(
    path: str | Path
) -> tuple[np.ndarray, WCS]:
    """Lit la première image 2D possédant des coordonnées célestes."""

    with fits.open(path, memmap=False) as hdus:

        for hdu in hdus:

            if hdu.data is None or np.ndim(hdu.data) != 2:
                continue

            if not np.issubdtype(hdu.data.dtype, np.number):
                continue

            wcs = WCS(hdu.header).celestial

            if not wcs.has_celestial:
                continue

            image = np.array(
                hdu.data,
                dtype=np.float64,
                copy=True
            )

            image[~np.isfinite(image)] = np.nan

            return image, wcs

    raise ValueError(
        f"No usable 2-D celestial image in {path}."
    )


# ============================================================
# 7. COMPARAISON SCIENTIFIQUE DE DEUX IMAGES
# ============================================================

def compare_images(
    reference_path: str | Path,
    comparison_path: str | Path,
    *,
    threshold_sigma: float = 5.0,
    coverage_threshold: float = 0.5,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Aligne deux images et calcule leur différence."""

    if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError("threshold_sigma must be positive.")

    if not 0 <= coverage_threshold <= 1:
        raise ValueError("Invalid coverage threshold.")

    reference, reference_wcs = read_fits_image(reference_path)

    comparison, comparison_wcs = read_fits_image(comparison_path)

    aligned, footprint = reproject_interp(
        (comparison, comparison_wcs),
        reference_wcs,
        shape_out=reference.shape,
        return_footprint=True
    )

    valid = (
        np.isfinite(reference)
        & np.isfinite(aligned)
        & np.isfinite(footprint)
        & (footprint > coverage_threshold)
    )

    if not np.any(valid):
        raise ValueError("No overlapping valid sky coverage.")

    difference = np.full(reference.shape, np.nan)

    difference[valid] = aligned[valid] - reference[valid]

    offset = float(np.median(difference[valid]))

    residual = np.full(reference.shape, np.nan)

    residual[valid] = difference[valid] - offset

    sigma = float(
        1.4826 * np.median(np.abs(residual[valid]))
    )

    significance = np.full(reference.shape, np.nan)

    if np.isfinite(sigma) and sigma > 0:

        significance[valid] = residual[valid] / sigma

        yy, xx = np.where(
            valid & (np.abs(significance) >= threshold_sigma)
        )

    else:

        yy, xx = np.array([], dtype=int), np.array([], dtype=int)

    candidates = [
        {
            "x": int(x),
            "y": int(y),
            "difference": float(difference[y, x]),
            "significance_sigma": float(significance[y, x])
        }
        for y, x in zip(yy, xx)
    ]

    artifacts = {}

    if output_dir is not None:

        folder = Path(output_dir).expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)

        diff_path = folder / "difference.fits"

        fits.PrimaryHDU(
            data=difference,
            header=reference_wcs.to_header()
        ).writeto(diff_path, overwrite=True)

        csv_path = folder / "pixels_candidats.csv"

        Table(
            rows=candidates,
            names=(
                "x",
                "y",
                "difference",
                "significance_sigma"
            )
        ).write(
            csv_path,
            format="ascii.csv",
            overwrite=True
        )

        artifacts = {
            "difference_fits": str(diff_path),
            "candidates_csv": str(csv_path)
        }

    return {
        "reference": reference,
        "aligned_comparison": aligned,
        "coverage": footprint,
        "valid_mask": valid,
        "difference": difference,
        "residual": residual,
        "significance": significance,
        "candidates": candidates,
        "median_offset": offset,
        "robust_sigma": sigma,
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "candidate_count": len(candidates),
        "threshold_sigma": float(threshold_sigma),
        "artifacts": artifacts,
        "warning": (
            "Exploratory pixel-level comparison only. "
            "Candidates are not confirmed astronomical sources."
        )
    }


# ============================================================
# 8. COMPARAISON DE DEUX OBSERVATIONS
# ============================================================

def compare_observations(
    earlier: Mapping[str, Any],
    later: Mapping[str, Any],
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_dir: str | Path | None = None,
    threshold_sigma: float = 5.0,
) -> dict[str, Any]:

    if earlier.get("observation_id") == later.get("observation_id"):
        raise ValueError("Select two different observations.")

    band_a = earlier.get("band_value")
    band_b = later.get("band_value")

    if (
        band_a
        and band_b
        and str(band_a).casefold() != str(band_b).casefold()
    ):
        raise ValueError("Different bands cannot be directly compared.")

    result = compare_images(
        cached_download(earlier, cache_dir),
        cached_download(later, cache_dir),
        threshold_sigma=threshold_sigma,
        output_dir=output_dir
    )

    result["earlier_observation"] = dict(earlier)
    result["later_observation"] = dict(later)

    return result


# ============================================================
# 9. COMPARAISON CHRONOLOGIQUE
# ============================================================

def compare_timeline(
    observations: Sequence[Mapping[str, Any]],
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_dir: str | Path | None = None,
    threshold_sigma: float = 5.0,
) -> list[dict[str, Any]]:

    ordered = sorted(observations, key=_sort_key)

    if len(ordered) < 2:
        raise ValueError("At least two observations are needed.")

    results = []

    for first, second in zip(ordered, ordered[1:]):

        pair_dir = None

        if output_dir is not None:

            pair_key = (
                f"{first.get('observation_id', '')}-"
                f"{second.get('observation_id', '')}"
            )

            pair_dir = (
                Path(output_dir)
                / hashlib.sha256(
                    pair_key.encode("utf-8")
                ).hexdigest()[:16]
            )

        results.append(
            compare_observations(
                first,
                second,
                cache_dir=cache_dir,
                output_dir=pair_dir,
                threshold_sigma=threshold_sigma
            )
        )

    return results


# ============================================================
# 10. INTERFACE GRAPHIQUE
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(140deg, #050D20, #0B1930);
    color: white;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

h1, h2, h3, p, label {
    color: white !important;
}

.hero {
    background: linear-gradient(110deg, #102C50, #10203B);
    border: 1px solid #28517B;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: white;
}

.hero-subtitle {
    color: #A9C8E8;
    font-size: 16px;
}

.stForm {
    background: #10223D;
    border: 1px solid #29466A;
    border-radius: 15px;
    padding: 22px;
}

[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #087FCB, #20B8ED);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    min-height: 48px;
}

.stButton button {
    background: #163454;
    color: white;
    border: 1px solid #3476A8;
    border-radius: 10px;
}

[data-testid="stImage"] {
    border: 1px solid #315577;
    border-radius: 14px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 11. EN-TÊTE
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🌌 SPHEREx Explorer</div>
    <div class="hero-subtitle">
        Explore the evolution of the universe through time.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 12. FORMULAIRE DE RECHERCHE
# ============================================================

st.markdown("### 🎯 Choose your destination")

with st.form("search_form"):

    col1, col2 = st.columns(2)

    with col1:

        ra = st.number_input(
            "Right Ascension (RA) in degrees",
            min_value=0.0,
            max_value=360.0,
            value=10.0
        )

    with col2:

        dec = st.number_input(
            "Declination (Dec) in degrees",
            min_value=-90.0,
            max_value=90.0,
            value=20.0
        )

    radius = st.number_input(
        "Search radius (degrees)",
        min_value=0.01,
        max_value=10.0,
        value=1.0
    )

    search = st.form_submit_button(
        "🚀 SEARCH SPHEREx OBSERVATIONS",
        use_container_width=True
    )


# ============================================================
# 13. EXÉCUTION DE LA RECHERCHE
# ============================================================

if search:

    st.session_state.search_done = False
    st.session_state.observations = []
    st.session_state.search_error = None

    try:

        with st.spinner("Searching the SPHEREx archive..."):

            observations = search_observations(
                ra_deg=ra,
                dec_deg=dec,
                radius_deg=radius,
                limit=100
            )

        st.session_state.observations = observations
        st.session_state.search_done = True

    except Exception as error:

        st.session_state.search_error = str(error)

        st.error(
            f"Archive search failed: {error}"
        )


# ============================================================
# 14. AFFICHAGE DES RÉSULTATS
# ============================================================

if st.session_state.search_done:

    observations = st.session_state.observations

    st.markdown("---")
    st.markdown("### 🪐 Exploration results")

    if not observations:

        st.warning(
            "No observations found for these coordinates."
        )

    else:

        st.success(
            f"{len(observations)} observation(s) found."
        )

        # Sélection de l'observation.
        if len(observations) > 1:

            index = st.slider(
                "Observation timeline",
                min_value=0,
                max_value=len(observations) - 1,
                value=0,
                format="Observation %d"
            )

        else:

            index = 0

            st.caption("Observation 1 / 1")

        current_observation = observations[index]

        # Téléchargement et lecture du fichier FITS.
        try:

            with st.spinner("Loading observation..."):

                fits_path = cached_download(current_observation)

                image_data, image_wcs = read_fits_image(fits_path)

            finite_pixels = image_data[np.isfinite(image_data)]

            if finite_pixels.size == 0:

                st.error(
                    "This observation contains no usable pixels."
                )

            else:

                # Amélioration du contraste pour l'affichage.
                low, high = np.percentile(
                    finite_pixels,
                    [1, 99]
                )

                if high > low:

                    display_image = np.clip(
                        (image_data - low) / (high - low),
                        0,
                        1
                    )

                else:

                    display_image = np.zeros_like(image_data)

                display_image = np.nan_to_num(display_image)

                st.markdown("#### 🔭 Current observation")

                st.image(
                    display_image,
                    caption="SPHEREx observation",
                    use_container_width=True
                )

                # Informations sur l'observation.
                st.markdown("#### Observation information")

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Observation",
                        f"{index + 1}/{len(observations)}"
                    )

                with col2:

                    st.write("**Date / Epoch**")

                    st.write(
                        str(
                            current_observation.get("epoch")
                            or "Not available"
                        )
                    )

                with col3:

                    st.write("**Band**")

                    st.write(
                        str(
                            current_observation.get("band_value")
                            or "Not available"
                        )
                    )

                st.caption(
                    "Observation ID: "
                    + str(
                        current_observation.get("observation_id")
                    )
                )

        except Exception as error:

            st.error(
                f"Could not load observation: {error}"
            )


# ============================================================
# 15. PIED DE PAGE
# ============================================================

st.markdown("---")

st.caption(
    "SPHEREx Explorer | NASA Space Apps Challenge"
)
