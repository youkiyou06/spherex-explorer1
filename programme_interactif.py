from __future__ import annotations

"""Version interactive autonome de SPHEREx Explorer.

Lancez-la avec : streamlit run programme_interactif.py
Le fichier `programme` original n'est pas modifié.
"""

import hashlib
import os
import tempfile
import time
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


# -----------------------------------------------------------------------------
# Configuration et état de session
# -----------------------------------------------------------------------------
DEFAULT_COLLECTIONS = ("spherex_dr1", "spherex_qr3")
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data"

st.set_page_config(page_title="SPHEREx Explorer", page_icon="🌌", layout="wide")

for key, default in {
    "observations": [],
    "search_done": False,
    "comparison_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -----------------------------------------------------------------------------
# Outils de métadonnées et d'archives
# -----------------------------------------------------------------------------
def plain(value: Any) -> Any:
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


def get_field(record: Mapping[str, Any], *names: str) -> Any:
    fields = {str(key).lower(): value for key, value in record.items()}
    for name in names:
        value = fields.get(name.lower())
        if value is not None and str(value).strip().lower() not in {"", "--", "nan", "none"}:
            return value
    return None


def sort_key(record: Mapping[str, Any]) -> tuple:
    mjd = get_field(record, "mjd", "mjd_obs", "t_min")
    try:
        if mjd is not None:
            return (0, float(mjd), str(record.get("observation_id", "")))
    except (TypeError, ValueError):
        pass
    date = get_field(record, "time", "date_obs", "obs_date", "t_min")
    return (1, str(date or ""), str(record.get("observation_id", "")))


def search_observations(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = 1.0,
    collections: Sequence[str] = DEFAULT_COLLECTIONS,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if not 0 <= float(ra_deg) <= 360:
        raise ValueError("L'ascension droite doit être comprise entre 0 et 360°.")
    if not -90 <= float(dec_deg) <= 90:
        raise ValueError("La déclinaison doit être comprise entre -90 et 90°.")
    if not 0 < float(radius_deg) <= 180:
        raise ValueError("Le rayon doit être compris entre 0 et 180°.")
    if int(limit) < 1:
        raise ValueError("La limite doit être supérieure à zéro.")

    position = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    records: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for collection in collections:
        table = Irsa.query_sia(pos=(position, float(radius_deg) * u.deg), collection=collection)
        url_column = next((name for name in table.colnames if name.lower() == "access_url"), None)
        if url_column is None:
            continue
        for row in table:
            record = {str(name): plain(row[name]) for name in table.colnames}
            raw_url = record.get(url_column)
            if not isinstance(raw_url, str):
                continue
            url = raw_url.strip()
            if urlparse(url).scheme.lower() not in {"http", "https"} or url in seen_urls:
                continue
            seen_urls.add(url)
            band = get_field(record, "band", "filter", "filter_name")
            record.update(
                access_url=url,
                collection=collection,
                band_value=band,
                epoch=get_field(record, "mjd", "mjd_obs", "time", "date_obs", "obs_date", "t_min"),
                observation_id=str(
                    get_field(record, "obs_id", "obsid", "dp_id", "obs_publisher_did")
                    or hashlib.sha256(url.encode("utf-8")).hexdigest()
                ),
            )
            records.append(record)
            if len(records) >= int(limit):
                break
        if len(records) >= int(limit):
            break
    return sorted(records, key=sort_key)


# -----------------------------------------------------------------------------
# Téléchargement cache et lecture FITS
# -----------------------------------------------------------------------------
def download_observation(observation: Mapping[str, Any], destination: Path) -> Path:
    url = observation.get("access_url")
    if not isinstance(url, str) or urlparse(url).scheme.lower() not in {"http", "https"}:
        raise ValueError("URL d'observation invalide.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(3):
        temporary: Path | None = None
        try:
            with requests.get(
                url, stream=True, timeout=(20.0, 300.0),
                headers={"User-Agent": "SPHEREx-Explorer/1.0"},
            ) as response:
                response.raise_for_status()
                expected = response.headers.get("Content-Length")
                expected_size = int(expected) if expected and expected.isdigit() else None
                received = 0
                with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as handle:
                    temporary = Path(handle.name)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            received += len(chunk)
            if expected_size is not None and received != expected_size:
                raise IOError(f"Téléchargement incomplet : {received}/{expected_size} octets.")
            with fits.open(temporary, memmap=False) as hdus:
                hdus.verify("exception")
                if not any(hdu.data is not None for hdu in hdus):
                    raise ValueError("Le FITS téléchargé ne contient pas de données.")
            os.replace(temporary, destination)
            temporary = None
            return destination
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Échec du téléchargement après trois tentatives : {last_error}")


def cached_download(observation: Mapping[str, Any]) -> Path:
    url = observation.get("access_url")
    if not isinstance(url, str) or not url:
        raise ValueError("L'observation ne contient pas d'URL de téléchargement.")
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
    return download_observation(observation, path)


def read_fits_image(path: Path) -> tuple[np.ndarray, WCS]:
    with fits.open(path, memmap=False) as hdus:
        for hdu in hdus:
            if hdu.data is None or np.ndim(hdu.data) != 2 or not np.issubdtype(hdu.data.dtype, np.number):
                continue
            wcs = WCS(hdu.header).celestial
            if wcs.has_celestial:
                image = np.array(hdu.data, dtype=np.float64, copy=True)
                image[~np.isfinite(image)] = np.nan
                return image, wcs
    raise ValueError(f"Aucune image céleste 2D exploitable dans {path}.")


def display_scale(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if not finite.size:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.nan_to_num(np.clip((image - low) / (high - low), 0, 1)).astype(np.float32)


def compare_observations(
    earlier: Mapping[str, Any], later: Mapping[str, Any], threshold_sigma: float = 5.0
) -> dict[str, Any]:
    if earlier.get("observation_id") == later.get("observation_id"):
        raise ValueError("Choisissez deux observations différentes.")
    band_a, band_b = earlier.get("band_value"), later.get("band_value")
    if band_a and band_b and str(band_a).casefold() != str(band_b).casefold():
        raise ValueError("Les bandes sont différentes : leur comparaison directe n'est pas appropriée.")
    reference, reference_wcs = read_fits_image(cached_download(earlier))
    comparison, comparison_wcs = read_fits_image(cached_download(later))
    aligned, footprint = reproject_interp(
        (comparison, comparison_wcs), reference_wcs,
        shape_out=reference.shape, return_footprint=True,
    )
    valid = np.isfinite(reference) & np.isfinite(aligned) & (footprint > 0.5)
    if not np.any(valid):
        raise ValueError("Les observations n'ont pas de couverture céleste valide en commun.")
    difference = np.full(reference.shape, np.nan, dtype=np.float64)
    difference[valid] = aligned[valid] - reference[valid]
    offset = float(np.median(difference[valid]))
    residual = np.full(reference.shape, np.nan, dtype=np.float64)
    residual[valid] = difference[valid] - offset
    sigma = float(1.4826 * np.median(np.abs(residual[valid])))
    significance = np.full(reference.shape, np.nan, dtype=np.float64)
    if np.isfinite(sigma) and sigma > 0:
        significance[valid] = residual[valid] / sigma
        yy, xx = np.where(valid & (np.abs(significance) >= threshold_sigma))
    else:
        yy, xx = np.array([], dtype=int), np.array([], dtype=int)
    candidates = [
        {"x": int(x), "y": int(y), "difference": float(difference[y, x]),
         "significance_sigma": float(significance[y, x])}
        for y, x in zip(yy, xx)
    ]
    return {
        "reference": reference, "aligned_comparison": aligned,
        "difference": difference, "significance": significance,
        "candidates": candidates, "median_offset": offset,
        "robust_sigma": sigma, "valid_pixel_count": int(np.count_nonzero(valid)),
        "threshold_sigma": float(threshold_sigma),
    }


# -----------------------------------------------------------------------------
# Interface Streamlit
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(140deg, #050D20, #0B1930); color: white;}
.block-container {max-width: 1250px; padding-top: 2rem;}
h1, h2, h3, p, label {color: white !important;}
.hero {background: linear-gradient(110deg, #102C50, #10203B); border: 1px solid #28517B; border-radius: 18px; padding: 28px; margin-bottom: 20px;}
.hero-title {font-size: 38px; font-weight: 800; color: white;}
.hero-subtitle {color: #A9C8E8; font-size: 16px;}
[data-testid="stForm"] {background: #10223D; border: 1px solid #29466A; border-radius: 15px; padding: 20px;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <div class="hero-title">🌌 SPHEREx Explorer</div>
  <div class="hero-subtitle">Rechercher, parcourir et comparer interactivement les observations du ciel.</div>
</div>
""", unsafe_allow_html=True)

st.subheader("🎯 Recherche dans les archives")
with st.form("search_form"):
    col1, col2 = st.columns(2)
    with col1:
        ra = st.number_input("Ascension droite (RA, degrés)", min_value=0.0, max_value=360.0, value=10.0)
    with col2:
        dec = st.number_input("Déclinaison (Dec, degrés)", min_value=-90.0, max_value=90.0, value=20.0)
    col3, col4 = st.columns(2)
    with col3:
        radius = st.number_input("Rayon de recherche (degrés)", min_value=0.01, max_value=10.0, value=1.0)
    with col4:
        limit = st.number_input("Nombre maximum de résultats", min_value=1, max_value=500, value=100, step=1)
    submitted = st.form_submit_button("🚀 Rechercher les observations", use_container_width=True)

if submitted:
    st.session_state.search_done = False
    st.session_state.observations = []
    st.session_state.comparison_result = None
    try:
        with st.spinner("Recherche dans les archives SPHEREx / IRSA…"):
            st.session_state.observations = search_observations(ra, dec, radius, limit=int(limit))
        st.session_state.search_done = True
    except Exception as error:
        st.error(f"La recherche a échoué : {error}")

if st.session_state.search_done:
    observations = st.session_state.observations
    st.markdown("---")
    st.subheader("🪐 Résultats")
    if not observations:
        st.warning("Aucune observation trouvée pour ces coordonnées.")
    else:
        st.success(f"{len(observations)} observation(s) trouvée(s).")
        table_rows = []
        for i, observation in enumerate(observations):
            table_rows.append({
                "N°": i + 1,
                "Époque": str(observation.get("epoch") or "Indisponible"),
                "Bande": str(observation.get("band_value") or "Indisponible"),
                "Collection": str(observation.get("collection") or ""),
                "ID": str(observation.get("observation_id", "")),
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        labels = [
            f"{i + 1} — {obs.get('epoch') or 'époque inconnue'} | "
            f"{obs.get('band_value') or 'bande inconnue'} | {obs.get('observation_id', '')[:24]}"
            for i, obs in enumerate(observations)
        ]
        st.markdown("### 🔭 Parcourir la chronologie")
        selected_index = st.select_slider(
            "Sélectionner une observation", options=list(range(len(observations))),
            format_func=lambda i: labels[i], key="observation_timeline",
        )
        current = observations[selected_index]
        try:
            with st.spinner("Téléchargement et lecture du FITS…"):
                image, _ = read_fits_image(cached_download(current))
            st.image(display_scale(image), caption=f"Observation {selected_index + 1} — image FITS (contraste ajusté)", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Observation", f"{selected_index + 1}/{len(observations)}")
            c2.metric("Époque", str(current.get("epoch") or "Indisponible"))
            c3.metric("Bande", str(current.get("band_value") or "Indisponible"))
            st.caption(f"Collection : {current.get('collection', 'inconnue')} · ID : {current.get('observation_id', 'inconnu')}")
        except Exception as error:
            st.error(f"Impossible de charger cette observation : {error}")

        st.markdown("---")
        st.subheader("🔬 Comparer deux observations")
        if len(observations) < 2:
            st.info("Il faut au moins deux observations pour effectuer une comparaison.")
        else:
            left, right = st.columns(2)
            with left:
                first_index = st.selectbox("Observation de référence (la plus ancienne)", range(len(observations)), index=0, format_func=lambda i: labels[i], key="compare_first")
            with right:
                second_default = min(1, len(observations) - 1)
                second_index = st.selectbox("Observation à comparer", range(len(observations)), index=second_default, format_func=lambda i: labels[i], key="compare_second")
            threshold = st.slider("Seuil indicatif des pixels candidats (σ)", min_value=2.0, max_value=10.0, value=5.0, step=0.5)
            if st.button("Comparer les images", type="primary", use_container_width=True):
                try:
                    with st.spinner("Alignement des images et calcul de la différence…"):
                        result = compare_observations(observations[first_index], observations[second_index], threshold)
                    st.session_state.comparison_result = result
                except Exception as error:
                    st.session_state.comparison_result = None
                    st.error(f"La comparaison a échoué : {error}")

            result = st.session_state.comparison_result
            if result is not None:
                st.markdown("#### Résultat de la comparaison")
                m1, m2, m3 = st.columns(3)
                m1.metric("Pixels valides", f"{result['valid_pixel_count']:,}")
                m2.metric("Décalage médian", f"{result['median_offset']:.5g}")
                m3.metric("Dispersion robuste (σ)", f"{result['robust_sigma']:.5g}")
                p1, p2, p3 = st.columns(3)
                p1.image(display_scale(result["reference"]), caption="Image de référence", use_container_width=True)
                p2.image(display_scale(result["aligned_comparison"]), caption="Image alignée", use_container_width=True)
                p3.image(display_scale(result["difference"]), caption="Différence (contraste ajusté)", use_container_width=True)
                st.caption("Comparaison exploratoire au niveau des pixels : les candidats ne sont pas des sources astronomiques confirmées.")
                candidates = result["candidates"]
                st.write(f"Pixels candidats au seuil choisi : **{len(candidates)}**")
                if candidates:
                    st.dataframe(candidates, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucun pixel candidat n'a dépassé le seuil choisi.")
                csv_bytes = Table(rows=candidates, names=("x", "y", "difference", "significance_sigma")).to_pandas().to_csv(index=False).encode("utf-8")
                st.download_button("Télécharger les candidats (CSV)", data=csv_bytes, file_name="pixels_candidats.csv", mime="text/csv")

st.markdown("---")
st.caption("SPHEREx Explorer · Application exploratoire pour les archives SPHEREx/IRSA")
