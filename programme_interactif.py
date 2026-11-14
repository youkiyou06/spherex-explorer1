from __future__ import annotations

"""SPHEREx Explorer — application Streamlit autonome.

Lancement : python -m streamlit run programme_interactif.py
Tout le code applicatif (recherche IRSA, cache FITS, affichage et comparaisons)
est regroupé dans ce fichier pour éviter les imports locaux supplémentaires.
"""

import csv
import hashlib
import io
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import astropy.units as u
import numpy as np
import requests
import streamlit as st
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa


# -----------------------------------------------------------------------------
# Configuration : les collections QR3/DR1 wide et deep disponibles dans IRSA
# -----------------------------------------------------------------------------
COLLECTIONS = {
    "QR3 — relevé large (wide)": "spherex_qr3",
    "QR3 — champ profond (deep)": "spherex_qr3_deep",
    "DR1 — relevé large (wide)": "spherex_dr1",
    "DR1 — champ profond (deep)": "spherex_dr1_deep",
}
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data" / "spherex_cache"

st.set_page_config(page_title="SPHEREx Explorer", page_icon="🌌", layout="wide")

for key, default in {
    "observations": [],
    "search_done": False,
    "search_summary": {},
    "comparison_result": None,
    "timeline_results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -----------------------------------------------------------------------------
# Métadonnées et recherche d'observations dans IRSA SIA2
# -----------------------------------------------------------------------------
def to_python(value: Any) -> Any:
    """Convertit les cellules NumPy/Astropy en valeurs affichables."""
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


def observation_sort_key(record: Mapping[str, Any]) -> tuple:
    epoch = get_field(record, "mjd", "mjd_obs", "t_min")
    try:
        if epoch is not None:
            return (0, float(epoch), str(record.get("observation_id", "")))
    except (TypeError, ValueError):
        pass
    epoch = get_field(record, "time", "date_obs", "obs_date")
    return (1, str(epoch or ""), str(record.get("observation_id", "")))


def search_observations(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    collections: Sequence[str],
    limit: int = 100,
    band_filter: str = "Toutes les bandes",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Interroge les collections choisies et renvoie aussi les erreurs partielles."""
    ra_deg, dec_deg, radius_deg = float(ra_deg), float(dec_deg), float(radius_deg)
    if not 0 <= ra_deg <= 360:
        raise ValueError("RA doit être comprise entre 0 et 360 degrés.")
    if not -90 <= dec_deg <= 90:
        raise ValueError("Dec doit être comprise entre -90 et 90 degrés.")
    if not 0 < radius_deg <= 180:
        raise ValueError("Le rayon doit être supérieur à 0 et inférieur ou égal à 180 degrés.")
    if not collections:
        raise ValueError("Sélectionnez au moins une collection SPHEREx.")
    if any(collection not in COLLECTIONS.values() for collection in collections):
        raise ValueError("Collection non reconnue.")
    if int(limit) < 1:
        raise ValueError("La limite doit être au moins égale à 1.")

    position = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    observations: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    errors: list[str] = []

    for collection in collections:
        try:
            table = Irsa.query_sia(pos=(position, radius_deg * u.deg), collection=collection)
        except Exception as error:
            errors.append(f"{collection} : {error}")
            continue

        columns = {str(name).lower(): name for name in table.colnames}
        url_column = columns.get("access_url")
        if url_column is None:
            errors.append(f"{collection} : la réponse IRSA ne contient pas access_url.")
            continue

        for row in table:
            record = {str(name): to_python(row[name]) for name in table.colnames}
            raw_url = record.get(str(url_column))
            if not isinstance(raw_url, str):
                continue
            url = raw_url.strip()
            if urlparse(url).scheme.lower() not in {"http", "https"} or url in seen_urls:
                continue

            band = get_field(record, "band", "filter", "filter_name", "instrume")
            if band_filter != "Toutes les bandes" and band is not None:
                if band_filter.casefold() not in str(band).casefold():
                    continue

            seen_urls.add(url)
            record["access_url"] = url
            record["collection"] = collection
            record["band_value"] = band
            record["epoch"] = get_field(
                record, "mjd", "mjd_obs", "t_min", "time", "date_obs", "obs_date"
            ) or "Époque non fournie"
            record["observation_id"] = str(
                get_field(record, "obs_id", "obsid", "dp_id", "obs_publisher_did")
                or hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
            )
            observations.append(record)
            if len(observations) >= int(limit):
                break
        if len(observations) >= int(limit):
            break

    observations.sort(key=observation_sort_key)
    if not observations and errors:
        raise RuntimeError("La recherche IRSA a échoué. " + " | ".join(errors))
    return observations, errors


# -----------------------------------------------------------------------------
# Téléchargement FITS sécurisé, cache local et lecture d'image céleste
# -----------------------------------------------------------------------------
def download_observation(observation: Mapping[str, Any], destination: Path) -> Path:
    url = observation.get("access_url")
    if not isinstance(url, str) or urlparse(url).scheme.lower() not in {"http", "https"}:
        raise ValueError("L'observation ne contient pas d'URL HTTP/HTTPS valide.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(3):
        temporary: Path | None = None
        try:
            with requests.get(
                url, stream=True, timeout=(20, 300),
                headers={"User-Agent": "SPHEREx-Explorer/1.0"},
            ) as response:
                response.raise_for_status()
                expected = response.headers.get("Content-Length")
                expected_size = int(expected) if expected and expected.isdigit() else None
                received = 0
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, suffix=".download", delete=False
                ) as handle:
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
                    raise ValueError("Le FITS téléchargé ne contient aucune donnée.")
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
        raise ValueError("URL d'accès manquante dans les métadonnées.")
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


def read_fits_image(path: str | Path) -> tuple[np.ndarray, WCS]:
    """Lit la première extension numérique 2D ayant un WCS céleste."""
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
    raise ValueError("Aucune extension FITS 2D avec coordonnées célestes (WCS) n'a été trouvée.")


def display_scale(image: np.ndarray, low_pct: float = 1, high_pct: float = 99) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(finite, [low_pct, high_pct])
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.nan_to_num(np.clip((image - low) / (high - low), 0, 1)).astype(np.float32)


# -----------------------------------------------------------------------------
# Reprojection bilinéaire en NumPy/Astropy (pas de module de projet séparé requis)
# -----------------------------------------------------------------------------
def reproject_bilinear(
    source: np.ndarray,
    source_wcs: WCS,
    target_wcs: WCS,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rééchantillonne une image sur la grille cible à partir de ses WCS."""
    height, width = target_shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    world_x, world_y = target_wcs.pixel_to_world_values(xx, yy)
    source_x, source_y = source_wcs.world_to_pixel_values(world_x, world_y)
    x0 = np.floor(source_x).astype(np.int64, copy=False)
    y0 = np.floor(source_y).astype(np.int64, copy=False)
    dx, dy = source_x - x0, source_y - y0
    src_h, src_w = source.shape
    inside = (
        np.isfinite(source_x) & np.isfinite(source_y)
        & (source_x >= 0) & (source_y >= 0)
        & (source_x < src_w - 1) & (source_y < src_h - 1)
    )
    aligned = np.full(target_shape, np.nan, dtype=np.float64)
    footprint = np.zeros(target_shape, dtype=np.float32)
    if not np.any(inside):
        return aligned, footprint

    rows, cols = np.where(inside)
    sx, sy = x0[inside], y0[inside]
    weights = ((1 - dx[inside]) * (1 - dy[inside]),
               dx[inside] * (1 - dy[inside]),
               (1 - dx[inside]) * dy[inside],
               dx[inside] * dy[inside])
    pixels = (source[sy, sx], source[sy, sx + 1],
              source[sy + 1, sx], source[sy + 1, sx + 1])
    numerator = np.zeros(len(rows), dtype=np.float64)
    denominator = np.zeros(len(rows), dtype=np.float64)
    for weight, pixel in zip(weights, pixels):
        good = np.isfinite(pixel)
        numerator[good] += weight[good] * pixel[good]
        denominator[good] += weight[good]
    good = denominator > 0
    aligned[rows[good], cols[good]] = numerator[good] / denominator[good]
    footprint[rows[good], cols[good]] = denominator[good]
    return aligned, footprint


def compare_images(
    reference: np.ndarray,
    reference_wcs: WCS,
    comparison: np.ndarray,
    comparison_wcs: WCS,
    threshold_sigma: float = 5.0,
    coverage_threshold: float = 0.5,
) -> dict[str, Any]:
    if not np.isfinite(threshold_sigma) or threshold_sigma <= 0:
        raise ValueError("Le seuil sigma doit être strictement positif.")
    aligned, footprint = reproject_bilinear(
        comparison, comparison_wcs, reference_wcs, reference.shape
    )
    valid = np.isfinite(reference) & np.isfinite(aligned) & (footprint > coverage_threshold)
    if not np.any(valid):
        raise ValueError("Les images n'ont pas de couverture céleste valide en commun.")

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
        "difference": difference, "residual": residual,
        "significance": significance, "valid_mask": valid,
        "candidates": candidates, "median_offset": offset,
        "robust_sigma": sigma, "valid_pixel_count": int(np.count_nonzero(valid)),
        "threshold_sigma": float(threshold_sigma),
        "coverage": footprint,
    }


def compare_observations(
    earlier: Mapping[str, Any], later: Mapping[str, Any], threshold_sigma: float
) -> dict[str, Any]:
    if earlier.get("observation_id") == later.get("observation_id"):
        raise ValueError("Choisissez deux observations différentes.")
    band_a, band_b = earlier.get("band_value"), later.get("band_value")
    if band_a and band_b and str(band_a).casefold() != str(band_b).casefold():
        raise ValueError("Les observations sont dans des bandes différentes ; comparaison directe refusée.")
    ref_image, ref_wcs = read_fits_image(cached_download(earlier))
    cmp_image, cmp_wcs = read_fits_image(cached_download(later))
    result = compare_images(ref_image, ref_wcs, cmp_image, cmp_wcs, threshold_sigma)
    result["reference_wcs"] = ref_wcs
    result["earlier_observation"] = dict(earlier)
    result["later_observation"] = dict(later)
    return result


def candidates_csv(candidates: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=("x", "y", "difference", "significance_sigma")
    )
    writer.writeheader()
    writer.writerows(candidates)
    return output.getvalue().encode("utf-8")


def difference_fits_bytes(result: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    hdu = fits.PrimaryHDU(
        data=result["difference"], header=result["reference_wcs"].to_header()
    )
    hdu.writeto(buffer)
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Interface et contrôles interactifs
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
  <div class="hero-subtitle">Rechercher, explorer et comparer les observations SPHEREx dans le navigateur.</div>
</div>
""", unsafe_allow_html=True)
st.info("Les données proviennent des archives IRSA. Une connexion Internet est nécessaire. Les détections affichées sont exploratoires et ne sont pas des sources confirmées.")

st.subheader("🎯 Recherche dans les archives")
with st.form("spherex_search_form"):
    c1, c2 = st.columns(2)
    with c1:
        ra = st.number_input("Ascension droite (RA, degrés)", min_value=0.0, max_value=360.0, value=10.0)
    with c2:
        dec = st.number_input("Déclinaison (Dec, degrés)", min_value=-90.0, max_value=90.0, value=20.0)
    c3, c4 = st.columns(2)
    with c3:
        radius = st.number_input("Rayon de recherche (degrés)", min_value=0.01, max_value=10.0, value=1.0)
    with c4:
        limit = st.number_input("Maximum d'observations", min_value=1, max_value=500, value=100, step=1)
    collection_labels = st.multiselect(
        "Collections SPHEREx à consulter", options=list(COLLECTIONS),
        default=["QR3 — relevé large (wide)", "DR1 — relevé large (wide)"],
    )
    band_filter = st.selectbox(
        "Filtre de bande", ["Toutes les bandes", "1", "2", "3", "4", "5", "6"],
        help="Le filtre est appliqué aux métadonnées de bande lorsqu'elles sont disponibles.",
    )
    submitted = st.form_submit_button("🚀 Rechercher dans IRSA", use_container_width=True)

if submitted:
    st.session_state.search_done = False
    st.session_state.observations = []
    st.session_state.comparison_result = None
    st.session_state.timeline_results = None
    try:
        selected_collections = [COLLECTIONS[label] for label in collection_labels]
        with st.spinner("Interrogation des archives IRSA / SPHEREx…"):
            found, query_errors = search_observations(
                ra, dec, radius, selected_collections, int(limit), band_filter
            )
        st.session_state.observations = found
        st.session_state.search_done = True
        st.session_state.search_summary = {"ra": ra, "dec": dec, "radius": radius}
        if query_errors:
            st.warning("Certaines collections n'ont pas répondu : " + " | ".join(query_errors))
    except Exception as error:
        st.error(f"La recherche a échoué : {error}")

if st.session_state.search_done:
    observations = st.session_state.observations
    st.markdown("---")
    st.subheader("🪐 Observations trouvées")
    if not observations:
        st.warning("Aucune observation trouvée pour cette position et ces filtres.")
    else:
        st.success(f"{len(observations)} observation(s) trouvée(s), triée(s) chronologiquement.")
        rows = [{
            "N°": i + 1,
            "Époque": str(obs.get("epoch") or "Indisponible"),
            "Bande": str(obs.get("band_value") or "Indisponible"),
            "Collection": next((label for label, value in COLLECTIONS.items() if value == obs.get("collection")), obs.get("collection", "")),
            "ID": str(obs.get("observation_id", "")),
        } for i, obs in enumerate(observations)]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        labels = [
            f"{i + 1} — {obs.get('epoch') or 'époque inconnue'} | "
            f"bande {obs.get('band_value') or '?'} | {obs.get('observation_id', '')[:20]}"
            for i, obs in enumerate(observations)
        ]
        st.markdown("### 🔭 Explorer une observation")
        selected_index = st.select_slider(
            "Déplacer le curseur pour parcourir les époques", options=list(range(len(observations))),
            format_func=lambda i: labels[i], key="observation_timeline",
        )
        current = observations[selected_index]
        try:
            with st.spinner("Téléchargement et lecture du fichier FITS…"):
                image, image_wcs = read_fits_image(cached_download(current))
            finite = image[np.isfinite(image)]
            if finite.size == 0:
                st.warning("Cette image ne contient aucun pixel exploitable.")
            else:
                p1, p2 = st.columns(2)
                with p1:
                    low_pct = st.slider("Borne basse du contraste (%)", 0.0, 20.0, 1.0, 0.5, key="low_pct")
                with p2:
                    high_pct = st.slider("Borne haute du contraste (%)", 80.0, 100.0, 99.0, 0.5, key="high_pct")
                if low_pct >= high_pct:
                    st.warning("La borne basse doit être inférieure à la borne haute.")
                else:
                    st.image(
                        display_scale(image, low_pct, high_pct),
                        caption=f"Observation {selected_index + 1} — contraste ajustable",
                        use_container_width=True,
                    )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Observation", f"{selected_index + 1}/{len(observations)}")
                m2.metric("Époque", str(current.get("epoch") or "Indisponible"))
                m3.metric("Bande", str(current.get("band_value") or "Indisponible"))
                m4.metric("Taille image", f"{image.shape[1]} × {image.shape[0]} pixels")
                st.caption(f"Collection : {current.get('collection')} · Identifiant : {current.get('observation_id')}")
        except Exception as error:
            st.error(f"Impossible de charger cette observation : {error}")

        st.markdown("---")
        st.subheader("🔬 Comparer deux observations")
        if len(observations) < 2:
            st.info("Il faut au moins deux observations pour comparer les images.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                first_index = st.selectbox("Image de référence", range(len(observations)), index=0, format_func=lambda i: labels[i], key="comparison_reference")
            with c2:
                second_index = st.selectbox("Image à comparer", range(len(observations)), index=min(1, len(observations) - 1), format_func=lambda i: labels[i], key="comparison_later")
            threshold = st.slider("Seuil indicatif des pixels candidats (σ)", 2.0, 10.0, 5.0, 0.5, key="comparison_threshold")
            if st.button("Comparer les images sélectionnées", type="primary", use_container_width=True):
                try:
                    with st.spinner("Téléchargement, alignement céleste et comparaison…"):
                        st.session_state.comparison_result = compare_observations(
                            observations[first_index], observations[second_index], threshold
                        )
                except Exception as error:
                    st.session_state.comparison_result = None
                    st.error(f"La comparaison a échoué : {error}")

            result = st.session_state.comparison_result
            if result is not None:
                st.markdown("#### Résultats")
                m1, m2, m3 = st.columns(3)
                m1.metric("Pixels valides", f"{result['valid_pixel_count']:,}")
                m2.metric("Décalage médian", f"{result['median_offset']:.5g}")
                m3.metric("Dispersion robuste", f"{result['robust_sigma']:.5g}")
                p1, p2, p3 = st.columns(3)
                p1.image(display_scale(result["reference"]), caption="Image de référence", use_container_width=True)
                p2.image(display_scale(result["aligned_comparison"]), caption="Image alignée sur la référence", use_container_width=True)
                p3.image(display_scale(result["difference"]), caption="Différence (image alignée − référence)", use_container_width=True)
                candidates = result["candidates"]
                st.write(f"Pixels candidats au seuil choisi : **{len(candidates)}**")
                if candidates:
                    st.dataframe(candidates, use_container_width=True, hide_index=True)
                else:
                    st.info("Aucun pixel n'a dépassé le seuil choisi.")
                csv_data = candidates_csv(candidates)
                st.download_button("Télécharger les candidats (CSV)", csv_data, "pixels_candidats.csv", "text/csv")
                st.download_button(
                    "Télécharger l'image différence (FITS)",
                    difference_fits_bytes(result), "difference.fits", "application/fits",
                )
                st.caption("Analyse exploratoire au niveau des pixels : les candidats ne sont pas des sources astronomiques confirmées.")

        st.markdown("---")
        st.subheader("⏳ Comparaison chronologique")
        st.write("Compare chaque observation avec la suivante dans l'ordre des époques. Cette opération peut télécharger plusieurs fichiers FITS.")
        timeline_threshold = st.slider("Seuil pour la chronologie (σ)", 2.0, 10.0, 5.0, 0.5, key="timeline_threshold")
        if len(observations) < 2:
            st.info("Il faut au moins deux observations pour comparer une chronologie.")
        elif st.button("Comparer toute la chronologie", use_container_width=True):
            timeline_results = []
            try:
                progress = st.progress(0, text="Préparation des comparaisons…")
                for pair_index, (first, second) in enumerate(zip(observations, observations[1:])):
                    progress.progress((pair_index + 1) / (len(observations) - 1), text=f"Comparaison {pair_index + 1}/{len(observations) - 1}…")
                    try:
                        pair_result = compare_observations(first, second, timeline_threshold)
                        timeline_results.append({
                            "first": first, "second": second,
                            "candidate_count": len(pair_result["candidates"]),
                            "valid_pixel_count": pair_result["valid_pixel_count"],
                            "median_offset": pair_result["median_offset"],
                            "robust_sigma": pair_result["robust_sigma"],
                            "csv": candidates_csv(pair_result["candidates"]),
                            "fits": difference_fits_bytes(pair_result),
                        })
                    except Exception as pair_error:
                        timeline_results.append({"first": first, "second": second, "error": str(pair_error)})
                st.session_state.timeline_results = timeline_results
            except Exception as error:
                st.error(f"La comparaison chronologique a échoué : {error}")

        timeline_results = st.session_state.timeline_results
        if timeline_results:
            summary = []
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
                for i, pair in enumerate(timeline_results, start=1):
                    label = f"{i:03d}_{str(pair['first'].get('observation_id', 'obs'))[:16]}_vers_{str(pair['second'].get('observation_id', 'obs'))[:16]}"
                    if "error" in pair:
                        summary.append({"Paire": f"{i}", "Statut": pair["error"], "Pixels candidats": None, "Pixels valides": None})
                        continue
                    summary.append({
                        "Paire": f"{pair['first'].get('epoch')} → {pair['second'].get('epoch')}",
                        "Statut": "Comparée",
                        "Pixels candidats": pair["candidate_count"],
                        "Pixels valides": pair["valid_pixel_count"],
                        "Décalage médian": pair["median_offset"],
                        "Dispersion robuste": pair["robust_sigma"],
                    })
                    zipped.writestr(f"{label}/pixels_candidats.csv", pair["csv"])
                    zipped.writestr(f"{label}/difference.fits", pair["fits"])
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.download_button(
                "Télécharger tous les résultats chronologiques (ZIP)",
                archive.getvalue(), "resultats_chronologie.zip", "application/zip",
            )

st.markdown("---")
st.caption("SPHEREx Explorer · Les données scientifiques sont interrogées à la demande dans les archives publiques IRSA.")
