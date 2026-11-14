"""Interface Streamlit de collecte de métadonnées et fichiers FITS SPHEREx.

Les données FITS téléchargées sont conservées telles quelles (octets d'origine) ;
seul l'aperçu à l'écran applique un étirement de contraste pour la lisibilité.
Le catalogue local est enregistré dans data/spherex_db/index.json.

Dépendances : streamlit, astroquery, astropy, numpy, matplotlib, requests
Lancement : streamlit run collecte_db_fits.py
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.data import conf as astropy_conf
from astroquery.ipac.irsa import Irsa


COLLECTIONS = {
    "QR3 — Wide": "spherex_qr3",
    "QR3 — Deep": "spherex_qr3_deep",
    "DR1 — Wide": "spherex_dr1",
    "DR1 — Deep": "spherex_dr1_deep",
}
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "data" / "spherex_db"
FITS_DIR = DB_DIR / "fits"
INDEX_FILE = DB_DIR / "index.json"
astropy_conf.remote_timeout = 180

st.set_page_config(page_title="Collecte SPHEREx FITS", page_icon="🛰️", layout="wide")
st.title("🛰️ Collecte de données SPHEREx — FITS sans perte")
st.write(
    "Recherchez des produits dans l’archive IRSA, enregistrez leurs métadonnées "
    "dans un index local et téléchargez les FITS originaux."
)
st.info(
    "Le fichier FITS sauvegardé n’est ni converti ni réécrit. L’aperçu à l’écran "
    "est uniquement une visualisation avec contraste ajusté : les calculs de la "
    "fonction Jeanne doivent utiliser les pixels lus directement depuis le FITS, "
    "pas une capture ou un PNG."
)


def json_value(value: Any) -> Any:
    """Convertit les cellules Astropy/NumPy en types compatibles JSON."""
    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def read_index() -> list[dict[str, Any]]:
    if not INDEX_FILE.exists():
        return []
    try:
        with INDEX_FILE.open("r", encoding="utf-8") as source:
            data = json.load(source)
        records = data.get("observations", [])
        return records if isinstance(records, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def write_index(records: list[dict[str, Any]]) -> None:
    """Écrit l'index atomiquement afin d'éviter un JSON partiel."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    handle_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=DB_DIR, suffix=".tmp", delete=False
        ) as handle:
            handle_name = handle.name
            json.dump({"observations": records}, handle, ensure_ascii=False, indent=2)
        os.replace(handle_name, INDEX_FILE)
    finally:
        if handle_name and os.path.exists(handle_name):
            os.unlink(handle_name)


def merge_into_index(records: list[dict[str, Any]]) -> int:
    """Ajoute les résultats sans doublons d'URL. Retourne le nombre ajouté."""
    stored = read_index()
    known_urls = {item.get("access_url") for item in stored}
    added = 0
    for record in records:
        url = record.get("access_url")
        if url and url not in known_urls:
            stored.append(record)
            known_urls.add(url)
            added += 1
    write_index(stored)
    return added


def search_irsa(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    collections: list[str],
    band_um: tuple[float, float] | None,
    time_mjd: tuple[float, float] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Recherche dans les collections choisies avec Astroquery/IRSA SIA."""
    if not 0 <= ra_deg < 360:
        raise ValueError("RA doit être comprise entre 0 inclus et 360° exclu.")
    if not -90 <= dec_deg <= 90:
        raise ValueError("Dec doit être comprise entre -90° et +90°.")
    if not 0 < radius_deg <= 10:
        raise ValueError("Le rayon de recherche doit être entre 0 et 10°.")
    if not collections:
        raise ValueError("Sélectionnez au moins une collection.")

    position = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []

    for collection in collections:
        query: dict[str, Any] = {
            "pos": (position, radius_deg * u.deg),
            "collection": collection,
        }
        if band_um:
            low, high = sorted(band_um)
            query["band"] = (low * u.um, high * u.um)
        if time_mjd:
            query["time"] = tuple(sorted(time_mjd))

        try:
            table = Irsa.query_sia(**query)
        except Exception as exc:
            errors.append(f"{collection}: {exc}")
            continue

        url_name = next((name for name in table.colnames if name.lower() == "access_url"), None)
        if url_name is None:
            errors.append(f"{collection}: colonne access_url absente")
            continue

        for row in table:
            record = {str(name): json_value(row[name]) for name in table.colnames}
            url = record.get(url_name)
            if not isinstance(url, str) or urlparse(url).scheme.lower() not in {"https", "http"}:
                continue
            if url in seen:
                continue
            seen.add(url)
            record["access_url"] = url
            record["collection"] = collection
            results.append(record)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    if not results and errors:
        raise RuntimeError("La requête IRSA a échoué. " + " | ".join(errors))
    return results


def download_original_fits(record: dict[str, Any]) -> Path:
    """Télécharge et valide le FITS, sans recoder ni modifier son contenu."""
    url = record.get("access_url")
    if not isinstance(url, str) or urlparse(url).scheme.lower() not in {"https", "http"}:
        raise ValueError("URL de téléchargement absente ou invalide.")

    FITS_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    suffix = ".fits.gz" if urlparse(url).path.lower().endswith(".fits.gz") else ".fits"
    destination = FITS_DIR / f"{digest}{suffix}"

    if destination.is_file() and destination.stat().st_size:
        try:
            with fits.open(destination, memmap=False) as hdus:
                hdus.verify("exception")
                return destination
        except Exception:
            destination.unlink(missing_ok=True)

    temporary: Path | None = None
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(20, 240),
            headers={"User-Agent": "SPHEREx-Explorer/1.0"},
        ) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                dir=FITS_DIR, suffix=".download", delete=False
            ) as handle:
                temporary = Path(handle.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        # Ouvre le fichier pour valider FITS; ne réécrit pas les HDU.
        with fits.open(temporary, memmap=False) as hdus:
            hdus.verify("exception")
            if not any(hdu.data is not None for hdu in hdus):
                raise ValueError("Le FITS téléchargé ne contient pas de données.")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def show_fits_preview(path: Path) -> None:
    """Affiche un aperçu; seul le rendu écran est étiré, jamais le FITS."""
    with fits.open(path, memmap=False) as hdus:
        selected = None
        for hdu in hdus:
            if hdu.data is None or np.ndim(hdu.data) < 2:
                continue
            array = np.asarray(hdu.data)
            if not np.issubdtype(array.dtype, np.number):
                continue
            selected = np.array(array, copy=True)
            while selected.ndim > 2:
                selected = selected[0]
            break

    if selected is None or selected.ndim != 2:
        raise ValueError("Aucune extension d’image 2D exploitable dans ce FITS.")
    finite = np.isfinite(selected)
    if not finite.any():
        raise ValueError("L’extension FITS ne contient aucun pixel fini.")

    # vmin/vmax influencent uniquement la palette du graphique, pas les pixels.
    vmin, vmax = np.nanpercentile(selected, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(selected)), float(np.nanmax(selected))
        if vmin == vmax:
            vmax = vmin + 1

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(
        selected,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, label="Valeur pixel (visualisation)")
    ax.set_title("Aperçu FITS — données sources inchangées")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


with st.form("query_form"):
    st.subheader("1. Rechercher des produits")
    col_ra, col_dec, col_radius = st.columns(3)
    with col_ra:
        ra = st.number_input("RA (degrés décimaux)", min_value=0.0, max_value=359.999999, value=10.6847, format="%.6f")
    with col_dec:
        dec = st.number_input("Dec (degrés décimaux)", min_value=-90.0, max_value=90.0, value=41.2688, format="%.6f")
    with col_radius:
        radius = st.number_input("Rayon (degrés)", min_value=0.001, max_value=10.0, value=0.5, step=0.1)

    labels = st.multiselect(
        "Collections IRSA",
        list(COLLECTIONS),
        default=["QR3 — Wide"],
    )
    with st.expander("Filtres facultatifs"):
        use_band = st.checkbox("Filtrer par plage spectrale (µm)")
        band_cols = st.columns(2)
        band_min = band_cols[0].number_input("Longueur d’onde min", min_value=0.0, value=0.75, step=0.1, disabled=not use_band)
        band_max = band_cols[1].number_input("Longueur d’onde max", min_value=0.0, value=5.0, step=0.1, disabled=not use_band)
        use_time = st.checkbox("Filtrer par plage temporelle MJD")
        time_cols = st.columns(2)
        time_min = time_cols[0].number_input("MJD min", value=60000.0, step=1.0, disabled=not use_time)
        time_max = time_cols[1].number_input("MJD max", value=61000.0, step=1.0, disabled=not use_time)
    limit = st.slider("Maximum de résultats", 1, 100, 25)
    submit = st.form_submit_button("🔎 Rechercher dans IRSA", type="primary", use_container_width=True)

if submit:
    try:
        selected = [COLLECTIONS[label] for label in labels]
        with st.spinner("Recherche dans l’archive IRSA…"):
            found = search_irsa(
                ra,
                dec,
                radius,
                selected,
                (band_min, band_max) if use_band else None,
                (time_min, time_max) if use_time else None,
                limit,
            )
        st.session_state["collection_results"] = found
        if found:
            added = merge_into_index(found)
            st.success(f"{len(found)} résultat(s) trouvé(s) ; {added} nouvelle(s) entrée(s) ajoutée(s) à l’index local.")
        else:
            st.warning("Aucun résultat renvoyé. Essayez une autre position, un rayon différent ou une autre collection.")
    except Exception as exc:
        st.error(f"Échec de la recherche : {exc}")

records = st.session_state.get("collection_results", [])
if records:
    st.subheader("2. Choisir, télécharger et visualiser un produit")

    def label_record(index: int) -> str:
        record = records[index]
        obs_id = record.get("obs_id") or record.get("obsid") or record.get("dp_id") or f"Produit {index + 1}"
        epoch = record.get("t_min") or record.get("date_obs") or record.get("time") or "époque inconnue"
        return f"{index + 1}. {record.get('collection', 'SPHEREx')} — {obs_id} — {epoch}"

    selected_index = st.selectbox("Produit IRSA", range(len(records)), format_func=label_record)
    current = records[selected_index]

    metadata_cols = st.columns(3)
    metadata_cols[0].metric("Collection", current.get("collection", "—"))
    metadata_cols[1].metric("Époque (MJD/date)", str(current.get("t_min") or current.get("date_obs") or "—"))
    metadata_cols[2].metric("Bande", str(current.get("energy_bandpassname") or current.get("band") or "—"))

    if st.button("⬇ Télécharger le FITS original et créer l’aperçu", type="primary"):
        try:
            with st.spinner("Téléchargement et validation du FITS…"):
                local_path = download_original_fits(current)
            st.session_state["active_fits_path"] = str(local_path)
            st.session_state["active_fits_url"] = current["access_url"]
            st.success(f"FITS original enregistré : {local_path.relative_to(BASE_DIR)}")
        except Exception as exc:
            st.error(f"Téléchargement/validation FITS impossible : {exc}")

    active_path = st.session_state.get("active_fits_path")
    if active_path and Path(active_path).is_file() and st.session_state.get("active_fits_url") == current.get("access_url"):
        try:
            show_fits_preview(Path(active_path))
            st.download_button(
                "Télécharger une copie du FITS original",
                data=Path(active_path).read_bytes(),
                file_name=Path(active_path).name,
                mime="application/fits",
            )
        except Exception as exc:
            st.error(f"Impossible de produire l’aperçu : {exc}")

    with st.expander("Métadonnées IRSA enregistrées dans la base locale"):
        st.json(current)

st.divider()
st.subheader("Base locale")
st.caption(f"Index : {INDEX_FILE.relative_to(BASE_DIR)} · Fichiers FITS : {FITS_DIR.relative_to(BASE_DIR)}")
local_records = read_index()
st.write(f"{len(local_records)} produit(s) catalogué(s). Les produits restent sur cette machine/instance Streamlit.")
if local_records:
    st.download_button(
        "Exporter l’index JSON",
        data=json.dumps({"observations": local_records}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="spherex_index.json",
        mime="application/json",
    )

st.caption("Source : NASA/IPAC IRSA SPHEREx. Les détections ou différences calculées après coup restent à vérifier scientifiquement.")
