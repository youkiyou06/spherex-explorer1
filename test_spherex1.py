from pathlib import Path
import sqlite3
import json
import hashlib

import numpy as np
import streamlit as st
import s3fs
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u

from reproject import reproject_interp
from streamlit_autorefresh import st_autorefresh
from scipy.ndimage import label


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE = Path(__file__).with_name("spherex_catalog_test.sqlite")
CACHE_DIR = Path(__file__).with_name("spherex_cache")
FITS_DIR = CACHE_DIR / "fits"
ALIGNED_DIR = CACHE_DIR / "aligned"

BUCKET = "nasa-irsa-spherex"
REGION = "us-east-1"

MAX_IMAGES = 4
DOWNSAMPLE_FACTOR = 4

# Deux détections sont considérées comme spatialement proches si leurs
# centres sont séparés d'au plus cette distance sur la grille alignée.
# À facteur 4, cela reste une association exploratoire, pas une mesure précise.
MATCH_RADIUS_PIXELS = 4.0

FITS_DIR.mkdir(parents=True, exist_ok=True)
ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="SPHEREx Explorer", layout="wide")
st.title("SPHEREx Explorer")
st.caption("Recherche, alignement, comparaison temporelle et détection expérimentale")


# ============================================================
# CONNEXION AWS
# ============================================================

@st.cache_resource
def get_s3_filesystem():
    return s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"region_name": REGION}
    )


# ============================================================
# GEOMETRIE DES EMPREINTES DU CATALOGUE
# ============================================================

def point_in_polygon(x, y, polygon):
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            intersection = (
                (xj - xi) * (y - yi) / ((yj - yi) + 1e-15) + xi
            )
            if x < intersection:
                inside = not inside
        j = i
    return inside


def footprint_contains(footprint_json, ra, dec):
    try:
        footprint = json.loads(footprint_json)
        if not isinstance(footprint, list) or len(footprint) < 3:
            return False

        coordinates = [
            (float(point[0]), float(point[1]))
            for point in footprint
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if len(coordinates) < 3:
            return False

        target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        vertices = SkyCoord(
            ra=[p[0] for p in coordinates] * u.deg,
            dec=[p[1] for p in coordinates] * u.deg,
            frame="icrs"
        )
        offsets = vertices.transform_to(SkyOffsetFrame(origin=target))
        polygon = list(zip(
            offsets.lon.to_value(u.deg),
            offsets.lat.to_value(u.deg)
        ))
        return point_in_polygon(0, 0, polygon)
    except (TypeError, ValueError, IndexError, json.JSONDecodeError):
        return False


# ============================================================
# RECHERCHE SQLITE
# ============================================================

@st.cache_data(show_spinner=False)
def search_observations(ra, dec):
    if not DATABASE.exists():
        raise FileNotFoundError(f"Base SQLite introuvable : {DATABASE}")

    connection = sqlite3.connect(DATABASE)
    try:
        rows = connection.execute("""
            SELECT s3_key, date_obs, mjd_obs, footprint, status
            FROM observations
            WHERE footprint IS NOT NULL
        """).fetchall()
    finally:
        connection.close()

    results = []
    for s3_key, date_obs, mjd_obs, footprint, status in rows:
        if str(status or "").strip().upper() not in ("OK", "SUCCESS"):
            continue
        if footprint_contains(footprint, ra, dec):
            results.append({
                "s3_key": s3_key,
                "date_obs": date_obs,
                "mjd_obs": mjd_obs
            })

    results.sort(
        key=lambda item: (
            item["mjd_obs"]
            if item["mjd_obs"] is not None else float("inf")
        )
    )
    return results[:MAX_IMAGES]


# ============================================================
# CACHE LOCAL FITS
# ============================================================

def local_fits_path(s3_key):
    digest = hashlib.sha256(s3_key.encode("utf-8")).hexdigest()
    return FITS_DIR / f"{digest}.fits"


def download_fits_if_needed(s3_key):
    local_path = local_fits_path(s3_key)
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    filesystem = get_s3_filesystem()
    key = s3_key.lstrip("/")
    remote_path = key if key.startswith("s3://") else f"{BUCKET}/{key}"
    temporary_path = local_path.with_suffix(".download")

    try:
        filesystem.get(remote_path, str(temporary_path))
        temporary_path.replace(local_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return local_path


# ============================================================
# LECTURE FITS A RESOLUTION REDUITE
# ============================================================

def load_reduced_image(s3_key):
    local_path = download_fits_if_needed(s3_key)

    with fits.open(local_path, memmap=False) as hdul:
        image_hdu = hdul["IMAGE"]
        image = np.asarray(
            image_hdu.data[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR],
            dtype=np.float32
        ).copy()

        original_wcs = WCS(image_hdu.header).celestial
        reduced_wcs = original_wcs.slice(
            (
                slice(None, None, DOWNSAMPLE_FACTOR),
                slice(None, None, DOWNSAMPLE_FACTOR)
            ),
            numpy_order=True
        )
        date_obs = image_hdu.header.get("DATE-OBS", "Date inconnue")

    return image, reduced_wcs, date_obs


def aligned_cache_path(s3_keys):
    signature = "|".join(s3_keys) + f"|factor={DOWNSAMPLE_FACTOR}"
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    return ALIGNED_DIR / f"{digest}.npz"


# ============================================================
# ALIGNEMENT ET CACHE
# ============================================================

def align_observations(s3_keys, ra, dec):
    cache_path = aligned_cache_path(s3_keys)
    _, reference_wcs, _ = load_reduced_image(s3_keys[0])

    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    target_x, target_y = reference_wcs.world_to_pixel(target)

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as saved:
                images = saved["images"]
                footprints = saved["footprints"]
                dates = saved["dates"]

            if len(images) == len(s3_keys):
                aligned = [
                    {
                        "image": images[i],
                        "footprint": footprints[i],
                        "date": str(dates[i]),
                        "key": s3_keys[i]
                    }
                    for i in range(len(images))
                ]
                return (
                    aligned, float(target_x), float(target_y),
                    True, reference_wcs
                )
        except Exception:
            try:
                cache_path.unlink()
            except OSError:
                pass

    loaded = []
    progress = st.progress(0)
    status_text = st.empty()

    for index, key in enumerate(s3_keys):
        status_text.write(f"Lecture FITS {index + 1}/{len(s3_keys)}…")
        image, image_wcs, date_obs = load_reduced_image(key)
        loaded.append({
            "image": image,
            "wcs": image_wcs,
            "date": date_obs,
            "key": key
        })
        progress.progress(int((index + 1) / len(s3_keys) * 30))

    reference_wcs = loaded[0]["wcs"]
    reference_shape = loaded[0]["image"].shape
    target_x, target_y = reference_wcs.world_to_pixel(target)

    aligned = []
    for index, observation in enumerate(loaded):
        status_text.write(f"Alignement {index + 1}/{len(loaded)}…")
        if index == 0:
            projected = observation["image"].copy()
            footprint = np.isfinite(projected).astype(np.float32)
        else:
            projected, footprint = reproject_interp(
                (observation["image"], observation["wcs"]),
                reference_wcs,
                shape_out=reference_shape,
                order="bilinear"
            )
            projected = projected.astype(np.float32)
            footprint = footprint.astype(np.float32)

        aligned.append({
            "image": projected,
            "footprint": footprint,
            "date": observation["date"],
            "key": observation["key"]
        })
        progress.progress(30 + int((index + 1) / len(loaded) * 65))

    temporary_cache = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_cache,
        images=np.stack([item["image"] for item in aligned]),
        footprints=np.stack([item["footprint"] for item in aligned]),
        dates=np.array([item["date"] for item in aligned])
    )
    temporary_cache.replace(cache_path)

    progress.progress(100)
    status_text.success("Alignement terminé et sauvegardé.")
    return aligned, float(target_x), float(target_y), False, reference_wcs


# ============================================================
# AFFICHAGE
# ============================================================

def normalize_for_display(image):
    valid = np.isfinite(image)
    if not np.any(valid):
        return np.zeros_like(image)
    values = image[valid]
    low, high = np.nanpercentile(values, [1, 99.5])
    if not np.isfinite(high) or high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def show_aligned_image(image, target_x, target_y, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(normalize_for_display(image), origin="lower", cmap="gray")
    ax.plot(
        target_x, target_y, marker="+", color="red",
        markersize=12, markeredgewidth=2
    )
    ax.set_title(title)
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    fig.tight_layout()
    st.pyplot(fig, width=550)
    plt.close(fig)


def show_difference_image(difference, valid, title):
    if difference is None or valid is None or not np.any(valid):
        st.warning("Aucune zone commune exploitable.")
        return

    shown = np.where(valid, difference, np.nan)
    values = shown[np.isfinite(shown)]
    limit = np.nanpercentile(np.abs(values), 99) if len(values) else 1
    if not np.isfinite(limit) or limit <= 0:
        limit = 1

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(
        shown, origin="lower", cmap="RdBu_r",
        vmin=-limit, vmax=limit
    )
    ax.set_title(title)
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    fig.tight_layout()
    st.pyplot(fig, width=550)
    plt.close(fig)


# ============================================================
# DETECTION D'ANOMALIES ENTRE DEUX OBSERVATIONS
# ============================================================

def detect_candidates(
    reference,
    current,
    reference_wcs,
    threshold_sigma=6.0,
    min_pixels=3
):
    valid = (
        np.isfinite(reference["image"])
        & np.isfinite(current["image"])
        & (reference["footprint"] > 0)
        & (current["footprint"] > 0)
    )

    if not np.any(valid):
        return [], None, valid

    difference = current["image"] - reference["image"]
    values = difference[valid]
    background = np.median(values)
    residual = difference - background
    residual_values = residual[valid]

    median_residual = np.median(residual_values)
    mad = np.median(np.abs(residual_values - median_residual))
    noise = 1.4826 * mad

    if not np.isfinite(noise) or noise <= 1e-12:
        noise = np.std(residual_values)
    if not np.isfinite(noise) or noise <= 1e-12:
        return [], difference, valid

    detection_mask = valid & (
        np.abs(residual) > threshold_sigma * noise
    )
    labels, number = label(
        detection_mask,
        structure=np.ones((3, 3), dtype=int)
    )

    candidates = []
    height, width = difference.shape

    for component_id in range(1, number + 1):
        ys, xs = np.where(labels == component_id)
        area = len(xs)
        if area < min_pixels or area > 100:
            continue
        if (
            np.any(xs < 3) or np.any(xs >= width - 3)
            or np.any(ys < 3) or np.any(ys >= height - 3)
        ):
            continue

        pixel_values = residual[ys, xs]
        weights = np.abs(pixel_values)
        if np.sum(weights) <= 0:
            continue

        center_x = float(np.average(xs, weights=weights))
        center_y = float(np.average(ys, weights=weights))
        significance = float(np.max(np.abs(pixel_values)) / noise)
        sky = reference_wcs.pixel_to_world(center_x, center_y)

        candidates.append({
            "RA (deg)": float(sky.ra.deg),
            "Dec (deg)": float(sky.dec.deg),
            "Pixels": int(area),
            "Significativité": significance,
            "Variation": (
                "Augmentation" if np.mean(pixel_values) > 0
                else "Diminution"
            ),
            "x": center_x,
            "y": center_y
        })

    candidates.sort(
        key=lambda item: item["Significativité"],
        reverse=True
    )
    return candidates[:30], difference, valid


def show_candidate_map(image, candidates, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(normalize_for_display(image), origin="lower", cmap="gray")
    for number, candidate in enumerate(candidates, start=1):
        ax.scatter(
            candidate["x"], candidate["y"],
            marker="o", facecolors="none",
            edgecolors="red", s=100
        )
        ax.annotate(
            str(number), (candidate["x"], candidate["y"]),
            color="yellow", fontsize=9
        )
    ax.set_title(title)
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    fig.tight_layout()
    st.pyplot(fig, width=550)
    plt.close(fig)


# ============================================================
# COMPARAISON DE TOUTES LES PAIRES SUCCESSIVES
# ============================================================

def compare_all_successive_pairs(
    aligned_data,
    reference_wcs,
    threshold_sigma,
    min_pixels
):
    pair_results = []
    all_detections = []

    for index in range(len(aligned_data) - 1):
        first = aligned_data[index]
        second = aligned_data[index + 1]

        candidates, difference, valid = detect_candidates(
            first,
            second,
            reference_wcs,
            threshold_sigma=threshold_sigma,
            min_pixels=min_pixels
        )

        pair_label = f"{index + 1} → {index + 2}"
        pair_results.append({
            "pair": pair_label,
            "first_date": first["date"],
            "second_date": second["date"],
            "candidates": candidates,
            "difference": difference,
            "valid": valid
        })

        for candidate in candidates:
            detection = dict(candidate)
            detection["pair"] = pair_label
            detection["pair_index"] = index
            all_detections.append(detection)

    # Regrouper les détections spatialement proches entre différentes paires.
    # Une paire ne peut contribuer qu'une fois à un même groupe.
    tracks = []
    for detection in sorted(
        all_detections,
        key=lambda item: item["Significativité"],
        reverse=True
    ):
        best_track = None
        best_distance = MATCH_RADIUS_PIXELS

        for track in tracks:
            if detection["pair_index"] in track["pair_indices"]:
                continue

            dx = detection["x"] - track["x"]
            dy = detection["y"] - track["y"]
            distance = float(np.hypot(dx, dy))

            if distance <= best_distance:
                best_distance = distance
                best_track = track

        if best_track is None:
            tracks.append({
                "x": detection["x"],
                "y": detection["y"],
                "detections": [detection],
                "pair_indices": {detection["pair_index"]}
            })
        else:
            best_track["detections"].append(detection)
            best_track["pair_indices"].add(detection["pair_index"])
            # Recalcul du centre moyen des détections associées.
            best_track["x"] = float(np.mean([
                item["x"] for item in best_track["detections"]
            ]))
            best_track["y"] = float(np.mean([
                item["y"] for item in best_track["detections"]
            ]))

    recurrent = []
    for track in tracks:
        detections = track["detections"]
        if len(detections) < 2:
            continue

        strongest = max(
            detections,
            key=lambda item: item["Significativité"]
        )
        sky = reference_wcs.pixel_to_world(track["x"], track["y"])

        recurrent.append({
            "RA (deg)": float(sky.ra.deg),
            "Dec (deg)": float(sky.dec.deg),
            "Paires détectées": len(detections),
            "Paires": ", ".join(
                item["pair"] for item in sorted(
                    detections, key=lambda item: item["pair_index"]
                )
            ),
            "Significativité max.": strongest["Significativité"],
            "x": track["x"],
            "y": track["y"]
        })

    recurrent.sort(
        key=lambda item: (
            item["Paires détectées"],
            item["Significativité max."]
        ),
        reverse=True
    )
    return pair_results, recurrent, all_detections


# ============================================================
# INTERFACE DE RECHERCHE
# ============================================================

st.sidebar.header("Recherche céleste")
ra = st.sidebar.number_input(
    "Ascension droite RA (degrés)",
    value=160.615237739,
    format="%.9f"
)
dec = st.sidebar.number_input(
    "Déclinaison Dec (degrés)",
    value=30.6690774567,
    format="%.9f"
)

if st.sidebar.button("Rechercher les observations"):
    st.session_state["search_coordinates"] = (float(ra), float(dec))
    for key in (
        "aligned_data", "target_x", "target_y", "reference_wcs",
        "current_index", "playing"
    ):
        st.session_state.pop(key, None)


if "search_coordinates" not in st.session_state:
    st.info("Entre les coordonnées puis lance la recherche.")
    st.stop()

search_ra, search_dec = st.session_state["search_coordinates"]

try:
    with st.spinner("Recherche dans le catalogue SQLite…"):
        observations = search_observations(search_ra, search_dec)
except Exception as error:
    st.error(f"Erreur de recherche : {error}")
    st.stop()

if not observations:
    st.warning("Aucune observation trouvée dans le catalogue indexé.")
    st.stop()

st.success(f"{len(observations)} observation(s) trouvée(s).")
st.subheader("Observations trouvées")
for index, observation in enumerate(observations):
    st.write(
        f"**{index + 1}.** {observation['date_obs']} "
        f"— `{observation['s3_key']}`"
    )

if len(observations) < 2:
    st.warning(
        "Il faut au moins deux observations pour comparer les changements."
    )
    st.stop()


# ============================================================
# TELECHARGEMENT ET ALIGNEMENT
# ============================================================

if "aligned_data" not in st.session_state:
    try:
        keys = tuple(item["s3_key"] for item in observations)
        with st.spinner("Téléchargement, lecture et alignement des images…"):
            (
                aligned_data, target_x, target_y,
                from_cache, reference_wcs
            ) = align_observations(keys, search_ra, search_dec)

        st.session_state["aligned_data"] = aligned_data
        st.session_state["target_x"] = target_x
        st.session_state["target_y"] = target_y
        st.session_state["reference_wcs"] = reference_wcs
        st.session_state["current_index"] = 0
        st.session_state["playing"] = False

        if from_cache:
            st.info("Images alignées récupérées depuis le cache local.")
    except Exception as error:
        st.error(f"Erreur pendant la préparation : {error}")
        st.stop()

aligned_data = st.session_state["aligned_data"]
target_x = st.session_state["target_x"]
target_y = st.session_state["target_y"]
reference_wcs = st.session_state["reference_wcs"]

st.success(f"{len(aligned_data)} images prêtes.")


# ============================================================
# NAVIGATION TEMPORELLE
# ============================================================

st.divider()
st.header("Exploration temporelle")

number_of_images = len(aligned_data)
if "current_index" not in st.session_state:
    st.session_state["current_index"] = 0
if "playing" not in st.session_state:
    st.session_state["playing"] = False

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("◀ Précédente", use_container_width=True):
        st.session_state["playing"] = False
        st.session_state["current_index"] = (
            st.session_state["current_index"] - 1
        ) % number_of_images
with col2:
    if st.button(
        "⏸ Pause" if st.session_state["playing"] else "▶ Lecture",
        use_container_width=True
    ):
        st.session_state["playing"] = not st.session_state["playing"]
with col3:
    if st.button("Suivante ▶", use_container_width=True):
        st.session_state["playing"] = False
        st.session_state["current_index"] = (
            st.session_state["current_index"] + 1
        ) % number_of_images

speed = st.slider(
    "Vitesse de lecture (millisecondes)",
    min_value=500, max_value=3000, value=1200, step=250
)

if st.session_state["playing"]:
    refresh_count = st_autorefresh(
        interval=speed,
        key="spherex_animation"
    )
    st.session_state["current_index"] = refresh_count % number_of_images

selected_index = st.slider(
    "Choisir une observation",
    min_value=0,
    max_value=number_of_images - 1,
    value=int(st.session_state["current_index"]),
    step=1
)
if selected_index != st.session_state["current_index"]:
    st.session_state["current_index"] = selected_index
    st.session_state["playing"] = False

current_index = int(st.session_state["current_index"])
current = aligned_data[current_index]

st.subheader(f"Observation {current_index + 1}/{number_of_images}")
st.write(f"**Date :** {current['date']}")
show_aligned_image(
    current["image"], target_x, target_y,
    f"Image alignée — {current['date']}"
)


# ============================================================
# ANALYSE COMPLETE DES PAIRES SUCCESSIVES
# ============================================================

st.divider()
st.header("Analyse de toutes les comparaisons successives")
st.write(
    "Le programme compare maintenant 1→2, 2→3, 3→4, etc. "
    "Il recherche ensuite les anomalies retrouvées à la même position "
    "dans plusieurs comparaisons."
)

threshold = st.slider(
    "Seuil de détection (écarts-types)",
    min_value=3.0, max_value=10.0, value=6.0, step=0.5
)
min_pixels = st.slider(
    "Nombre minimal de pixels par zone",
    min_value=2, max_value=10, value=3
)

with st.spinner("Analyse de toutes les paires successives…"):
    pair_results, recurrent_candidates, all_detections = (
        compare_all_successive_pairs(
            aligned_data,
            reference_wcs,
            threshold,
            min_pixels
        )
    )

st.subheader("Résultats par comparaison")
for result in pair_results:
    with st.expander(
        f"{result['pair']} — {result['first_date']} → {result['second_date']}"
    ):
        st.write(
            f"**{len(result['candidates'])} candidat(s)** détecté(s) "
            "dans cette comparaison."
        )
        show_difference_image(
            result["difference"],
            result["valid"],
            f"Différence {result['pair']}"
        )
        if result["candidates"]:
            rows = []
            for number, candidate in enumerate(result["candidates"], start=1):
                rows.append({
                    "N°": number,
                    "RA (deg)": round(candidate["RA (deg)"], 6),
                    "Dec (deg)": round(candidate["Dec (deg)"], 6),
                    "Pixels": candidate["Pixels"],
                    "Significativité": round(
                        candidate["Significativité"], 2
                    ),
                    "Variation": candidate["Variation"]
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun candidat avec les paramètres actuels.")

st.divider()
st.subheader("Candidats retrouvés dans plusieurs comparaisons")

if recurrent_candidates:
    st.success(
        f"{len(recurrent_candidates)} zone(s) retrouvée(s) "
        "dans au moins deux comparaisons."
    )
    recurrent_rows = []
    for number, candidate in enumerate(recurrent_candidates, start=1):
        recurrent_rows.append({
            "N°": number,
            "RA (deg)": round(candidate["RA (deg)"], 6),
            "Dec (deg)": round(candidate["Dec (deg)"], 6),
            "Paires détectées": candidate["Paires détectées"],
            "Comparaisons": candidate["Paires"],
            "Significativité max.": round(
                candidate["Significativité max."], 2
            )
        })
    st.dataframe(
        recurrent_rows,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(
        normalize_for_display(aligned_data[0]["image"]),
        origin="lower",
        cmap="gray"
    )
    for number, candidate in enumerate(recurrent_candidates, start=1):
        ax.scatter(
            candidate["x"], candidate["y"],
            marker="o", facecolors="none",
            edgecolors="red", s=110
        )
        ax.annotate(
            str(number), (candidate["x"], candidate["y"]),
            color="yellow", fontsize=9
        )
    ax.set_title("Zones récurrentes sur la grille de référence")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")
    fig.tight_layout()
    st.pyplot(fig, width=550)
    plt.close(fig)
else:
    st.info(
        "Aucune zone n'a été retrouvée dans au moins deux comparaisons "
        "avec les paramètres actuels. Cela ne prouve pas qu'il n'y a "
        "aucun objet variable."
    )

st.caption(
    "Attention : une zone récurrente n'est pas automatiquement un objet "
    "astronomique. Des défauts de détecteur, des erreurs de reprojection, "
    "des variations de fond, le bruit et la calibration peuvent produire "
    "des anomalies. Cette méthode est exploratoire et ne confirme aucune "
    "découverte."
)


# ============================================================
# VUE D'ENSEMBLE
# ============================================================

st.divider()
st.header("Vue d'ensemble des observations")
columns = st.columns(2)
for index, observation in enumerate(aligned_data):
    with columns[index % 2]:
        st.write(f"**Observation {index + 1}**")
        st.caption(observation["date"])
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(
            normalize_for_display(observation["image"]),
            origin="lower",
            cmap="gray"
        )
        ax.set_axis_off()
        st.pyplot(fig, width=220)
        plt.close(fig)


# ============================================================
# INFORMATIONS
# ============================================================

st.divider()
st.caption(
    "FITS locaux : spherex_cache/fits | Images alignées : "
    "spherex_cache/aligned"
)
st.caption(
    "SPHEREx Explorer — prototype expérimental d'analyse temporelle."
)
