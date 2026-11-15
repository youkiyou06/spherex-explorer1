
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

# Facteur 4 : aperçu à résolution réduite.
# Une image 2040 x 2040 devient environ 510 x 510.
DOWNSAMPLE_FACTOR = 4

FITS_DIR.mkdir(parents=True, exist_ok=True)
ALIGNED_DIR.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="SPHEREx Explorer",
    layout="wide"
)

st.title("SPHEREx Explorer")
st.caption("Alignement et comparaison temporelle")


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
# GEOMETRIE DES EMPREINTES
# ============================================================

def point_in_polygon(x, y, polygon):

    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):

        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if (yi > y) != (yj > y):

            intersection = (
                (xj - xi) * (y - yi)
                / ((yj - yi) + 1e-15)
                + xi
            )

            if x < intersection:
                inside = not inside

        j = i

    return inside


def footprint_contains(footprint_json, ra, dec):

    try:
        footprint = json.loads(footprint_json)
    except (TypeError, json.JSONDecodeError):
        return False

    if not isinstance(footprint, list) or len(footprint) < 3:
        return False

    try:
        coordinates = [
            (float(point[0]), float(point[1]))
            for point in footprint
            if isinstance(point, (list, tuple))
            and len(point) >= 2
        ]
    except (TypeError, ValueError, IndexError):
        return False

    if len(coordinates) < 3:
        return False

    target = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs"
    )

    vertices = SkyCoord(
        ra=[p[0] for p in coordinates] * u.deg,
        dec=[p[1] for p in coordinates] * u.deg,
        frame="icrs"
    )

    offsets = vertices.transform_to(
        SkyOffsetFrame(origin=target)
    )

    polygon = list(zip(
        offsets.lon.to_value(u.deg),
        offsets.lat.to_value(u.deg)
    ))

    return point_in_polygon(0, 0, polygon)


# ============================================================
# RECHERCHE SQLITE
# ============================================================

@st.cache_data
def search_observations(ra, dec):

    if not DATABASE.exists():
        raise FileNotFoundError(
            f"Base SQLite introuvable : {DATABASE}"
        )

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

        normalized_status = (
            str(status).strip().upper()
            if status is not None else ""
        )

        if normalized_status not in ("OK", "SUCCESS"):
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
            if item["mjd_obs"] is not None
            else float("inf")
        )
    )

    return results[:MAX_IMAGES]


# ============================================================
# CACHE LOCAL DES FITS
# ============================================================

def local_fits_path(s3_key):

    filename = hashlib.sha256(
        s3_key.encode("utf-8")
    ).hexdigest() + ".fits"

    return FITS_DIR / filename


def download_fits_if_needed(s3_key):

    local_path = local_fits_path(s3_key)

    # Si le fichier existe déjà, aucun téléchargement.
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    filesystem = get_s3_filesystem()

    key = s3_key.lstrip("/")

    if key.startswith("s3://"):
        remote_path = key
    else:
        remote_path = f"{BUCKET}/{key}"

    temporary_path = local_path.with_suffix(".download")

    try:

        filesystem.get(
            remote_path,
            str(temporary_path)
        )

        temporary_path.replace(local_path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return local_path


# ============================================================
# LECTURE DES FITS EN RESOLUTION REDUITE
# ============================================================

def load_reduced_image(s3_key):

    local_path = download_fits_if_needed(s3_key)

    with fits.open(
        local_path,
        memmap=False
    ) as hdul:

        image_hdu = hdul["IMAGE"]

        # On ne garde qu'un pixel sur quatre dans chaque dimension.
        image = np.asarray(
            image_hdu.data[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR],
            dtype=np.float32
        ).copy()

        original_wcs = WCS(
            image_hdu.header
        ).celestial

        reduced_wcs = original_wcs.slice(
            (
                slice(None, None, DOWNSAMPLE_FACTOR),
                slice(None, None, DOWNSAMPLE_FACTOR)
            ),
            numpy_order=True
        )

        date_obs = image_hdu.header.get(
            "DATE-OBS",
            "Date inconnue"
        )

    return image, reduced_wcs, date_obs


# ============================================================
# IDENTIFIANT DU CACHE D'ALIGNEMENT
# ============================================================

def aligned_cache_path(s3_keys):

    signature = (
        "|".join(s3_keys)
        + f"|factor={DOWNSAMPLE_FACTOR}"
    )

    digest = hashlib.sha256(
        signature.encode("utf-8")
    ).hexdigest()

    return ALIGNED_DIR / f"{digest}.npz"


# ============================================================
# ALIGNEMENT AVEC SAUVEGARDE PERMANENTE
# ============================================================

def align_observations(s3_keys, ra, dec):

    cache_path = aligned_cache_path(s3_keys)

    # --------------------------------------------------------
    # CACHE DEJA CREE
    # --------------------------------------------------------

    if cache_path.exists():

        with np.load(
            cache_path,
            allow_pickle=False
        ) as saved:

            images = saved["images"]
            footprints = saved["footprints"]
            dates = saved["dates"]
            target_x = float(saved["target_x"])
            target_y = float(saved["target_y"])

        aligned = []

        for i in range(len(images)):

            aligned.append({
                "image": images[i],
                "footprint": footprints[i],
                "date": str(dates[i]),
                "key": s3_keys[i]
            })

        return aligned, target_x, target_y, True

    # --------------------------------------------------------
    # PREMIER ALIGNEMENT
    # --------------------------------------------------------

    loaded = []

    progress = st.progress(0)
    status_text = st.empty()

    for index, key in enumerate(s3_keys):

        status_text.write(
            f"Lecture de l'observation {index + 1}/{len(s3_keys)}..."
        )

        image, image_wcs, date_obs = load_reduced_image(key)

        loaded.append({
            "image": image,
            "wcs": image_wcs,
            "date": date_obs,
            "key": key
        })

        progress.progress(
            int((index + 1) / len(s3_keys) * 35)
        )

    reference = loaded[0]

    reference_wcs = reference["wcs"]
    reference_shape = reference["image"].shape

    aligned = []

    for index, observation in enumerate(loaded):

        status_text.write(
            f"Alignement de l'observation {index + 1}/{len(loaded)}..."
        )

        if index == 0:

            projected = observation["image"].copy()

            footprint = np.isfinite(
                projected
            ).astype(np.float32)

        else:

            projected, footprint = reproject_interp(
                (
                    observation["image"],
                    observation["wcs"]
                ),
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

        progress.progress(
            35 + int((index + 1) / len(loaded) * 60)
        )

    # Position de la cible dans la grille commune.
    target = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs"
    )

    target_x, target_y = reference_wcs.world_to_pixel(target)

    # Sauvegarde locale : les prochains lancements
    # n'auront plus besoin de refaire l'alignement.
    np.savez(
        cache_path,
        images=np.stack([
            item["image"] for item in aligned
        ]),
        footprints=np.stack([
            item["footprint"] for item in aligned
        ]),
        dates=np.array([
            item["date"] for item in aligned
        ]),
        target_x=float(target_x),
        target_y=float(target_y)
    )

    progress.progress(100)
    status_text.success("Alignement terminé et sauvegardé.")

    return aligned, float(target_x), float(target_y), False


# ============================================================
# AFFICHAGE
# ============================================================

def normalize_for_display(image):

    valid = np.isfinite(image)

    if not np.any(valid):
        return np.zeros_like(image)

    values = image[valid]

    low, high = np.nanpercentile(
        values,
        [1, 99.5]
    )

    if high <= low:
        return np.zeros_like(image)

    result = (image - low) / (high - low)

    return np.clip(result, 0, 1)


def show_aligned_image(image, target_x, target_y, title):

    normalized = normalize_for_display(image)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(
        normalized,
        origin="lower",
        cmap="gray"
    )

    ax.plot(
        target_x,
        target_y,
        marker="+",
        color="red",
        markersize=12,
        markeredgewidth=2
    )

    ax.set_title(title)
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")

    fig.tight_layout()

    st.pyplot(fig, width=550)

    plt.close(fig)


def show_difference(reference, current):

    valid = (
        np.isfinite(reference["image"])
        & np.isfinite(current["image"])
        & (current["footprint"] > 0)
        & (reference["footprint"] > 0)
    )

    if not np.any(valid):
        st.warning("Aucune zone commune exploitable.")
        return

    difference = np.full_like(
        reference["image"],
        np.nan,
        dtype=np.float32
    )

    difference[valid] = (
        current["image"][valid]
        - reference["image"][valid]
    )

    values = difference[valid]

    limit = np.nanpercentile(
        np.abs(values),
        99
    )

    if not np.isfinite(limit) or limit == 0:
        limit = 1

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.imshow(
        difference,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit
    )

    ax.set_title("Différence avec la première observation")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Pixels")

    fig.tight_layout()

    st.pyplot(fig, width=550)

    plt.close(fig)


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

    st.session_state["search_coordinates"] = (
        float(ra),
        float(dec)
    )

    for key in (
        "aligned_data",
        "target_x",
        "target_y",
        "current_index",
        "playing"
    ):
        st.session_state.pop(key, None)


if "search_coordinates" not in st.session_state:

    st.info("Entre les coordonnées puis lance la recherche.")
    st.stop()


search_ra, search_dec = st.session_state["search_coordinates"]


# ============================================================
# RECHERCHE
# ============================================================

try:

    with st.spinner("Recherche dans le catalogue..."):

        observations = search_observations(
            search_ra,
            search_dec
        )

except Exception as error:

    st.error(f"Erreur de recherche : {error}")
    st.stop()


if not observations:

    st.warning(
        "Aucune observation trouvée dans le catalogue indexé."
    )

    st.stop()


st.success(
    f"{len(observations)} observation(s) trouvée(s)."
)

st.subheader("Observations trouvées")

for index, observation in enumerate(observations):

    st.write(
        f"**{index + 1}.** {observation['date_obs']} "
        f"— `{observation['s3_key']}`"
    )


# ============================================================
# TELECHARGEMENT ET ALIGNEMENT
# ============================================================

if "aligned_data" not in st.session_state:

    try:

        keys = tuple(
            observation["s3_key"]
            for observation in observations
        )

        with st.spinner(
            "Préparation des images..."
        ):

            aligned_data, target_x, target_y, from_cache = (
                align_observations(
                    keys,
                    search_ra,
                    search_dec
                )
            )

            st.session_state["aligned_data"] = aligned_data
            st.session_state["target_x"] = target_x
            st.session_state["target_y"] = target_y

            st.session_state["current_index"] = 0
            st.session_state["playing"] = False

            if from_cache:
                st.success(
                    "Images alignées récupérées depuis le cache local."
                )

    except Exception as error:

        st.error(f"Erreur pendant la préparation : {error}")
        st.stop()


aligned_data = st.session_state["aligned_data"]

target_x = st.session_state["target_x"]
target_y = st.session_state["target_y"]

st.success(
    f"{len(aligned_data)} image(s) prêtes."
)


# ============================================================
# NAVIGATION TEMPORELLE
# ============================================================

st.divider()
st.header("Comparaison temporelle")

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

        st.session_state["playing"] = (
            not st.session_state["playing"]
        )


with col3:

    if st.button("Suivante ▶", use_container_width=True):

        st.session_state["playing"] = False

        st.session_state["current_index"] = (
            st.session_state["current_index"] + 1
        ) % number_of_images


speed = st.slider(
    "Vitesse de lecture (millisecondes)",
    min_value=500,
    max_value=3000,
    value=1200,
    step=250
)


if st.session_state["playing"] and number_of_images > 1:

    refresh_count = st_autorefresh(
        interval=speed,
        key="spherex_animation"
    )

    st.session_state["current_index"] = (
        refresh_count % number_of_images
    )


current_index = st.slider(
    "Choisir une observation",
    min_value=0,
    max_value=number_of_images - 1,
    value=st.session_state["current_index"],
    step=1
)

if current_index != st.session_state["current_index"]:

    st.session_state["current_index"] = current_index
    st.session_state["playing"] = False


current_index = st.session_state["current_index"]

current = aligned_data[current_index]


st.subheader(
    f"Observation {current_index + 1}/{number_of_images}"
)

st.write(f"**Date :** {current['date']}")

show_aligned_image(
    current["image"],
    target_x,
    target_y,
    f"Image alignée — {current['date']}"
)


# ============================================================
# COMPARAISON PAR DIFFERENCE
# ============================================================

st.divider()
st.header("Analyse des changements")

if current_index == 0:

    st.info(
        "La première observation sert de référence. "
        "Sélectionne une autre observation pour afficher "
        "la différence."
    )

else:

    st.write(
        "Différence pixel par pixel entre l'observation "
        "sélectionnée et la première observation."
    )

    show_difference(
        aligned_data[0],
        current
    )

    st.caption(
        "Cette comparaison est exploratoire. Les différences "
        "peuvent provenir du bruit, du fond, de la calibration "
        "ou d'effets instrumentaux. Elles ne prouvent pas "
        "la présence d'un nouvel objet."
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
# INFORMATIONS SUR LE CACHE
# ============================================================

st.divider()

st.caption(
    "Les FITS téléchargés sont conservés dans spherex_cache/fits. "
    "Les images alignées sont conservées dans spherex_cache/aligned."
)

st.caption(
    "SPHEREx Explorer — prototype de comparaison temporelle."
)
