from io import BytesIO
from datetime import datetime, timezone

import requests
from astropy.table import Table
from astropy.time import Time


SIA_URL = "https://irsa.ipac.caltech.edu/SIA"

# Les quatre collections d'images spectrales
SPECTRAL_COLLECTIONS = [
    "spherex_dr1",
    "spherex_dr1_deep",
    "spherex_qr3",
    "spherex_qr3_deep",
]

# Collection distincte : cubes spectraux DR1
CUBE_COLLECTIONS = [
    "spherex_dr1_cube",
]

ALL_COLLECTIONS = SPECTRAL_COLLECTIONS + CUBE_COLLECTIONS

MAX_RESULTS_PER_COLLECTION = 5000
REQUEST_TIMEOUT = 90


def convert_date(value):
    """Convertit une date MJD ou ISO en texte lisible."""
    if value is None:
        return "Date inconnue"

    try:
        value = str(value).strip()

        if not value:
            return "Date inconnue"

        # Les dates SPHEREx sont généralement fournies en MJD.
        try:
            number = float(value)
            return Time(number, format="mjd").isot
        except (ValueError, TypeError):
            pass

        # Sinon, on tente une date ISO.
        try:
            return Time(value).isot
        except Exception:
            return value

    except Exception:
        return "Date inconnue"


def safe_value(row, column, default=""):
    """Récupère proprement une valeur dans une ligne de table."""
    try:
        value = row[column]

        if value is None:
            return default

        if hasattr(value, "mask") and value.mask:
            return default

        value = str(value).strip()

        if value.lower() in ("", "none", "nan", "--"):
            return default

        return value

    except Exception:
        return default


def search_collection(ra, dec, radius, collection):
    """
    Recherche les observations d'une collection SPHEREx
    autour des coordonnées demandées.

    ra et dec sont en degrés.
    radius est en degrés.
    """

    params = {
        "COLLECTION": collection,
        "POS": f"CIRCLE {ra} {dec} {radius}",
        "RESPONSEFORMAT": "VOTABLE",
        "MAXREC": MAX_RESULTS_PER_COLLECTION,
    }

    response = requests.get(
        SIA_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": "SPHEREx-Explorer/1.0"
        },
    )

    response.raise_for_status()

    table = Table.read(
        BytesIO(response.content),
        format="votable",
    )

    results = []

    for row in table:
        access_url = safe_value(row, "access_url")
        cloud_access = safe_value(row, "cloud_access")

        # Il faut au moins une adresse d'accès.
        if not access_url and not cloud_access:
            continue

        date = convert_date(
            safe_value(row, "t_min")
        )

        filename = safe_value(row, "obs_id")

        if not filename:
            filename = safe_value(row, "obs_publisher_did")

        if not filename:
            filename = access_url.split("/")[-1] if access_url else "Produit SPHEREx"

        results.append({
            "collection": collection,
            "filename": filename,
            "date": date,
            "access_url": access_url,
            "cloud_access": cloud_access,
            "is_cube": collection in CUBE_COLLECTIONS,
            "instrument": safe_value(row, "instrument_name"),
            "band": safe_value(row, "em_band"),
            "target": safe_value(row, "target_name"),
            "ra": safe_value(row, "s_ra"),
            "dec": safe_value(row, "s_dec"),
        })

    return results


def search_all_spherex(
    ra,
    dec,
    radius=0.05,
    include_deep=True,
    include_cubes=True,
):
    """
    Recherche les produits SPHEREx disponibles
    autour d'une position.

    Retourne :
        all_results : liste de tous les produits
        errors      : erreurs éventuelles par collection
    """

    collections = [
        "spherex_dr1",
        "spherex_qr3",
    ]

    if include_deep:
        collections.extend([
            "spherex_dr1_deep",
            "spherex_qr3_deep",
        ])

    if include_cubes:
        collections.append("spherex_dr1_cube")

    all_results = []
    errors = []

    for collection in collections:
        try:
            results = search_collection(
                ra=ra,
                dec=dec,
                radius=radius,
                collection=collection,
            )

            all_results.extend(results)

        except requests.exceptions.Timeout:
            errors.append(
                f"{collection} : délai d'attente dépassé."
            )

        except requests.exceptions.RequestException as error:
            errors.append(
                f"{collection} : erreur réseau : {error}"
            )

        except Exception as error:
            errors.append(
                f"{collection} : erreur : {error}"
            )

    # Évite les doublons si une même URL apparaît plusieurs fois.
    unique_results = []
    seen = set()

    for item in all_results:
        identifier = (
            item["access_url"]
            or item["cloud_access"]
            or item["filename"]
        )

        if identifier in seen:
            continue

        seen.add(identifier)
        unique_results.append(item)

    # Tri chronologique lorsque les dates sont reconnues.
    def sort_key(item):
        try:
            return Time(item["date"]).mjd
        except Exception:
            return float("inf")

    unique_results.sort(key=sort_key)

    return unique_results, errors


def split_products(results):
    """Sépare les images spectrales et les cubes."""
    spectral_images = []
    cubes = []

    for item in results:
        if item.get("is_cube"):
            cubes.append(item)
        else:
            spectral_images.append(item)

    return spectral_images, cubes