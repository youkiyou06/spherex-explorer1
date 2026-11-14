import requests
import xml.etree.ElementTree as ET

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import (
    urlsplit, urlunsplit, parse_qsl, urlencode
)

SIA_URL = "https://irsa.ipac.caltech.edu/SIA"


def date_to_mjd(date):
    """Convertit une date YYYY-MM-DD en MJD."""
    dt = datetime.strptime(date, "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)

    return (dt - epoch).total_seconds() / 86400


def search_images(ra, dec, search_radius=0.01):
    """Recherche les images SPHEREx autour des coordonnées."""

    collections = [
        "spherex_dr1",
        "spherex_qr3"
    ]

    observations = []

    for collection in collections:

        params = {
            "COLLECTION": collection,
            "POS": f"CIRCLE {ra} {dec} {search_radius}"
        }

        response = requests.get(
            SIA_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        root = ET.fromstring(response.content)

        # Recherche de la table VOTable
        table = next(
            (
                element for element in root.iter()
                if element.tag.endswith("TABLE")
            ),
            None
        )

        if table is None:
            continue

        # Récupération des noms de colonnes
        fields = [
            field.get("name", field.get("ID", "")).lower()
            for field in table
            if field.tag.endswith("FIELD")
        ]

        # Lecture des observations
        for tr in table.iter():

            if not tr.tag.endswith("TR"):
                continue

            values = [
                td.text or ""
                for td in tr
                if td.tag.endswith("TD")
            ]

            row = dict(zip(fields, values))

            if row.get("access_url"):
                observations.append(row)

    return observations


def extract_image(
    ra,
    dec,
    date=None,
    detector=None,
    size=0.1,
    search_radius=0.01,
    output="spherex_image.fits"
):
    """Recherche et télécharge une image SPHEREx."""

    # Vérification des coordonnées
    if not 0 <= ra < 360:
        raise ValueError("RA doit être comprise entre 0 et 360.")

    if not -90 <= dec <= 90:
        raise ValueError("DEC doit être comprise entre -90 et 90.")

    if size <= 0 or search_radius <= 0:
        raise ValueError("Les tailles doivent être positives.")

    # Vérification du détecteur
    if detector is not None:
        detector = detector.upper()

        if detector not in {f"D{i}" for i in range(1, 7)}:
            raise ValueError("Détecteur invalide : D1 à D6.")

    # Recherche des observations
    observations = search_images(
        ra,
        dec,
        search_radius
    )

    # Filtrage par détecteur
    if detector:
        observations = [
            obs for obs in observations
            if obs.get(
                "energy_bandpassname", ""
            ).upper().endswith(detector)
        ]

    if not observations:
        raise ValueError("Aucune observation correspondante.")

    # Sélection selon la date
    if date:

        requested_mjd = date_to_mjd(date)

        observations = [
            obs for obs in observations
            if obs.get("t_min", "").strip()
        ]

        if not observations:
            raise ValueError("Aucune date disponible.")

        selected = min(
            observations,
            key=lambda obs: abs(
                float(obs["t_min"]) - requested_mjd
            )
        )

    else:

        selected = max(
            observations,
            key=lambda obs: float(
                obs.get("t_min", "-1")
            )
        )

    # Construction de l'URL de découpe
    original_url = selected["access_url"]

    parts = urlsplit(original_url)

    query = dict(parse_qsl(parts.query))

    query.update({
        "center": f"{ra},{dec}",
        "size": str(size)
    })

    cutout_url = urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(query),
        parts.fragment
    ))

    # Téléchargement
    response = requests.get(
        cutout_url,
        timeout=180
    )

    response.raise_for_status()

    # Vérification élémentaire du fichier FITS
    if not response.content.startswith(b"SIMPLE  ="):
        raise ValueError(
            "Le serveur n'a pas retourné un fichier FITS valide."
        )

    # Enregistrement
    output = Path(output)

    output.write_bytes(response.content)

    print("Extraction réussie.")
    print("Observation :", selected.get("obs_id"))
    print("Détecteur :", selected.get("energy_bandpassname"))
    print("Date MJD :", selected.get("t_min"))
    print("Fichier :", output.resolve())

    return output
