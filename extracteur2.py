"""Recherche des produits image SPHEREx dans l'archive IRSA.

La page SPHEREx d'IRSA (https://irsa.ipac.caltech.edu/Missions/spherex.html)
présente l'accès aux données de la mission. Ce script interroge le TAP/ObsCore
d'IRSA, filtre les images par position céleste, puis télécharge le premier
produit correspondant. Il faut une connexion Internet et le paquet `requests`.
"""

import csv
import io
import re
from pathlib import Path
from urllib.parse import urlparse

import requests


TAP_URL = "https://irsa.ipac.caltech.edu/TAP/sync"


def _requete_tap(ra, dec, rayon_deg):
    """Construit et exécute la requête ADQL pour les images autour d'une position."""
    adql = f"""
        SELECT TOP 100 obs_collection, obs_id, obs_title, access_url
        FROM ivoa.ObsCore
        WHERE dataproduct_type = 'image'
          AND LOWER(obs_collection) LIKE '%spherex%'
          AND INTERSECTS(s_region, CIRCLE('ICRS', {float(ra)}, {float(dec)}, {float(rayon_deg)})) = 1
        ORDER BY obs_collection, obs_id
    """

    response = requests.get(
        TAP_URL,
        params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql},
        timeout=90,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    if not response.text.strip():
        raise LookupError("La requête TAP IRSA a renvoyé une réponse vide.")

    # Certaines réponses d'erreur du service sont aussi renvoyées en CSV.
    texte = response.text.upper()
    if "ERROR" in texte or "FAIL" in texte or "EXCEPTION" in texte:
        raise LookupError(f"Erreur de l'archive IRSA :\n{response.text[:1000]}")

    try:
        lignes = list(csv.DictReader(io.StringIO(response.text)))
    except csv.Error as exc:
        raise LookupError("La réponse IRSA n'est pas un CSV exploitable.") from exc

    lignes = [ligne for ligne in lignes if ligne.get("access_url")]
    return lignes


def rechercher_image(ra, dec, rayon_deg=0.1, dossier="spherex_images"):
    """Télécharge et renvoie le chemin du premier FITS image dans la région.

    ra, dec : ascension droite et déclinaison en degrés (ICRS/J2000).
    rayon_deg : rayon de recherche en degrés.
    """
    ra = float(ra)
    dec = float(dec)
    rayon_deg = float(rayon_deg)

    if not (0 <= ra < 360 and -90 <= dec <= 90):
        raise ValueError("RA doit être dans [0, 360) et Dec dans [-90, 90].")
    if rayon_deg <= 0:
        raise ValueError("Le rayon doit être supérieur à zéro.")

    lignes = _requete_tap(ra, dec, rayon_deg)

    if not lignes:
        # Fallback: la collection peut ne pas contenir 'spherex' exactement.
        adql_fallback = f"""
            SELECT TOP 100 obs_collection, obs_id, obs_title, access_url
            FROM ivoa.ObsCore
            WHERE dataproduct_type = 'image'
              AND INTERSECTS(s_region, CIRCLE('ICRS', {ra}, {dec}, {rayon_deg})) = 1
            ORDER BY obs_collection, obs_id
        """
        response = requests.get(
            TAP_URL,
            params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql_fallback},
            timeout=90,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        try:
            lignes = list(csv.DictReader(io.StringIO(response.text)))
        except csv.Error:
            lignes = []
        lignes = [ligne for ligne in lignes if ligne.get("access_url")]

    if not lignes:
        raise LookupError(
            "Aucune image trouvée. Vérifiez la couverture SPHEREx et les filtres "
            "ObsCore de l'archive IRSA. Le jeu de données peut ne pas être "
            "indexé pour cette région ou cette collection exacte."
        )

    produit = lignes[0]
    url = produit["access_url"].strip()

    telechargement = requests.get(
        url,
        timeout=180,
        stream=True,
        headers={"User-Agent": "Mozilla/5.0"},
        allow_redirects=True,
    )
    telechargement.raise_for_status()

    # Utilise le nom fourni par l'archive, avec un nom sûr en solution de repli.
    nom = Path(urlparse(url).path).name or "spherex_image.fits"
    nom = re.sub(r"[^A-Za-z0-9._-]", "_", nom)
    if not nom.lower().endswith((".fits", ".fits.gz", ".fit", ".fit.gz")):
        nom += ".fits"

    destination = Path(dossier)
    destination.mkdir(parents=True, exist_ok=True)
    chemin = destination / nom

    with chemin.open("wb") as fichier:
        for bloc in telechargement.iter_content(chunk_size=1024 * 1024):
            if bloc:
                fichier.write(bloc)

    return chemin


def main():
    print("Recherche d'image SPHEREx dans l'archive IRSA")
    ra = float(input("Ascension droite RA (degrés) : "))
    dec = float(input("Déclinaison Dec (degrés) : "))
    rayon = float(input("Rayon de recherche (degrés, défaut 0.1) [0.1] : ") or "0.1")
    chemin = rechercher_image(ra, dec, rayon)
    print(f"Image téléchargée : {chemin}")


if __name__ == "__main__":
    main()
