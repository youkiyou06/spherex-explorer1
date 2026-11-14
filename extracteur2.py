"""Recherche et téléchargement de produits FITS SPHEREx depuis IRSA.

Installation (Python 3.13 conseillé) :
    python -m pip install requests astroquery astropy

Le service SIA d'IRSA est interrogé par Astroquery. Les progrès sont affichés
pour distinguer une recherche en cours d'un programme bloqué.
"""

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.data import conf as astropy_conf
from astroquery.ipac.irsa import Irsa


COLLECTIONS = [
    "spherex_dr1",
    "spherex_qr3",
    "spherex_dr1_deep",
    "spherex_qr3_deep",
]

# Les requêtes d'archives peuvent être lentes.
astropy_conf.remote_timeout = 180


def _est_fits(debut):
    """Reconnaît l'en-tête FITS standard ou la signature gzip d'un FITS compressé."""
    return debut.startswith(b"SIMPLE  ") or debut.startswith(b"\x1f\x8b")


def rechercher_image(ra, dec, rayon_deg=0.1, dossier="spherex_images"):
    """Télécharge le premier produit FITS SPHEREx trouvé autour de la position."""
    ra, dec, rayon_deg = float(ra), float(dec), float(rayon_deg)
    if not (0 <= ra < 360 and -90 <= dec <= 90):
        raise ValueError("RA doit être dans [0, 360) et Dec dans [-90, 90].")
    if rayon_deg <= 0:
        raise ValueError("Le rayon doit être supérieur à zéro.")

    position = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    print("Interrogation de l'archive IRSA (SIA)…", flush=True)
    try:
        resultats = Irsa.query_sia(
            pos=(position, rayon_deg * u.deg),
            collection=COLLECTIONS,
        )
    except Exception as exc:
        raise RuntimeError(
            "La requête SIA IRSA a échoué. Vérifiez Internet, les dépendances, "
            f"puis réessayez. Détail : {exc}"
        ) from exc

    nombre = len(resultats)
    print(f"Résultats IRSA : {nombre}", flush=True)
    if nombre == 0:
        raise LookupError(
            "Aucun produit SPHEREx ne couvre cette position dans les collections "
            "DR1/QR3. Essayez une autre position ou un rayon supérieur."
        )
    if "access_url" not in resultats.colnames:
        raise LookupError(f"La réponse IRSA ne contient pas access_url : {resultats.colnames}")

    destination = Path(dossier)
    destination.mkdir(parents=True, exist_ok=True)
    erreurs = []

    for numero, ligne in enumerate(resultats, start=1):
        valeur = ligne["access_url"]
        if valeur is None or getattr(valeur, "mask", False):
            continue
        url = str(valeur).strip()
        if not url or url.lower() == "masked":
            continue

        print(f"Téléchargement du produit {numero}/{nombre}…", flush=True)
        chemin_temp = None
        try:
            with requests.get(url, timeout=(20, 180), stream=True, allow_redirects=True) as response:
                response.raise_for_status()
                nom = Path(unquote(urlparse(response.url).path)).name or f"spherex_{numero}.fits"
                nom = re.sub(r"[^A-Za-z0-9._-]", "_", nom)
                if not nom.lower().endswith((".fits", ".fits.gz", ".fit", ".fit.gz")):
                    nom += ".fits"
                chemin = destination / nom
                chemin_temp = chemin.with_name(chemin.name + ".part")

                debut = bytearray()
                with chemin_temp.open("wb") as fichier:
                    for bloc in response.iter_content(chunk_size=1024 * 1024):
                        if bloc:
                            if len(debut) < 80:
                                debut.extend(bloc[: 80 - len(debut)])
                            fichier.write(bloc)

            if not _est_fits(bytes(debut)):
                raise ValueError(
                    "La réponse téléchargée n'est pas un FITS "
                    f"(premiers octets : {bytes(debut[:80])!r})."
                )
            chemin_temp.replace(chemin)
            return chemin
        except (requests.RequestException, OSError, ValueError) as exc:
            erreurs.append(f"{url}: {exc}")
            if chemin_temp is not None:
                chemin_temp.unlink(missing_ok=True)

    detail = "\n".join(erreurs[:5])
    raise LookupError(
        "Des résultats ont été trouvés, mais aucun téléchargement n'a produit un FITS valide.\n"
        + detail
    )


def main():
    print("Recherche d'image SPHEREx dans l'archive IRSA", flush=True)
    try:
        ra = float(input("Ascension droite RA (degrés) : "))
        dec = float(input("Déclinaison Dec (degrés) : "))
        rayon = float(input("Rayon de recherche en degrés [0.1] : ") or "0.1")
        chemin = rechercher_image(ra, dec, rayon)
        print(f"Image FITS téléchargée et vérifiée : {chemin.resolve()}", flush=True)
    except (ValueError, LookupError, RuntimeError) as exc:
        print(f"Échec : {exc}", flush=True)


if __name__ == "__main__":
    main()
