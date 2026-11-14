"""Recherche et téléchargement de produits FITS SPHEREx depuis IRSA.

Installation (Python 3.13 conseillé) :
    python -m pip install requests astroquery astropy

Le service SIA d'IRSA est interrogé par Astroquery. Les progrès sont affichés
pour distinguer une recherche en cours d'un programme bloqué.

Le script supporte maintenant un filtrage additionnel sur la bande spectrale
(BAND, en µm) et sur la plage temporelle (TIME, en MJD).
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


def _applique_filtres_resultats(resultats, band_wave_um=None, time_mjd=None):
    """Filtre les résultats côté client si les colonnes sont disponibles."""
    if resultats is None or len(resultats) == 0:
        return resultats

    if band_wave_um is not None:
        bmin, bmax = map(float, band_wave_um)
        if bmin > bmax:
            bmin, bmax = bmax, bmin
        if "em_min" in resultats.colnames and "em_max" in resultats.colnames:
            masque = []
            for ligne in resultats:
                em_min = ligne["em_min"]
                em_max = ligne["em_max"]
                if em_min is None or em_max is None:
                    masque.append(False)
                    continue
                overlap = not (float(em_max) < bmin or float(em_min) > bmax)
                masque.append(overlap)
            resultats = resultats[masque]

    if time_mjd is not None:
        tmin, tmax = map(float, time_mjd)
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        if "t_min" in resultats.colnames and "t_max" in resultats.colnames:
            masque = []
            for ligne in resultats:
                tt_min = ligne["t_min"]
                tt_max = ligne["t_max"]
                if tt_min is None or tt_max is None:
                    masque.append(False)
                    continue
                overlap = not (float(tt_max) < tmin or float(tt_min) > tmax)
                masque.append(overlap)
            resultats = resultats[masque]

    return resultats


def rechercher_image(ra, dec, rayon_deg=0.1, dossier="spherex_images", band_wave_um=None, time_mjd=None):
    """Télécharge le premier produit FITS SPHEREx trouvé autour de la position.

    band_wave_um: tuple (min_um, max_um) ou None
    time_mjd: tuple (min_mjd, max_mjd) ou None
    """
    ra, dec, rayon_deg = float(ra), float(dec), float(rayon_deg)
    if not (0 <= ra < 360 and -90 <= dec <= 90):
        raise ValueError("RA doit être dans [0, 360) et Dec dans [-90, 90].")
    if rayon_deg <= 0:
        raise ValueError("Le rayon doit être supérieur à zéro.")

    position = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    query = {"pos": (position, rayon_deg * u.deg), "collection": COLLECTIONS}

    if band_wave_um is not None:
        bmin, bmax = map(float, band_wave_um)
        if bmin > bmax:
            bmin, bmax = bmax, bmin
        query["band"] = (bmin * u.um, bmax * u.um)
    if time_mjd is not None:
        tmin, tmax = map(float, time_mjd)
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        query["time"] = (tmin, tmax)

    print("Interrogation de l'archive IRSA (SIA)…", flush=True)
    if band_wave_um is not None:
        print(f"Filtre spectrale : {band_wave_um[0]}–{band_wave_um[1]} µm", flush=True)
    if time_mjd is not None:
        print(f"Filtre temporel : MJD {time_mjd[0]}–{time_mjd[1]}", flush=True)

    try:
        resultats = Irsa.query_sia(**query)
    except Exception as exc:
        raise RuntimeError(
            "La requête SIA IRSA a échoué. Vérifiez Internet, les dépendances, "
            f"puis réessayez. Détail : {exc}"
        ) from exc

    resultats = _applique_filtres_resultats(resultats, band_wave_um, time_mjd)
    nombre = len(resultats)
    print(f"Résultats IRSA : {nombre}", flush=True)
    if nombre == 0:
        raise LookupError(
            "Aucun produit SPHEREx ne couvre cette position avec les filtres demandés. "
            "Essayez un rayon plus grand ou des paramètres spectrales/temporels différents."
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

        # Filtre optionnel spectral et temporel.
        print("Filtre spectrale (µm) : min max [vide = ignorer]", flush=True)
        bande = input("  > ").strip()
        band_wave_um = None if not bande else tuple(float(x) for x in bande.split())

        print("Filtre temporel (MJD) : min max [vide = ignorer]", flush=True)
        temps = input("  > ").strip()
        time_mjd = None if not temps else tuple(float(x) for x in temps.split())

        chemin = rechercher_image(ra, dec, rayon, band_wave_um=band_wave_um, time_mjd=time_mjd)
        print(f"Image FITS téléchargée et vérifiée : {chemin.resolve()}", flush=True)
    except (ValueError, LookupError, RuntimeError) as exc:
        print(f"Échec : {exc}", flush=True)


if __name__ == "__main__":
    main()
