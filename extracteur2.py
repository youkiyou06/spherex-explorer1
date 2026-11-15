"""Recherche des produits FITS SPHEREx et les ouvre dans un navigateur.

Installation (Python 3.13 conseillé) :
    python -m pip install requests astroquery astropy matplotlib numpy

Le service SIA d'IRSA est interrogé par Astroquery. Les images FITS sont
converties en PNG en mémoire puis présentées dans une page HTML locale ouverte
dans le navigateur. Aucun fichier FITS n'est enregistré sur le disque.
"""

import base64
import html
import tempfile
import webbrowser
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
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


def _applique_filtre_temporel(resultats, time_mjd=None):
    """Filtre les résultats côté client si les colonnes temporelles existent."""
    if resultats is None or len(resultats) == 0 or time_mjd is None:
        return resultats

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


def _fits_vers_png(url, numero):
    """Télécharge un FITS en mémoire et retourne son image au format PNG."""
    with requests.get(url, timeout=(20, 180), stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        contenu = response.content

    if not contenu:
        raise ValueError("Le téléchargement du FITS est vide.")

    with fits.open(BytesIO(contenu), memmap=False) as hdul:
        image_hdu = next(
            (hdu for hdu in hdul if getattr(hdu, "data", None) is not None and np.ndim(hdu.data) >= 2),
            None,
        )
        if image_hdu is None:
            raise ValueError("Aucune image exploitable n'a été trouvée dans ce FITS.")

        image = np.asarray(image_hdu.data, dtype=np.float32)
        while image.ndim > 2:
            image = image[0]

        masque = np.isfinite(image)
        if not masque.any():
            raise ValueError("L'image ne contient aucune valeur finie.")

        vmin, vmax = np.percentile(image[masque], [1, 99])
        if vmin == vmax:
            vmin, vmax = np.min(image[masque]), np.max(image[masque])
            if vmin == vmax:
                vmin = vmax - 1

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        fig.colorbar(ax.images[0], ax=ax, label="Valeur du pixel")
        ax.set_title(f"Produit SPHEREx {numero}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        png = BytesIO()
        fig.savefig(png, format="png", bbox_inches="tight")
        plt.close(fig)
        return png.getvalue()


def _ouvrir_galerie(images):
    """Crée une galerie HTML temporaire avec les images intégrées et l'ouvre."""
    cartes = []
    for numero, (url, png) in enumerate(images, start=1):
        image_base64 = base64.b64encode(png).decode("ascii")
        titre = html.escape(f"Produit SPHEREx {numero}")
        lien = html.escape(url, quote=True)
        cartes.append(
            f'<section><h2>{titre}</h2>'
            f'<a href="{lien}" target="_blank" rel="noopener">'
            f'<img src="data:image/png;base64,{image_base64}" alt="{titre}"></a>'
            f"</section>"
        )

    page = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Images SPHEREx</title>
<style>
body { font-family: sans-serif; margin: 2rem auto; max-width: 1000px; padding: 0 1rem; background: #171717; color: #eee; }
section { margin: 2rem 0 3rem; }
img { display: block; max-width: 100%; height: auto; background: #000; }
a { color: #9cf; }
</style>
</head>
<body>
<h1>Images SPHEREx</h1>
""" + "\n".join(cartes) + "\n</body>\n</html>"

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".html", prefix="spherex_", delete=False
    ) as fichier:
        fichier.write(page)
        chemin_page = Path(fichier.name)

    if not webbrowser.open(chemin_page.as_uri()):
        print(f"Impossible d'ouvrir automatiquement le navigateur. Ouvrez : {chemin_page}", flush=True)
    else:
        print(f"Galerie ouverte dans le navigateur : {chemin_page}", flush=True)


def rechercher_image(ra, dec, rayon_deg=0.1, nombre_images=1, time_mjd=None):
    """Ouvre dans le navigateur jusqu'à nombre_images produits trouvés.

    time_mjd: tuple (min_mjd, max_mjd) ou None
    """
    ra, dec, rayon_deg = float(ra), float(dec), float(rayon_deg)
    nombre_images = int(nombre_images)
    if not (0 <= ra < 360 and -90 <= dec <= 90):
        raise ValueError("RA doit être dans [0, 360) et Dec dans [-90, 90].")
    if rayon_deg <= 0:
        raise ValueError("Le rayon doit être supérieur à zéro.")
    if nombre_images <= 0:
        raise ValueError("Le nombre d'images doit être supérieur à zéro.")

    position = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    query = {"pos": (position, rayon_deg * u.deg), "collection": COLLECTIONS}

    if time_mjd is not None:
        tmin, tmax = map(float, time_mjd)
        if tmin > tmax:
            tmin, tmax = tmax, tmin
        query["time"] = (tmin, tmax)

    print("Interrogation de l'archive IRSA (SIA)…", flush=True)
    if time_mjd is not None:
        print(f"Filtre temporel : MJD {time_mjd[0]}–{time_mjd[1]}", flush=True)

    try:
        resultats = Irsa.query_sia(**query)
    except Exception as exc:
        raise RuntimeError(
            "La requête SIA IRSA a échoué. Vérifiez Internet, les dépendances, "
            f"puis réessayez. Détail : {exc}"
        ) from exc

    resultats = _applique_filtre_temporel(resultats, time_mjd)
    nombre_resultats = len(resultats)
    print(f"Résultats IRSA : {nombre_resultats}", flush=True)
    if nombre_resultats == 0:
        raise LookupError(
            "Aucun produit SPHEREx ne couvre cette position avec les filtres demandés. "
            "Essayez un rayon plus grand ou une autre plage temporelle."
        )
    if "access_url" not in resultats.colnames:
        raise LookupError(f"La réponse IRSA ne contient pas access_url : {resultats.colnames}")

    images = []
    erreurs = []

    for numero, ligne in enumerate(resultats, start=1):
        if len(images) >= nombre_images:
            break

        valeur = ligne["access_url"]
        if valeur is None or getattr(valeur, "mask", False):
            continue
        url = str(valeur).strip()
        if not url or url.lower() == "masked":
            continue

        print(f"Récupération de l'image {len(images) + 1}/{nombre_images}…", flush=True)
        try:
            images.append((url, _fits_vers_png(url, numero)))
        except (requests.RequestException, OSError, ValueError) as exc:
            erreurs.append(f"{url}: {exc}")

    if not images:
        detail = "\n".join(erreurs[:5])
        raise LookupError(
            "Des résultats ont été trouvés, mais aucun ne contenait une image FITS exploitable.\n"
            + detail
        )

    _ouvrir_galerie(images)
    if len(images) < nombre_images:
        print(f"Seulement {len(images)} image(s) exploitable(s) trouvée(s).", flush=True)
    return [url for url, _ in images]


def main():
    print("Recherche d'images SPHEREx dans l'archive IRSA", flush=True)
    try:
        ra = float(input("Ascension droite RA (degrés) : "))
        dec = float(input("Déclinaison Dec (degrés) : "))
        rayon = float(input("Rayon de recherche en degrés [0.1] : ") or "0.1")
        nombre_images = int(input("Nombre d'images à ouvrir [1] : ") or "1")

        print("Filtre temporel (MJD) : min max [vide = ignorer]", flush=True)
        temps = input("  > ").strip()
        time_mjd = None if not temps else tuple(float(x) for x in temps.split())

        urls = rechercher_image(
            ra,
            dec,
            rayon,
            nombre_images=nombre_images,
            time_mjd=time_mjd,
        )
        print(f"{len(urls)} image(s) ouverte(s) dans le navigateur.", flush=True)
    except (ValueError, LookupError, RuntimeError) as exc:
        print(f"Échec : {exc}", flush=True)


if __name__ == "__main__":
    main()
