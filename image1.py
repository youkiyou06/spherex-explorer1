"""Recherche et ouverture de produits FITS SPHEREx dans le navigateur.

Installation (Python 3.13 conseillé) :
    python -m pip install requests astroquery astropy matplotlib numpy

Le service SIA d'IRSA est interrogé par Astroquery. Les FITS sont convertis en PNG en mémoire puis ouverts dans une page HTML locale. La galerie affiche aussi les métadonnées fournies par IRSA et les en-têtes FITS. Aucun fichier FITS n'est enregistré sur le disque.
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
from astropy.wcs import WCS
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


def _texte_valeur(valeur):
    """Convertit proprement les valeurs FITS/Astropy en texte affichable."""
    if np.ma.is_masked(valeur):
        return "—"
    if isinstance(valeur, np.generic):
        valeur = valeur.item()
    if isinstance(valeur, bytes):
        return valeur.decode("utf-8", errors="replace").strip()
    if valeur is None:
        return "—"
    return str(valeur)


def _metadonnees_irsa(ligne):
    """Retourne toutes les colonnes du résultat SIA d'IRSA sous forme de lignes."""
    return [(str(nom), _texte_valeur(ligne[nom])) for nom in ligne.colnames]


def _fits_vers_png(url, numero, metadata_irsa):
    """Télécharge un FITS en mémoire et retourne son PNG et ses métadonnées."""
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
        forme_originale = image.shape
        while image.ndim > 2:
            image = image[0]

        masque = np.isfinite(image)
        if not masque.any():
            raise ValueError("L'image ne contient aucune valeur finie.")

        valeurs = image[masque]
        vmin, vmax = np.percentile(valeurs, [1, 99])
        if vmin == vmax:
            vmin, vmax = np.min(valeurs), np.max(valeurs)
            if vmin == vmax:
                vmin = vmax - 1

        metadonnees = [
            ("URL du produit", url),
            ("Extension image sélectionnée", f"{image_hdu.name} (HDU {hdul.index_of(image_hdu)})"),
            ("Forme des données FITS", str(forme_originale)),
            ("Dimensions affichées (pixels)", f"{image.shape[1]} × {image.shape[0]}"),
            ("Type de données FITS", str(image_hdu.data.dtype)),
            ("Pixels finis", f"{valeurs.size:,}"),
            ("Pixels non finis (NaN/Inf)", f"{image.size - valeurs.size:,}"),
            ("Minimum / maximum", f"{np.min(valeurs):.7g} / {np.max(valeurs):.7g}"),
            ("Moyenne / médiane", f"{np.mean(valeurs):.7g} / {np.median(valeurs):.7g}"),
            ("Écart-type", f"{np.std(valeurs):.7g}"),
            ("Percentiles 1 / 50 / 99", " / ".join(f"{v:.7g}" for v in np.percentile(valeurs, [1, 50, 99]))),
            ("Échelle d'affichage (percentiles 1–99)", f"{vmin:.7g} – {vmax:.7g}"),
        ]

        # Si l'en-tête contient une WCS céleste, fournit aussi les coordonnées
        # du centre de l'image (les mots-clés WCS bruts restent dans l'en-tête).
        try:
            wcs = WCS(image_hdu.header).celestial
            if wcs.has_celestial:
                centre = wcs.pixel_to_world((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)
                if isinstance(centre, SkyCoord):
                    metadonnees.append(("Centre WCS (ICRS)", centre.icrs.to_string("hmsdms", precision=2)))
                    metadonnees.append(("Centre WCS (degrés)", f"RA {centre.icrs.ra.deg:.8f}°, Dec {centre.icrs.dec.deg:.8f}°"))
        except Exception:
            # Une WCS absente ou non standard ne doit pas empêcher l'affichage.
            pass

        entetes = []
        for indice_hdu, hdu in enumerate(hdul):
            lignes = []
            for carte in hdu.header.cards:
                cle = str(carte.keyword).strip()
                if not cle:
                    continue
                lignes.append((cle, _texte_valeur(carte.value), _texte_valeur(carte.comment)))
            entetes.append((f"{hdu.name} — HDU {indice_hdu} ({type(hdu).__name__})", lignes))

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label="Valeur du pixel")
        ax.set_title(f"Produit SPHEREx {numero}")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        png = BytesIO()
        fig.savefig(png, format="png", bbox_inches="tight")
        plt.close(fig)

        return png.getvalue(), metadonnees, entetes, metadata_irsa


def _table_html(lignes, entetes_colonnes=("Champ", "Valeur")):
    """Construit un tableau HTML en échappant les valeurs externes."""
    entete = "".join(f"<th>{html.escape(str(nom))}</th>" for nom in entetes_colonnes)
    corps = []
    for ligne in lignes:
        cellules = "".join(f"<td>{html.escape(_texte_valeur(valeur))}</td>" for valeur in ligne)
        corps.append(f"<tr>{cellules}</tr>")
    return f"<div class=\"table-wrap\"><table><thead><tr>{entete}</tr></thead><tbody>{''.join(corps)}</tbody></table></div>"


def _ouvrir_galerie(images):
    """Crée une galerie HTML temporaire avec images et métadonnées."""
    cartes = []
    for numero, (url, png, metadonnees, entetes, metadata_irsa) in enumerate(images, start=1):
        image_base64 = base64.b64encode(png).decode("ascii")
        titre = html.escape(f"Produit SPHEREx {numero}")
        lien = html.escape(url, quote=True)

        sections = [
            f"<details open><summary>Métadonnées de l'image</summary>{_table_html(metadonnees)}</details>",
            f"<details><summary>Métadonnées du résultat IRSA ({len(metadata_irsa)} champs)</summary>{_table_html(metadata_irsa)}</details>",
        ]
        for nom_hdu, lignes in entetes:
            sections.append(
                f"<details><summary>En-tête FITS : {html.escape(nom_hdu)} ({len(lignes)} cartes)</summary>"
                f"{_table_html(lignes, ('Mot-clé FITS', 'Valeur', 'Commentaire'))}</details>"
            )

        cartes.append(
            f'<section><h2>{titre}</h2>'
            f'<a href="{lien}" target="_blank" rel="noopener">'
            f'<img src="data:image/png;base64,{image_base64}" alt="{titre}"></a>'
            f'{"".join(sections)}</section>'
        )

    page = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Images SPHEREx et métadonnées</title>
<style>
body { font-family: sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; background: #171717; color: #eee; }
section { margin: 2rem 0 3rem; padding-bottom: 2rem; border-bottom: 1px solid #555; }
img { display: block; max-width: 100%; height: auto; background: #000; margin-bottom: 1rem; }
a { color: #9cf; }
details { margin: .6rem 0; border: 1px solid #444; border-radius: 5px; padding: .6rem; }
summary { cursor: pointer; font-weight: bold; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin-top: .6rem; font-size: .9rem; }
th, td { text-align: left; vertical-align: top; border: 1px solid #444; padding: .35rem .5rem; overflow-wrap: anywhere; }
th { background: #292929; }
tr:nth-child(even) { background: #202020; }
</style>
</head>
<body>
<h1>Images SPHEREx</h1>
<p>Cliquez sur une image pour ouvrir le produit FITS d'origine. Les en-têtes FITS complets et les champs renvoyés par IRSA sont consultables ci-dessous.</p>
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
            png, metadonnees, entetes, metadata_irsa = _fits_vers_png(
                url, numero, _metadonnees_irsa(ligne)
            )
            images.append((url, png, metadonnees, entetes, metadata_irsa))
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
    return [url for url, *_ in images]


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
