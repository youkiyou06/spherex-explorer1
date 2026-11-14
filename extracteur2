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


def rechercher_image(ra, dec, rayon_deg=0.1, dossier="spherex_images"):
	"""Télécharge et renvoie le chemin du premier FITS image dans la région.

	ra, dec : ascension droite et déclinaison en degrés (ICRS/J2000).
	rayon_deg : rayon de recherche en degrés.
	"""
	if not (0 <= float(ra) < 360 and -90 <= float(dec) <= 90):
		raise ValueError("RA doit être dans [0, 360) et Dec dans [-90, 90].")
	if float(rayon_deg) <= 0:
		raise ValueError("Le rayon doit être supérieur à zéro.")

	# ObsCore décrit les produits d'archive ; la région est testée contre le cône.
	adql = f"""
		SELECT TOP 100 obs_collection, obs_id, obs_title, access_url
		FROM ivoa.ObsCore
		WHERE dataproduct_type = 'image'
		  AND (obs_collection LIKE '%SPHEREX%' OR obs_collection LIKE '%SPHEREx%')
		  AND INTERSECTS(s_region, CIRCLE('ICRS', {float(ra)}, {float(dec)}, {float(rayon_deg)})) = 1
	"""
	response = requests.get(
		TAP_URL,
		params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql},
		timeout=90,
	)
	response.raise_for_status()
	lignes = list(csv.DictReader(io.StringIO(response.text)))
	lignes = [ligne for ligne in lignes if ligne.get("access_url")]
	if not lignes:
		raise LookupError(
			"Aucune image trouvée. Vérifiez la couverture SPHEREx et les filtres "
			"ObsCore de l'archive IRSA."
		)

	produit = lignes[0]
	url = produit["access_url"]
	telechargement = requests.get(url, timeout=180, stream=True)
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
