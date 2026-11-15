"""Launcher Streamlit qui installe astroquery s'il manque, puis exécute testDB2.

À lancer depuis le dossier du dépôt avec :
    python -m streamlit run testDB2_auto.py

Cette installation automatique nécessite un environnement Python local avec pip.
Elle n'est pas prise en charge par les exécuteurs Pyodide dans le navigateur.
"""

from __future__ import annotations

import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path


if importlib.util.find_spec("astroquery") is None:
    print("Le module astroquery est absent : installation en cours…", flush=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "astroquery"]
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise SystemExit(
            "Installation automatique d'astroquery impossible. "
            "Lancez localement : python -m pip install astroquery. "
            "Les environnements Pyodide dans le navigateur ne prennent pas "
            "en charge cette installation pip."
        ) from error

script = Path(__file__).with_name("testDB2")
if not script.is_file():
    raise SystemExit(f"Fichier principal introuvable : {script}")

# run_name différent de __main__ pour éviter d'exécuter aussi le bloc de test
# en ligne de commande contenu dans testDB2 avant d'afficher l'interface.
runpy.run_path(str(script))
