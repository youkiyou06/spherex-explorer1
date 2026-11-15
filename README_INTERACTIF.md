# SPHEREx Explorer — version interactive

Ce dépôt contient une application Streamlit autonome dans `programme_interactif.py`. Elle permet de rechercher des observations dans les archives IRSA, d'afficher les images FITS et de comparer deux observations.

## Lancer l'application sur son ordinateur

Prérequis : Python 3.10 ou plus récent et une connexion Internet (l'application interroge IRSA et télécharge les fichiers FITS à la demande).

1. Clonez le dépôt et ouvrez son dossier :

   ```bash
   git clone https://github.com/youkiyou06/SPHEREx-Explorer.git
   cd SPHEREx-Explorer
   ```

   Si le dépôt reste privé, il faut avoir reçu un accès GitHub pour le cloner.

2. Créez et activez un environnement virtuel :

   **macOS / Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows PowerShell**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Installez les dépendances :

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements_interactif.txt
   ```

4. Démarrez l'application :

   ```bash
   python -m streamlit run programme_interactif.py
   ```

   Ouvrez dans votre navigateur l'adresse locale affichée par Streamlit (généralement `http://localhost:8501`).

Les fichiers FITS récupérés sont mis en cache dans `data/` à côté du programme.

## Accès par les collaborateurs

Le code est disponible sur la branche principale du dépôt. Les collaborateurs ayant accès à ce dépôt privé peuvent le consulter et lancer l'application localement. Pour un lien web utilisable sans installation, il faut déployer l'application sur un service d'hébergement et configurer son accès ; ce changement ne rend pas le dépôt public.

L'application interroge l'archive IRSA et télécharge les observations choisies. Les téléchargements peuvent prendre du temps et consommer de la bande passante. Les résultats de comparaison sont exploratoires et ne constituent pas une confirmation de sources astronomiques.
