# Pixel Patrol - Détecteur de falsifications d'images

## Projet étudiant reliant les UE HAI804I Analyse et traitement des images et HAI809I Codage et compression multimédia 

Pixel Patrol est une solution Python de détection de falsifications dans des images, utilisant des technologies  d'analyse d'images pour identifier les modifications et manipulations d'images. Notre objectif est de fournir un outil fiable pour aider à maintenir l'intégrité et la véracité des contenus visuels.

## Installation

Pour exécuter ce projet, vous aurez besoin de Python et de `pip` installés sur votre système. Il est recommandé d'utiliser un environnement virtuel pour gérer les dépendances.

### Configuration de l'environnement virtuel

1. **Créer un environnement virtuel** :
    ```sh
    python -m venv pixel_patrol_env
    ```
    Cette commande crée un nouvel environnement virtuel nommé `pixel_patrol_env` dans le répertoire courant.
<br/>
2. **Activer l'environnement virtuel** :
    - Sur Windows :
      ```sh
      .\pixel_patrol_env\Scripts\activate
      ```
    - Sur macOS et Linux :
      ```sh
      source pixel_patrol_env/bin/activate
      ```
    Une fois activé, votre invite de commande devrait vous avertir du changement d'environnement.

### Installation des dépendances

Avec l'environnement virtuel activé, installez les dépendances nécessaires à l'aide de `pip` :

```sh
pip install -r requirements.txt
```

## Utilisation

```sh
python3 src/app.py
```

La fenêtre de l'application devrait s'ouvrir. Vous pouvez y charger une image, lancer la détection de falsification et visualiser l'image résultante après traitement.

## Structure du projet

- `src/` : contient les scripts source de l'application.
    - `app.py` : implémentation de l'interface graphique
    - `detection.py` : logique de détection de falsification dans une image
    - `test.pgm` : contient la dernière image générée par le programme
- `CRs` : contient les compte-rendus qui détaillent les avancements du projet
- `.gitignore` : répertorie les fichiers et dossiers à ignorer lors des ajouts sur le git
- `requirements.txt` : regroupe les dépendances nécessaires à l'utilisation du projet