Ce dépôt concerne un projet de recherche de Master 1 en intéllignece artificielle et apprentissage automatique (TER) qui consiste à réaliser un benchmarking de la robustesse de différents agrégateurs en apprentissage fédéré, à travers des simulations avec la bibliothèque Byzfl.

Le projet est organisé en 3 répertoires :

    Bilans_hebdomadaires : Contient les bilans de chaque semaine de travail (fichiers PDF et code source LaTeX).

    Comptes_rendus_reunions : Contient les comptes rendus de chaque réunion avec nos tuteurs (fichiers PDF et code source LaTeX).

    code : Contient toutes nos expériences (scénarios d'attaques byzantines, agrégateurs testés et scripts de génération des courbes de convergence).

Nous avons opté pour l'outil "UV" pour la configuration Python et la reproductibilité de nos expériences. Une fois le dépôt téléchargé, voici la procédure à suivre :

Installer uv (s'il n'est pas déjà présent sur votre machine) :
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
Activer l'environnement virtuel avec les deux commandes  suivantes (depuis la racine du projet):
    uv venv
    source .venv/bin/activate

Installer les dépendances exactes du projet (le fichier "uv.lock" se trouve dans le dossier racine du projet):
    uv pip install -r uv.lock