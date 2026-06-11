Ce dépôt concerne un projet de recherche de Master 1 en intéllignece artificielle et apprentissage automatique (TER) qui consiste à réaliser un benchmarking de la robustesse de différents agrégateurs en apprentissage fédéré, à travers des simulations avec la bibliothèque Byzfl.

Le projet est organisé en 3 répertoires :

    Bilans_hebdomadaires : Contient les bilans de chaque semaine de travail (fichiers PDF et code source LaTeX).

    Comptes_rendus_reunions : Contient les comptes rendus de chaque réunion avec nos tuteurs (fichiers PDF et code source LaTeX).

    code : Contient toutes nos expériences (scénarios d'attaques byzantines, agrégateurs testés et scripts de génération des courbes de convergence).

    data : contient tous les jeux de données utilisés lors de nos expériences.

Nous avons opté pour l'outil "UV" pour la configuration Python et la reproductibilité de nos expériences. Une fois le dépôt téléchargé, voici la procédure à suivre :

Installer uv (s'il n'est pas déjà présent sur votre machine) :

    curl -LsSf https://astral.sh/uv/install.sh | sh
    
Activer l'environnement virtuel avec les deux commandes  suivantes (depuis la racine du projet):

    uv venv --python python3.12
    source .venv/bin/activate

Installer les dépendances exactes du projet (le fichier "uv.lock" se trouve dans le dossier racine du projet):

    uv pip install -r uv.lock

## Organisation des scripts

Les scripts exécutables sont placés une seule fois dans `code/` avec des noms français explicites.

Structure principale :

- `code/moteur_experiences.py` : moteur d'exécution des expériences.
- `code/lancer_experiences.py` : lancement des scénarios JSON.
- `code/comparer_experiences.py` : comparaison globale des courbes d'accuracy.
- `code/regenerer_graphiques.py` : régénération des figures depuis le CSV.
- `code/experiences_avec_moyenne/moyenne_experiences.py` : lancement répété d'une expérience pour calculer la moyenne.
- `code/distributeur_une_classe_par_client/` : distributeur maison une classe par client.

## Lancer les expériences à partir de fichiers JSON

Les scénarios d'expériences sont décrits dans `code/configurations_experiences/`. Un seul lanceur exécute les expériences sélectionnées.

Lister les scénarios disponibles :

    python code/lancer_experiences.py --list

Lancer un scénario précis :

    python code/lancer_experiences.py --scenario iid_clean_average

Lancer tous les scénarios décrits dans l'index :

    python code/lancer_experiences.py --all

Les résultats sont enregistrés dans `code/resultats_experiences/` avec `resultats_combines.csv`, un dossier `graphiques/` et un dossier `comparaison/`.


Pour régénérer les courbes à partir du CSV déjà généré :

    python code/regenerer_graphiques.py

Pour produire la comparaison globale entre toutes les expériences :

    python code/comparer_experiences.py



Execution GPU :

Pour exécuter des expériences avec accélération GPU, il faut se rendre sur Google Colab et exécuter les commandes figurant dans le fichier texte `commandes_ggl_colab.txt`.

