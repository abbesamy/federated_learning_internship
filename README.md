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
    
    uv pip sync uv.lock

Le fichier uv.lock a été généré grace à cette commande :
    
    uv pip compile requirements.txt -o uv.lock

## Organisation des scripts

Les scripts exécutables sont placés une seule fois dans `code/` avec des noms français explicites.

Structure principale :

- `code/moteur_experiences.py` : moteur d'exécution des expériences.
- `code/lancer_experiences.py` : lancement des scénarios JSON.
- `code/comparer_experiences.py` : comparaison globale des courbes d'accuracy.
- `code/regenerer_graphiques.py` : régénération des figures depuis le CSV.
- `code/comparer_attaques_agregateurs.py` : comparaison des agrégateurs pour chaque attaque.
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

Pour comparer les agrégateurs attaque par attaque à partir de tous les CSV de résultats disponibles :

    python code/comparer_attaques_agregateurs.py



Execution GPU :

Pour exécuter des expériences avec accélération GPU, il faut se rendre sur Google Colab et exécuter les commandes figurant dans le fichier texte `commandes_ggl_colab.txt`.





















maintenant on va passer au bilan de la semaine du 01/06 on l'on fait tout cela :

-ajouté de nouveaux scénarios à ceux deja présent dans notre dépot.

-Nous avons implémenté les expériences moyennant sur 10 seeds dont le code se trouve dans le répertoire expérience avec moyennes, et dont l'utilisation est expliqué dans le readme dédié dans ce même répertoire.

-Nous avons aussi corrigé certaines erreurs que nous avons constaté lors de la dernière réunion concernant la génération de graphiques, et nous avons aussi fait en sorte qu'il soient généré dans les répertoires dédié afin d'organiser au mieux notre dépot.

-Nous avons executé des expériences sur 20 rounds avant de pouvoir faire des expériences sur colab dont les résultats se sont montré infructueux, la'accuracy dépassait à peine Les 50 %.

-Une fois le protocole d'execution sur google colab mis en place, nous avons pu lancer de plus grandes expériences sur google colab sur 500 rounds par scenarion dont les résultats etait trèes interessants, on atteignait les 0.96 et 0.97 d'accuracy. Tu peux chiffrer grace au graphique que je te donne. Tu peux aussi lfaire en sorte de mettre les graphiques pour illustrer dans le rapports notre travail, j'aurais juste à importer les graphique et à mettre le nom du fichier png dans le rapports latex, en s'en fout ici de la taille du rapport, il doit etre plus grand, on est censé avoir bien avancé. les graphique que je te donne la concerne ipm avec les différents agrégateurs.

-Nous avons aussi expérimenté av

- maintenant on va passer au bilan de la semaine du 01/06 on l'on fait tout cela :

-ajouté de nouveaux scénarios à ceux deja présent dans notre dépot.

-Nous avons implémenté les expériences moyennant sur 10 seeds dont le code se trouve dans le répertoire expérience avec moyennes, et dont l'utilisation est expliqué dans le readme dédié dans ce même répertoire.

-Nous avons aussi corrigé certaines erreurs que nous avons constaté lors de la dernière réunion concernant la génération de graphiques, et nous avons aussi fait en sorte qu'il soient généré dans les répertoires dédié afin d'organiser au mieux notre dépot.

-Nous avons executé des expériences sur 20 rounds avant de pouvoir faire des expériences sur colab dont les résultats se sont montré infructueux, la'accuracy dépassait à peine Les 50 %.

-Une fois le protocole d'execution sur google colab mis en place, nous avons pu lancer de plus grandes expériences sur google colab sur 500 rounds par scenarion dont les résultats etait trèes interessants, on atteignait les 0.96 et 0.97 d'accuracy. Tu peux chiffrer grace au graphique que je te donne. Tu peux aussi lfaire en sorte de mettre les graphiques pour illustrer dans le rapports notre travail, j'aurais juste à importer les graphique et à mettre le nom du fichier png dans le rapports latex, en s'en fout ici de la taille du rapport, il doit etre plus grand, on est censé avoir bien avancé. les graphique que je te donne la concerne ipm avec les différents agrégateurs.

-Nous avons aussi expérimenté av

- 