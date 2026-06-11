# Comparaison des distributeurs

Ce dossier contient un script indépendant pour comparer le distributeur maison avec le distributeur de référence de `byzfl`, sans toucher aux anciens scripts.

## But

Le script affiche la différence de répartition des labels entre :

- `distributeur_une_classe_par_client` : une seule classe par client
- `byzfl.DataDistributor` : répartition de référence, par défaut `iid`

Il exporte aussi :

- `comptes_comparaison.csv` : comptes par client et par classe
- `carte_chaleur_comparaison.png` : deux cartes de chaleur côte à côte

## Lancer la comparaison

Depuis la racine du projet :

```bash
python code/distributeur_une_classe_par_client/comparaison/comparer_distributions.py
```

## Options utiles

On peut changer quelques paramètres :

```bash
python code/distributeur_une_classe_par_client/comparaison/comparer_distributions.py \
  --dataset MNIST \
  --num-clients 10 \
  --batch-size 64 \
  --reference-distribution iid \
  --output-dir code/distributeur_une_classe_par_client/comparaison/resultats
```

