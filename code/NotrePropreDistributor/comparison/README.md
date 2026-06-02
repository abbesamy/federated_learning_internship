# Comparaison des distributeurs

Ce dossier contient un script indépendant pour comparer le distributeur maison avec le distributeur de référence de `byzfl`, sans toucher aux anciens scripts.

## But

Le script affiche la différence de répartition des labels entre :

- `NotrePropreDistributor` : une seule classe par client
- `byzfl.DataDistributor` : répartition de référence, par défaut `iid`

Il exporte aussi :

- un fichier CSV avec les comptes par client et par classe
- une image PNG avec deux cartes de chaleur côte à côte

## Lancer la comparaison

Depuis la racine du projet :

```bash
python code/NotrePropreDistributor/comparison/compare_distributors.py
```

## Options utiles

Tu peux changer quelques paramètres :

```bash
python code/NotrePropreDistributor/comparison/compare_distributors.py \
  --dataset MNIST \
  --num-clients 10 \
  --batch-size 64 \
  --reference-distribution iid \
  --output-dir code/NotrePropreDistributor/comparison/outputs
```

Si tu veux tester le référentiel non-iid de `byzfl` :

```bash
python code/NotrePropreDistributor/comparison/compare_distributors.py \
  --reference-distribution dirichlet_niid \
  --distribution-parameter 0.5
```
