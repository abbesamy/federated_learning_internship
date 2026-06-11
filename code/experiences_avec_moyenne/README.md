# Expérience avec moyenne

Ce dossier contient un lanceur dédié pour exécuter **la même expérience 10 fois** avec des seeds différentes, puis calculer la moyenne de la courbe d'accuracy.

## Utilisation

```bash
python code/experiences_avec_moyenne/moyenne_experiences.py --scenario-file code/configurations_experiences/scenarios/iid_clean_average.json --runs 10
```

### Sorties

- `resultats_bruts.csv` : résultats bruts des 10 exécutions
- `courbe_moyenne.csv` : moyenne et écart-type par round
- `courbe_moyenne.png` : courbe moyenne avec bande d'écart-type

