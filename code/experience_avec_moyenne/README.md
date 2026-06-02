# Expérience avec moyenne

Ce dossier contient un lanceur dédié pour exécuter **la même expérience 10 fois** avec des seeds différentes, puis calculer la moyenne de la courbe d'accuracy.

## Utilisation

```bash
python code/experience_avec_moyenne/average_experiment.py --scenario-file code/experiment_configs/scenarios/iid_clean_average.json --runs 10
```

### Sorties

- `all_runs.csv` : résultats bruts des 10 exécutions
- `mean_curve.csv` : moyenne et écart-type par round
- `mean_curve.png` : courbe moyenne avec bande d'écart-type

### Options utiles

- `--seed-start 0` : première seed
- `--seed-step 1` : pas entre deux seeds
- `--rounds 1000` : override du nombre de rounds