# Notre Propre Distributor

Ce dossier contient un distributeur maison qui force une répartition **non-iid extrême** : chaque client reçoit uniquement les exemples d'une seule classe.

## Idée

Si on prend `10` clients et `10` classes, on peut faire :

- client 0 -> classe 0
- client 1 -> classe 1
- client 2 -> classe 2
- ...
- client 9 -> classe 9

Chaque client apprend donc sur une seule classe.

## Fichiers

- `notre_propre_distributor.py` : classe principale `SingleClassDataDistributor`
- `demo.py` : petit script qui affiche la classe attribuée à chaque client

## Utilisation

Depuis la racine du projet :

```bash
python code/NotrePropreDistributor/demo.py
```

Si tu veux réutiliser le distributeur dans un autre script :

```python
from code.NotrePropreDistributor.notre_propre_distributor import build_mnist_one_class_loaders

client_loaders, test_loader, distributor = build_mnist_one_class_loaders(
    data_root="data/MNIST",
    dataset_name="MNIST",
    num_clients=10,
    batch_size=64,
)
```

## Ce que fait le code

- lit les labels du dataset (`targets` ou `labels`)
- regroupe les indices par classe
- attribue une classe unique à chaque client
- crée un `DataLoader` par client

## Remarque

Le comportement par défaut suppose `10` clients et `10` classes. Si tu veux moins ou plus de clients, il faudra décider comment réutiliser les classes.
