from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from byzfl import Client, ByzantineClient, Server, DataDistributor
from byzfl.utils.misc import set_random_seed

# SCRIPT_DIR dans step1/code/proportion_byzantins/
SCRIPT_DIR = Path(__file__).resolve().parent

DATA_PATH = SCRIPT_DIR.parents[1] / "data"

def test_tolerance(nb_honest, nb_byz):
    print(f"\n--- Test avec {nb_honest} Honnêtes et {nb_byz} Byzantins ---")
    
    # Données pointant directement vers step1/data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.FashionMNIST(root=str(DATA_PATH), train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=str(DATA_PATH), train=False, download=True, transform=transform)
    
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    distributor = DataDistributor({
        "data_distribution_name": "iid",
        "nb_honest": nb_honest,
        "data_loader": DataLoader(train_ds, batch_size=64, shuffle=True),
        "batch_size": 64,
    })
    
    # Serveur robuste Krum
    server = Server({
        "model_name": "cnn_mnist", 
        "device": "cpu", 
        "test_loader": test_loader,
        "optimizer_name": "SGD", 
        "learning_rate": 0.1,
        "weight_decay": 0.0,
        "milestones": [1000],
        "learning_rate_decay": 0.25,
        "pre_agg_list": [],
        "aggregator_info": {"name": "Krum", "parameters": {"f": nb_byz}}
    })
    
    # Clients honnêtes
    clients = [Client({
        "model_name": "cnn_mnist", 
        "device": "cpu", 
        "optimizer_name": "SGD", 
        "learning_rate": 0.01, 
        "loss_name": "CrossEntropyLoss",
        "weight_decay": 0.0,
        "milestones": [1000],
        "learning_rate_decay": 0.25,
        "LabelFlipping": False,
        "momentum": 0.1,
        "store_per_client_metrics": False,
        "training_dataloader": loader, 
        "nb_labels": 10
    }) for loader in distributor.split_data()]
    
    # Attaquant SignFlipping
    attacker = ByzantineClient({"name": "SignFlipping", "f": nb_byz, "parameters": {}})
    
    # Entraînement 
    for r in range(200):
        state = server.get_dict_parameters()
        for c in clients:
            c.set_model_state(state)
            c.compute_gradients()
            
        honest_grads = [c.get_flat_gradients_with_momentum() for c in clients]
        bad_grads = attacker.apply_attack(honest_grads)
        server.update_model_with_gradients(honest_grads + bad_grads)
        
    print(f"Précision finale : {float(server.compute_test_accuracy()):.4f}")

if __name__ == "__main__":
    set_random_seed(42)
    # 5 Honnêtes, 1 Byz 
    test_tolerance(nb_honest=4, nb_byz=2)
    # 4 Honnêtes, 2 Byz 
    test_tolerance(nb_honest=5, nb_byz=1)
    # 3 Honnêtes, 3 Byz 
    test_tolerance(nb_honest=3, nb_byz=3)