import torch
import copy
from gradient_quality_impact import GradientQualityImpactAssessment
from server import FederatedServer  # Your existing class
from clients import FederatedClient  # Existing client class
from models import Net
from utils import load_mnist_data, create_client_data
from torch.utils.data import DataLoader, random_split

# Define ablation configurations (α, β, γ)
weight_configs = [
    (1.0, 0.0, 0.0),  # Only performance impact
    (0.0, 1.0, 0.0),  # Only gradient stability
    (0.0, 0.0, 1.0),  # Only generalization impact
    (0.7, 0.2, 0.1),  # Default config
    (0.5, 0.3, 0.2),
    (0.4, 0.4, 0.2)
]

# Prepare data
train_dataset, test_dataset = load_mnist_data()
val_size = len(test_dataset) // 2
val_dataset, test_dataset = random_split(test_dataset, [val_size, len(test_dataset) - val_size])
test_loader = DataLoader(test_dataset, batch_size=64)
val_loader = DataLoader(val_dataset, batch_size=64)

# Prepare model
global_model = Net(input_dim=28*28, hidden_dim=128, output_dim=10)

# Simulate client updates
num_clients = 4
client_loaders, _, client_data_sizes = create_client_data(train_dataset, num_clients)
client_models = []

for i in range(num_clients):
    client = FederatedClient(
        client_id=i,
        model=copy.deepcopy(global_model),
        data_loader=client_loaders[i],
        epochs=5
    )
    client.train()
    client_models.append(client.model)

# Run ablation
for (alpha, beta, gamma) in weight_configs:
    print(f"\n== Evaluating with α={alpha}, β={beta}, γ={gamma} ==")
    
    gqia = GradientQualityImpactAssessment(
        test_loader=test_loader,
        validation_loader=val_loader,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )
    
    scores = gqia.evaluate_all_contributions(global_model, client_models, client_data_sizes)
    
    for i, score in enumerate(scores):
        print(f"Client {i} score: {score:.2f}")
