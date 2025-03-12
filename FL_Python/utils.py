import torch
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def load_mnist_data():
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    return train_dataset, test_dataset

def create_client_data(dataset, num_clients, iid=True, test_split=0.2):
    """
    Split dataset among clients with train/test splits for each client
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        iid: If True, data is split iid, otherwise it's split non-iid
        test_split: Fraction of client data to use for testing
        
    Returns:
        client_train_loaders: List of training DataLoader for each client
        client_test_loaders: List of testing DataLoader for each client
        client_samples: List of number of samples per client
    """
    client_train_loaders = []
    client_test_loaders = []
    client_samples = []
    
    if iid:
        # IID split: each client gets random samples
        samples_per_client = len(dataset) // num_clients
        indices = torch.randperm(len(dataset))
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(dataset)
            client_indices = indices[start_idx:end_idx]
            
            # Split into train and test
            num_test = int(len(client_indices) * test_split)
            test_indices = client_indices[:num_test]
            train_indices = client_indices[num_test:]
            
            # Create datasets and loaders
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            client_train_loaders.append(train_loader)
            client_test_loaders.append(test_loader)
            client_samples.append(len(train_dataset))  # Only count training samples
            
    else:
        # Non-IID split: each client gets samples biased towards certain classes
        # Get targets for all samples
        if isinstance(dataset.targets, torch.Tensor):
            targets = dataset.targets.numpy()
        else:
            targets = np.array(dataset.targets)
        
        # Sort samples by target
        sorted_indices = np.argsort(targets)
        
        # Each client gets 2 primary classes with higher probability
        samples_per_client = len(dataset) // num_clients
        primary_classes_per_client = 2
        num_classes = 10  # For MNIST
        
        indices_by_class = [[] for _ in range(num_classes)]
        for idx in sorted_indices:
            target = targets[idx]
            indices_by_class[target].append(idx)
        
        for i in range(num_clients):
            # Assign primary classes to this client (with overlap between clients)
            client_indices = []
            primary_classes = [(i*primary_classes_per_client) % num_classes, 
                              (i*primary_classes_per_client + 1) % num_classes]
            
            # 80% data from primary classes, 20% from other classes
            primary_indices = []
            for c in primary_classes:
                start_idx = (i * len(indices_by_class[c])) // num_clients
                end_idx = ((i + 1) * len(indices_by_class[c])) // num_clients
                primary_indices.extend(indices_by_class[c][start_idx:end_idx])
            
            # Randomly sample from remaining classes
            remaining_indices = [idx for c in range(num_classes) if c not in primary_classes
                                for idx in indices_by_class[c]]
            remaining_indices = np.random.choice(
                remaining_indices, 
                size=max(0, samples_per_client - len(primary_indices)),
                replace=False
            )
            
            client_indices = primary_indices + remaining_indices.tolist()
            
            # Split into train and test
            num_test = int(len(client_indices) * test_split)
            # Shuffle indices before splitting to ensure class balance in test set
            np.random.shuffle(client_indices)
            test_indices = client_indices[:num_test]
            train_indices = client_indices[num_test:]
            
            # Create datasets and loaders
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            client_train_loaders.append(train_loader)
            client_test_loaders.append(test_loader)
            client_samples.append(len(train_dataset))  # Only count training samples
    
    return client_train_loaders, client_test_loaders, client_samples

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot global accuracy
    ax1.plot(history['global_accuracy'])
    ax1.set_title('Global Model Accuracy')
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Accuracy (%)')
    
    # Plot client accuracies
    for client_id, client_acc in history['client_accuracy'].items():
        ax2.plot(client_acc, label=f'Client {client_id}')
    ax2.set_title('Client Model Accuracy')
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('federated_learning_results.png')
    plt.show()