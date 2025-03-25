import torch
import copy
import numpy as np
from tqdm import tqdm
import time

class GradientQualityImpactAssessment:
    """
    Gradient Quality Impact Assessment (GQIA) - A novel approach for evaluating
    client contributions in federated learning based on model improvement metrics.
    
    This method evaluates each client's contribution by measuring the positive or negative
    impact their gradient has on multiple dimensions of model performance.
    """
    
    def __init__(self, test_loader, validation_loader=None, metrics=None, alpha=0.7, beta=0.2, gamma=0.1):
        """
        Initialize the GQIA evaluator.
        
        Args:
            test_loader: DataLoader for testing model performance
            validation_loader: Optional separate validation set (if None, uses test_loader)
            metrics: Dictionary of metric functions to evaluate model performance
                    Default metrics are accuracy, loss, and generalization capability
            alpha: Weight for performance improvement (0-1)
            beta: Weight for gradient stability (0-1)
            gamma: Weight for generalization impact (0-1)
        """
        self.test_loader = test_loader
        self.validation_loader = validation_loader if validation_loader else test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default metrics if none provided
        if metrics is None:
            self.metrics = {
                'accuracy': self._calculate_accuracy,
                'loss': self._calculate_loss,
                'generalization': self._calculate_generalization_gap
            }
        else:
            self.metrics = metrics
            
        # Weighting parameters for different aspects of contribution quality
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.alpha = alpha  # Weight for performance improvement
        self.beta = beta    # Weight for gradient stability
        self.gamma = gamma  # Weight for generalization impact
        
        # Cache for model evaluations to avoid redundant calculations
        self.evaluation_cache = {}
        
    def evaluate_contribution(self, global_model, client_model, client_data_size):
        """
        Evaluate the quality of a client's contribution based on model improvement
        and other metrics.
        
        Args:
            global_model: The global model before applying the client's update
            client_model: The client's updated model
            client_data_size: Size of the client's dataset
            
        Returns:
            A score from 0-100 reflecting the quality of the contribution
        """
        # Create a model that includes only this client's gradient
        client_gradient_model = self._apply_single_client_gradient(global_model, client_model)
        
        # Calculate baseline performance metrics
        global_metrics = self._evaluate_model(global_model)
        
        # Calculate performance after applying this client's gradient only
        client_contribution_metrics = self._evaluate_model(client_gradient_model)
        
        # Calculate performance improvement factor
        performance_impact = self._calculate_performance_impact(global_metrics, client_contribution_metrics)
        
        # Calculate gradient stability factor
        gradient_stability = self._calculate_gradient_stability(global_model, client_model, client_data_size)
        
        # Calculate generalization impact
        generalization_impact = self._calculate_generalization_impact(global_model, client_gradient_model)
        
        # Calculate overall contribution score
        contribution_score = (
            self.alpha * performance_impact +
            self.beta * gradient_stability +
            self.gamma * generalization_impact
        ) * 100  # Scale to 0-100 range
        
        # Ensure score is in valid range
        contribution_score = max(0, min(100, contribution_score))
        
        return contribution_score
    
    def evaluate_all_contributions(self, global_model, client_models, client_data_sizes):
        """
        Evaluate contributions from all clients.
        
        Args:
            global_model: The global model before aggregation
            client_models: List of client models after local training
            client_data_sizes: List of client dataset sizes
            
        Returns:
            List of contribution scores
        """
        scores = []
        for client_model, data_size in zip(client_models, client_data_sizes):
            score = self.evaluate_contribution(global_model, client_model, data_size)
            scores.append(score)
        
        return scores
    
    def _apply_single_client_gradient(self, global_model, client_model):
        """
        Create a model that applies only the gradient from a single client.
        
        Formula: new_model = global_model + (client_model - global_model)
        
        Args:
            global_model: Global model before update
            client_model: Client's updated model
            
        Returns:
            A new model with only this client's gradient applied
        """
        # Create a deep copy of the global model
        new_model = copy.deepcopy(global_model)
        
        # Apply client's gradient (model difference)
        with torch.no_grad():
            for (global_param, client_param, new_param) in zip(
                global_model.parameters(), client_model.parameters(), new_model.parameters()
            ):
                # Calculate client's gradient
                gradient = client_param.data - global_param.data
                # Apply gradient to global model
                new_param.data = global_param.data + gradient
        
        return new_model
    
    def _calculate_performance_impact(self, base_metrics, contribution_metrics):
        """
        Calculate how much this client's gradient improves model performance.
        
        Args:
            base_metrics: Metrics of global model before update
            contribution_metrics: Metrics after applying this client's gradient
            
        Returns:
            Score between 0-1 representing performance improvement
        """
        # Calculate weighted improvement across all metrics
        improvements = {}
        
        # For accuracy, higher is better
        if 'accuracy' in base_metrics:
            acc_base = base_metrics['accuracy']
            acc_new = contribution_metrics['accuracy']
            improvements['accuracy'] = (acc_new - acc_base) / (1 - acc_base + 1e-10)  # Normalized improvement
            
        # For loss, lower is better
        if 'loss' in base_metrics:
            loss_base = base_metrics['loss']
            loss_new = contribution_metrics['loss']
            improvements['loss'] = (loss_base - loss_new) / (loss_base + 1e-10)  # Normalized improvement
        
        # For generalization gap, lower is better
        if 'generalization' in base_metrics:
            gen_base = base_metrics['generalization']
            gen_new = contribution_metrics['generalization']
            improvements['generalization'] = (gen_base - gen_new) / (gen_base + 1e-10)
            
        # Calculate weighted average of improvements
        if not improvements:
            return 0
            
        avg_improvement = sum(improvements.values()) / len(improvements)
        
        # Apply sigmoid to map to [0, 1] range with a reasonable scaling
        return 1 / (1 + np.exp(-5 * avg_improvement))
    
    def _calculate_gradient_stability(self, global_model, client_model, data_size):
        """
        Calculate the stability of the client's gradient based on magnitude and direction.
        
        Args:
            global_model: Global model before update
            client_model: Client's updated model
            data_size: Size of client's dataset
            
        Returns:
            Score between 0-1 representing gradient stability
        """
        # Calculate L2 norm of gradient
        grad_norm = 0
        param_norm = 0
        
        for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
            grad = client_param.data - global_param.data
            grad_norm += torch.norm(grad, p=2).item() ** 2
            param_norm += torch.norm(global_param.data, p=2).item() ** 2
        
        grad_norm = np.sqrt(grad_norm)
        param_norm = np.sqrt(param_norm)
        
        # Normalize by data size to account for different client dataset sizes
        relative_grad_norm = grad_norm / (np.sqrt(data_size) + 1e-10)
        
        # Calculate gradient to parameter ratio
        grad_param_ratio = grad_norm / (param_norm + 1e-10)
        
        # Ideal ratio should be small but non-zero
        # Using exponential decay function to penalize both very large and very small gradients
        stability = np.exp(-5 * abs(grad_param_ratio - 0.01))
        
        return stability
    
    def _calculate_generalization_impact(self, global_model, client_gradient_model):
        """
        Calculate how the client's gradient affects model generalization.
        
        Args:
            global_model: Global model before update
            client_gradient_model: Model after applying the client's gradient
            
        Returns:
            Score between 0-1 representing generalization impact
        """
        # Calculate generalization gap before and after applying client gradient
        before_gap = self._calculate_generalization_gap_for_model(global_model)
        after_gap = self._calculate_generalization_gap_for_model(client_gradient_model)
        
        # Improvement in generalization (reduction in gap)
        gap_improvement = (before_gap - after_gap) / (before_gap + 1e-10)
        
        # Apply sigmoid to map to [0, 1] range
        return 1 / (1 + np.exp(-5 * gap_improvement))
    
    def _evaluate_model(self, model):
        """
        Evaluate a model using all defined metrics.
        
        Args:
            model: The PyTorch model to evaluate
            
        Returns:
            Dictionary of metric results
        """
        # Use cached results if available
        model_id = id(model)
        if model_id in self.evaluation_cache:
            return self.evaluation_cache[model_id]
        
        # Move model to the appropriate device
        model.to(self.device)
        model.eval()
        
        # Calculate all metrics
        results = {}
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(model)
        
        # Cache the results
        self.evaluation_cache[model_id] = results
        
        return results
    
    def _calculate_accuracy(self, model):
        """Calculate accuracy on the test set"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total
    
    def _calculate_loss(self, model):
        """Calculate loss on the test set"""
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        return total_loss / total_samples
    
    def _calculate_generalization_gap(self, model):
        """Calculate generalization gap (difference between train and validation performance)"""
        # Use this metric if we have separate validation data
        if self.validation_loader == self.test_loader:
            # If no separate validation set, use a proxy metric
            return 0.0
            
        return self._calculate_generalization_gap_for_model(model)
    
    def _calculate_generalization_gap_for_model(self, model):
        """Calculate generalization gap for a specific model"""
        # Calculate accuracy on test set
        test_accuracy = self._calculate_accuracy(model)
        
        # Calculate accuracy on validation set (or a subset of test set)
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in self.validation_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_accuracy = val_correct / val_total
        
        # Calculate absolute difference
        return abs(test_accuracy - val_accuracy)
    
    def analyze_contributions(self, global_model, client_models, client_data_sizes, client_ids=None):
        """
        Provide a detailed analysis of each client's contribution.
        
        Args:
            global_model: The global model before aggregation
            client_models: List of client models after local training
            client_data_sizes: List of client dataset sizes
            client_ids: List of client identifiers (optional)
            
        Returns:
            Dictionary with detailed analysis
        """
        if client_ids is None:
            client_ids = [f"Client {i}" for i in range(len(client_models))]
            
        # Get baseline metrics
        global_metrics = self._evaluate_model(global_model)
        
        # Evaluate each client's contribution
        detailed_analysis = {
            'global_metrics': global_metrics,
            'clients': []
        }
        
        for i, (client_model, data_size, client_id) in enumerate(zip(client_models, client_data_sizes, client_ids)):
            # Create model with only this client's gradient
            client_gradient_model = self._apply_single_client_gradient(global_model, client_model)
            
            # Get metrics after applying this client's gradient
            contribution_metrics = self._evaluate_model(client_gradient_model)
            
            # Calculate performance impact
            performance_impact = self._calculate_performance_impact(global_metrics, contribution_metrics)
            
            # Calculate gradient stability
            gradient_stability = self._calculate_gradient_stability(global_model, client_model, data_size)
            
            # Calculate generalization impact
            generalization_impact = self._calculate_generalization_impact(global_model, client_gradient_model)
            
            # Calculate final score
            final_score = (
                self.alpha * performance_impact +
                self.beta * gradient_stability +
                self.gamma * generalization_impact
            ) * 100
            
            # Add to analysis
            detailed_analysis['clients'].append({
                'client_id': client_id,
                'data_size': data_size,
                'contribution_metrics': contribution_metrics,
                'performance_impact': performance_impact,
                'gradient_stability': gradient_stability,
                'generalization_impact': generalization_impact,
                'final_score': final_score
            })
        
        return detailed_analysis

    def clear_cache(self):
        """Clear the evaluation cache"""
        self.evaluation_cache = {}


# Example usage
if __name__ == "__main__":
    # This is a demonstration of how to use the GQIA
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split
    
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = nn.functional.log_softmax(x, dim=1)
            return output
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Split into validation and test sets
    test_size = len(test_dataset) // 2
    val_size = len(test_dataset) - test_size
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Initialize GQIA
    gqia = GradientQualityImpactAssessment(test_loader, val_loader)
    
    # Create a global model
    global_model = SimpleModel()
    
    # Simulate client models (here we just create random updates for demonstration)
    client_models = []
    client_data_sizes = [1000, 2000, 3000, 4000]
    
    for i in range(4):
        client_model = copy.deepcopy(global_model)
        
        # Create random updates of different qualities
        with torch.no_grad():
            for param in client_model.parameters():
                # Add random noise to parameters (simulating training)
                noise = torch.randn_like(param) * (0.01 / (i+1))  # Different noise levels
                param.data += noise
        
        client_models.append(client_model)
    
    # Evaluate contributions
    contribution_scores = gqia.evaluate_all_contributions(global_model, client_models, client_data_sizes)
    
    print("Contribution Scores:")
    for i, score in enumerate(contribution_scores):
        print(f"Client {i+1}: {score:.2f}")
    
    # Get detailed analysis
    detailed_analysis = gqia.analyze_contributions(global_model, client_models, client_data_sizes)
    
    print("\nDetailed Analysis:")
    for client in detailed_analysis['clients']:
        print(f"\nClient {client['client_id']}:")
        print(f"  Data Size: {client['data_size']}")
        print(f"  Performance Impact: {client['performance_impact']:.4f}")
        print(f"  Gradient Stability: {client['gradient_stability']:.4f}")
        print(f"  Generalization Impact: {client['generalization_impact']:.4f}")
        print(f"  Final Score: {client['final_score']:.2f}")