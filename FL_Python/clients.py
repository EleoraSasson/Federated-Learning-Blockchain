import torch
import torch.nn as nn
import torch.optim as optim
import copy

class FederatedClient:
    def __init__(self, client_id, model, data_loader, test_loader=None, 
                 learning_rate=0.01, epochs=5):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train(self):
        """Train the model on local data and return updated weights"""
        # Set the model to training mode
        self.model.train()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Print progress
            print(f'Client {self.client_id}, Epoch {epoch+1}: Loss = {epoch_loss/len(self.data_loader):.4f}')
        
        # Return the updated model weights
        return self.model.get_weights()
    
    def evaluate(self):
        """Evaluate the model on client-specific test data"""
        if self.test_loader is None:
            print(f"Client {self.client_id} has no test data")
            return None
        
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_loss /= len(self.test_loader)
        print(f'Client {self.client_id} - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return accuracy
    
    def update_model(self, global_model):
        """Update local model with global model"""
        # Copy the parameters from the global model to the local model
        self.model.load_state_dict(global_model.state_dict())