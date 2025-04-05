import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from flwr.common import NDArrays, Scalar

from .model import AnomalyDetectionModel

logger = logging.getLogger("fl_client")

class FlowerClient(fl.client.NumPyClient):
    """Flower client implementation for edge nodes."""
    
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader,
        node_id: str,
        device: torch.device = None,
        privacy_setting: Dict = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.node_id = node_id
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.privacy_setting = privacy_setting or {
            "enabled": False,
            "noise_multiplier": 0.1,
            "max_grad_norm": 1.0
        }
        
        # Move model to device
        self.model.to(self.device)
        
        # Training history
        self.training_history = []
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on the local dataset."""
        # Set model parameters
        self.set_parameters(parameters)
        
        # Get training config
        epochs = int(config.get("epochs", 1))
        mu = float(config.get("mu", 0.0))  # Proximal term weight for FedProx
        use_fedprox = bool(config.get("use_fedprox", False))
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        self.model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        num_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs.unsqueeze(1))  # Add sequence dimension
                loss = criterion(outputs, labels)
                
                # Add proximal term if using FedProx
                if use_fedprox and mu > 0:
                    # Get the global model parameters
                    global_params = parameters
                    
                    # Calculate proximal term
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.get_parameters({}), global_params):
                        proximal_term += ((mu / 2) * 
                                         np.sum(np.square(local_param - global_param)))
                    
                    # Add proximal term to loss
                    loss += proximal_term
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping if privacy is enabled
                if self.privacy_setting["enabled"]:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.privacy_setting["max_grad_norm"]
                    )
                
                optimizer.step()
                
                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                epoch_loss += loss.item() * inputs.size(0)
            
            # Calculate epoch statistics
            epoch_loss /= len(self.train_loader.dataset)
            epoch_accuracy = 100.0 * correct / total
            
            logger.info(f"Node {self.node_id}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            
            # Update training metrics
            train_loss = epoch_loss
            train_accuracy = epoch_accuracy
            num_samples = len(self.train_loader.dataset)
        
        # Save training history
        self.training_history.append({
            "loss": train_loss,
            "accuracy": train_accuracy
        })
        
        # Return updated model parameters and metrics
        return self.get_parameters({}), num_samples, {
            "loss": float(train_loss),
            "accuracy": float(train_accuracy)
        }
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on the local test dataset."""
        # Set model parameters
        self.set_parameters(parameters)
        
        # Set up loss function
        criterion = nn.BCEWithLogitsLoss()
        
        # Evaluation loop
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs.unsqueeze(1))
                batch_loss = criterion(outputs, labels)
                
                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loss += batch_loss.item() * inputs.size(0)
        
        # Calculate average loss and accuracy
        avg_loss = loss / len(self.test_loader.dataset)
        accuracy = 100.0 * correct / total
        
        logger.info(f"Node {self.node_id}, Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Return evaluation metrics
        return float(avg_loss), total, {
            "accuracy": float(accuracy)
        }
