import os
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_aggregator")

class AnomalyDetectionModel(nn.Module):
    """Simple LSTM-based model for anomaly detection on turbine data"""
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(AnomalyDetectionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out

class ModelAggregator:
    """Class to handle model aggregation for federated learning"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def initialize_model(self, input_size=5):
        """Initialize a new global model"""
        model = AnomalyDetectionModel(input_size=input_size)
        return model
    
    def load_model(self, model_path):
        """Load a model from a file"""
        model = AnomalyDetectionModel()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def aggregate_models(self, model_paths: List[str]):
        """
        Aggregate multiple models using FedAvg algorithm
        
        Args:
            model_paths: List of paths to model files
            
        Returns:
            Aggregated model
        """
        if not model_paths:
            raise ValueError("No models provided for aggregation")
        
        # Load models
        models = [self.load_model(path) for path in model_paths]
        
        # Initialize aggregated model with the same architecture
        aggregated_model = self.initialize_model()
        
        # Get state dictionaries
        state_dicts = [model.state_dict() for model in models]
        
        # Simple averaging of model parameters (FedAvg algorithm)
        # In a real-world scenario, you might want to weight by data size or other factors
        avg_state_dict = {}
        for key in state_dicts[0].keys():
            # Stack parameters along a new dimension
            stacked_params = torch.stack([sd[key] for sd in state_dicts])
            # Average along the new dimension
            avg_state_dict[key] = torch.mean(stacked_params, dim=0)
        
        # Load averaged parameters into the aggregated model
        aggregated_model.load_state_dict(avg_state_dict)
        
        logger.info(f"Successfully aggregated {len(models)} models")
        
        return aggregated_model
    
    def evaluate_aggregated_model(self, model, test_data=None):
        """
        Evaluate the aggregated model on test data
        
        In a real implementation, this would use actual test data.
        For this simulation, we'll return mock metrics.
        """
        # Mock evaluation metrics
        metrics = {
            "accuracy": 92.5,
            "precision": 0.91,
            "recall": 0.89,
            "f1_score": 0.90
        }
        
        return metrics
