import os
import torch
import logging
import json
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from common.fl.client import FlowerClient
from common.fl.model import AnomalyDetectionModel
from training.trainer import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fl_client")

class EdgeNodeFlowerClient:
    """Wrapper for Flower client implementation in edge nodes."""
    
    def __init__(self, node_id: str, aggregator_url: str):
        self.node_id = node_id
        self.aggregator_url = aggregator_url
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.flower_client = None
        self.trainer = ModelTrainer(node_id=node_id)
        
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        
        # Privacy settings
        self.privacy_setting = {
            "enabled": False,
            "noise_multiplier": 0.1,
            "max_grad_norm": 1.0
        }
        
        # Client state
        self.client_running = False
        self.current_round = 0
    
    def _prepare_data(self):
        """Prepare data for federated learning."""
        # Get data loaders from the trainer
        train_loader, test_loader, input_size = self.trainer._prepare_data()
        return train_loader, test_loader, input_size
    
    def initialize_model(self, input_size: int = 5):
        """Initialize the model."""
        self.model = AnomalyDetectionModel(input_size=input_size)
        self.model.to(self.device)
        return self.model
    
    def load_model(self, model_path: str):
        """Load model from file."""
        try:
            # Get input size first
            _, _, input_size = self._prepare_data()
            
            # Initialize model with correct input size
            if self.model is None:
                self.model = self.initialize_model(input_size)
            
            # Load state dict
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def start_client(self, server_address: str):
        """Start Flower client and connect to server."""
        try:
            # Prepare data
            train_loader, test_loader, input_size = self._prepare_data()
            
            # Initialize model if needed
            if self.model is None:
                self.model = self.initialize_model(input_size)
            
            # Create Flower client
            self.flower_client = FlowerClient(
                model=self.model,
                train_loader=train_loader,
                test_loader=test_loader,
                node_id=self.node_id,
                device=self.device,
                privacy_setting=self.privacy_setting
            )
            
            # Start client
            logger.info(f"Starting Flower client, connecting to {server_address}")
            fl.client.start_numpy_client(
                server_address=server_address,
                client=self.flower_client
            )
            
            # Save the final model
            torch.save(self.model.state_dict(), f"models/model_node_{self.node_id}.pt")
            
            logger.info("Flower client finished")
            return True
            
        except Exception as e:
            logger.error(f"Error in Flower client: {str(e)}")
            return False
    
    def update_privacy_settings(self, settings: Dict):
        """Update privacy settings."""
        self.privacy_setting.update(settings)
        logger.info(f"Updated privacy settings: {settings}")
    
    def get_client_status(self):
        """Get client status."""
        return {
            "node_id": self.node_id,
            "model_loaded": self.model is not None,
            "privacy_settings": self.privacy_setting,
            "device": str(self.device)
        }
