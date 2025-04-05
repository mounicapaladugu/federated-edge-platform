import os
import flwr as fl
import torch
import logging
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import threading
import time

from common.fl.model import AnomalyDetectionModel
from common.fl.strategy import FedAvgWithModelVersioning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fl_server")

class FlowerServer:
    """Flower server implementation for federated learning orchestration."""
    
    def __init__(self, node_count: int = 3):
        self.node_count = node_count
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AnomalyDetectionModel()
        self.model.to(self.device)
        
        # Create directories for model storage
        os.makedirs("models", exist_ok=True)
        os.makedirs("updates", exist_ok=True)
        
        # Server state
        self.server_running = False
        self.current_round = 0
        self.server_thread = None
        self.strategy = None
        
        # FL configuration
        self.config = {
            "use_fedprox": False,  # Whether to use FedProx instead of FedAvg
            "mu": 0.01,            # Proximal term weight for FedProx
            "min_fit_clients": max(2, node_count - 1),  # Minimum number of clients for training
            "min_evaluate_clients": max(2, node_count - 1),  # Minimum number of clients for evaluation
            "min_available_clients": node_count,  # Minimum number of available clients
            "privacy": {
                "enabled": False,  # Whether to enable differential privacy
                "noise_multiplier": 0.1,  # Noise multiplier for DP
                "max_grad_norm": 1.0  # Max gradient norm for clipping
            }
        }
    
    def initialize_model(self) -> None:
        """Initialize a new global model."""
        logger.info("Initializing new global model")
        self.model = AnomalyDetectionModel()
        self.model.to(self.device)
        
        # Save the initial model
        torch.save(self.model.state_dict(), "models/global_model.pt")
        
        # Save metadata
        metadata = {
            "version": 0,
            "timestamp": time.time(),
            "round": 0
        }
        
        with open("models/global_metadata.json", "w") as f:
            json.dump(metadata, f)
    
    def load_model(self, version: Optional[int] = None) -> bool:
        """Load a model from storage, optionally specifying a version."""
        try:
            if version is not None:
                # Load a specific version from the strategy's history
                if self.strategy:
                    model_version = self.strategy.get_model_version(version)
                    if model_version:
                        # Convert numpy arrays to torch tensors
                        state_dict = {
                            key: torch.tensor(value)
                            for key, value in zip(self.model.state_dict().keys(), model_version["weights"])
                        }
                        self.model.load_state_dict(state_dict)
                        logger.info(f"Loaded model version {version}")
                        return True
                    else:
                        logger.error(f"Model version {version} not found")
                        return False
                else:
                    logger.error("Strategy not initialized, cannot load model version")
                    return False
            else:
                # Load the latest model
                model_path = "models/global_model.pt"
                if os.path.exists(model_path):
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    logger.info("Loaded latest global model")
                    return True
                else:
                    logger.error("No global model found")
                    return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def save_model(self, state_dict: Dict, version: int, round_num: int) -> None:
        """Save the model to storage."""
        try:
            # Save model state
            torch.save(state_dict, "models/global_model.pt")
            
            # Save a versioned copy
            os.makedirs("models/versions", exist_ok=True)
            torch.save(state_dict, f"models/versions/global_model_v{version}.pt")
            
            # Save metadata
            metadata = {
                "version": version,
                "timestamp": time.time(),
                "round": round_num
            }
            
            with open("models/global_metadata.json", "w") as f:
                json.dump(metadata, f)
            
            with open(f"models/versions/global_metadata_v{version}.json", "w") as f:
                json.dump(metadata, f)
            
            logger.info(f"Saved model version {version} from round {round_num}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def start_server(self, config: Optional[Dict] = None) -> None:
        """Start the Flower server in a background thread."""
        if self.server_running:
            logger.warning("Server is already running")
            return
        
        # Update config if provided
        if config:
            self.config.update(config)
        
        # Initialize the server in a background thread
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.server_running = True
        logger.info("Flower server started in background thread")
    
    def _run_server(self) -> None:
        """Run the Flower server."""
        try:
            # Initialize model if it doesn't exist
            model_path = "models/global_model.pt"
            if not os.path.exists(model_path):
                self.initialize_model()
            else:
                self.load_model()
            
            # Define strategy
            self.strategy = FedAvgWithModelVersioning(
                fraction_fit=1.0,  # Sample 100% of available clients for training
                fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
                min_fit_clients=self.config["min_fit_clients"],
                min_evaluate_clients=self.config["min_evaluate_clients"],
                min_available_clients=self.config["min_available_clients"],
                on_fit_config_fn=self._fit_config,
                on_evaluate_config_fn=self._evaluate_config,
                initial_parameters=fl.common.ndarrays_to_parameters(
                    [val.cpu().numpy() for _, val in self.model.state_dict().items()]
                ),
                # FedProx parameters
                mu=self.config["mu"],
                use_fedprox=self.config["use_fedprox"],
                # Privacy parameters
                privacy_setting=self.config["privacy"]
            )
            
            # Start Flower server
            history = fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=fl.server.ServerConfig(num_rounds=1),  # We'll control rounds manually
                strategy=self.strategy
            )
            
            logger.info("Flower server finished")
            self.server_running = False
            
        except Exception as e:
            logger.error(f"Error in Flower server: {str(e)}")
            self.server_running = False
    
    def _fit_config(self, server_round: int) -> Dict:
        """Return training configuration for clients."""
        self.current_round = server_round
        config = {
            "epochs": 1,
            "batch_size": 32,
            "round": server_round,
            "mu": self.config["mu"] if self.config["use_fedprox"] else 0.0,
            "use_fedprox": self.config["use_fedprox"]
        }
        return config
    
    def _evaluate_config(self, server_round: int) -> Dict:
        """Return evaluation configuration for clients."""
        return {
            "round": server_round
        }
    
    def get_server_status(self) -> Dict:
        """Get the current status of the server."""
        return {
            "running": self.server_running,
            "current_round": self.current_round,
            "node_count": self.node_count,
            "config": self.config
        }
    
    def update_config(self, config: Dict) -> None:
        """Update the server configuration."""
        self.config.update(config)
        logger.info(f"Updated server configuration: {config}")
    
    def rollback_model(self, version: int) -> bool:
        """Roll back to a specific model version."""
        if not self.strategy:
            logger.error("Strategy not initialized, cannot rollback")
            return False
        
        success = self.strategy.rollback_to_version(version)
        if success:
            # Also load the model
            return self.load_model(version)
        return False
