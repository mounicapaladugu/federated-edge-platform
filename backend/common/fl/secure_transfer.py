import os
import json
import shutil
import logging
import time
from typing import Dict, Optional, List
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("secure_transfer")

class SecureModelTransfer:
    """
    Handles secure model transfer between aggregator and edge nodes
    in air-gapped or restricted connectivity environments.
    """
    
    def __init__(self, 
                 node_id: Optional[str] = None, 
                 is_aggregator: bool = False,
                 mount_dir: str = "/mnt/secure_transfer"):
        """
        Initialize the secure transfer module.
        
        Args:
            node_id: ID of the edge node (None if aggregator)
            is_aggregator: Whether this instance is running on the aggregator
            mount_dir: Directory simulating a mounted USB drive or secure file share
        """
        self.node_id = node_id
        self.is_aggregator = is_aggregator
        self.mount_dir = mount_dir
        
        # Create necessary directories
        os.makedirs(mount_dir, exist_ok=True)
        
        if is_aggregator:
            # Create directories for each node
            self.models_dir = os.path.join(mount_dir, "models")
            self.updates_dir = os.path.join(mount_dir, "updates")
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.updates_dir, exist_ok=True)
            
            # Create node-specific directories
            for i in range(1, 10):  # Support up to 9 nodes
                node_model_dir = os.path.join(self.models_dir, f"node_{i}")
                node_update_dir = os.path.join(self.updates_dir, f"node_{i}")
                os.makedirs(node_model_dir, exist_ok=True)
                os.makedirs(node_update_dir, exist_ok=True)
        else:
            # Edge node directories
            self.node_models_dir = os.path.join(mount_dir, "models", f"node_{node_id}")
            self.node_updates_dir = os.path.join(mount_dir, "updates", f"node_{node_id}")
            os.makedirs(self.node_models_dir, exist_ok=True)
            os.makedirs(self.node_updates_dir, exist_ok=True)
    
    def export_global_model(self, model_path: str, metadata: Dict) -> bool:
        """
        Export the global model to the secure transfer directory for all nodes.
        
        Args:
            model_path: Path to the global model file
            metadata: Model metadata
            
        Returns:
            bool: Success status
        """
        if not self.is_aggregator:
            logger.error("Only the aggregator can export global models")
            return False
        
        try:
            # Save metadata
            metadata_path = os.path.join(self.models_dir, "global_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            
            # Copy model to each node's directory
            for i in range(1, 10):  # Support up to 9 nodes
                node_dir = os.path.join(self.models_dir, f"node_{i}")
                if os.path.exists(node_dir):
                    dest_model_path = os.path.join(node_dir, "global_model.pt")
                    shutil.copy(model_path, dest_model_path)
                    
                    # Copy metadata
                    dest_metadata_path = os.path.join(node_dir, "global_metadata.json")
                    shutil.copy(metadata_path, dest_metadata_path)
                    
                    logger.info(f"Exported global model to node_{i}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting global model: {str(e)}")
            return False
    
    def import_global_model(self) -> Optional[Dict]:
        """
        Import the global model from the secure transfer directory.
        
        Returns:
            Dict: Metadata of the imported model or None if no model found
        """
        if self.is_aggregator:
            logger.error("This method is only for edge nodes")
            return None
        
        try:
            # Check if model exists
            model_path = os.path.join(self.node_models_dir, "global_model.pt")
            metadata_path = os.path.join(self.node_models_dir, "global_metadata.json")
            
            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                logger.info("No new global model found")
                return None
            
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Get local model path
            local_model_dir = "models"
            os.makedirs(local_model_dir, exist_ok=True)
            local_model_path = os.path.join(local_model_dir, f"model_node_{self.node_id}.pt")
            
            # Copy model
            shutil.copy(model_path, local_model_path)
            
            # Save metadata
            local_metadata_path = os.path.join(local_model_dir, f"metadata_node_{self.node_id}.json")
            with open(local_metadata_path, "w") as f:
                json.dump(metadata, f)
            
            logger.info(f"Imported global model version {metadata.get('version', 'unknown')}")
            
            # Optionally, remove the model from the transfer directory to avoid reimporting
            # Uncomment the following lines if you want this behavior
            # os.remove(model_path)
            # os.remove(metadata_path)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error importing global model: {str(e)}")
            return None
    
    def export_model_update(self, model_path: str, metrics: Dict) -> bool:
        """
        Export a model update to the secure transfer directory.
        
        Args:
            model_path: Path to the model update file
            metrics: Training metrics
            
        Returns:
            bool: Success status
        """
        if self.is_aggregator:
            logger.error("This method is only for edge nodes")
            return False
        
        try:
            # Create update metadata
            update_metadata = {
                "node_id": self.node_id,
                "timestamp": time.time(),
                "metrics": metrics
            }
            
            # Save metadata
            metadata_path = os.path.join(self.node_updates_dir, "update_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(update_metadata, f)
            
            # Copy model update
            dest_model_path = os.path.join(self.node_updates_dir, "model_update.pt")
            shutil.copy(model_path, dest_model_path)
            
            logger.info(f"Exported model update from node {self.node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model update: {str(e)}")
            return False
    
    def import_model_updates(self) -> List[Dict]:
        """
        Import model updates from all nodes.
        
        Returns:
            List[Dict]: List of update metadata and paths
        """
        if not self.is_aggregator:
            logger.error("Only the aggregator can import model updates")
            return []
        
        updates = []
        
        try:
            # Check each node's update directory
            for i in range(1, 10):  # Support up to 9 nodes
                node_update_dir = os.path.join(self.updates_dir, f"node_{i}")
                if not os.path.exists(node_update_dir):
                    continue
                
                model_path = os.path.join(node_update_dir, "model_update.pt")
                metadata_path = os.path.join(node_update_dir, "update_metadata.json")
                
                if os.path.exists(model_path) and os.path.exists(metadata_path):
                    # Load metadata
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    # Get local paths
                    local_updates_dir = "updates"
                    os.makedirs(local_updates_dir, exist_ok=True)
                    local_model_path = os.path.join(local_updates_dir, f"update_node_{i}.pt")
                    
                    # Copy model
                    shutil.copy(model_path, local_model_path)
                    
                    updates.append({
                        "node_id": str(i),
                        "model_path": local_model_path,
                        "metadata": metadata
                    })
                    
                    logger.info(f"Imported model update from node {i}")
                    
                    # Optionally, remove the update from the transfer directory
                    # Uncomment the following lines if you want this behavior
                    # os.remove(model_path)
                    # os.remove(metadata_path)
            
            return updates
            
        except Exception as e:
            logger.error(f"Error importing model updates: {str(e)}")
            return []
    
    def check_for_new_model(self) -> bool:
        """
        Check if a new global model is available.
        
        Returns:
            bool: True if a new model is available
        """
        if self.is_aggregator:
            logger.error("This method is only for edge nodes")
            return False
        
        model_path = os.path.join(self.node_models_dir, "global_model.pt")
        metadata_path = os.path.join(self.node_models_dir, "global_metadata.json")
        
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            return False
        
        # Check if the model is newer than our current one
        try:
            with open(metadata_path, "r") as f:
                new_metadata = json.load(f)
            
            local_metadata_path = os.path.join("models", f"metadata_node_{self.node_id}.json")
            
            if not os.path.exists(local_metadata_path):
                return True  # No local model, so new model is available
            
            with open(local_metadata_path, "r") as f:
                current_metadata = json.load(f)
            
            # Compare versions (assuming version is a string like "round_X")
            new_version = new_metadata.get("version", "")
            current_version = current_metadata.get("version", "")
            
            return new_version != current_version
            
        except Exception as e:
            logger.error(f"Error checking for new model: {str(e)}")
            return False
