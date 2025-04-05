from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
import os
import torch
import json
import logging
import requests
from typing import Dict, Any
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edge_node_api")

# Initialize router
router = APIRouter(prefix="/api", tags=["edge_node"])

# Get environment variables
NODE_ID = os.getenv("NODE_ID", "1")
AGGREGATOR_URL = os.getenv("AGGREGATOR_URL", "http://localhost:8000")

# Initialize model trainer
model_trainer = ModelTrainer(node_id=NODE_ID)

# Training metrics
training_metrics = {
    "current_round": 0,
    "accuracy": 0.0,
    "loss": 0.0,
    "training_complete": False
}

@router.get("/status")
async def get_status():
    """Get the status of the edge node"""
    model_path = os.path.join("models", f"model_node_{NODE_ID}.pt")
    model_exists = os.path.exists(model_path)
    
    return {
        "node_id": NODE_ID,
        "status": "running",
        "model_loaded": model_exists,
        "training_metrics": training_metrics
    }

@router.post("/receive_model")
async def receive_model(background_tasks: BackgroundTasks, model_file: UploadFile = File(...), metadata: UploadFile = File(None)):
    """
    Receive a model from the central hub/aggregator
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save the model file
        model_path = os.path.join("models", f"model_node_{NODE_ID}.pt")
        with open(model_path, "wb") as f:
            f.write(await model_file.read())
        
        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join("models", f"metadata_node_{NODE_ID}.json")
            with open(metadata_path, "wb") as f:
                f.write(await metadata.read())
            
            # Load metadata
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
                
            logger.info(f"Received model version: {metadata_dict.get('version', 'unknown')}")
        
        # Schedule training in the background
        background_tasks.add_task(train_model_task)
        
        return {"message": "Model received successfully, training scheduled"}
    
    except Exception as e:
        logger.error(f"Error receiving model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error receiving model: {str(e)}")

@router.get("/training_status")
async def get_training_status():
    """Get the current training status"""
    return training_metrics

def train_model_task():
    """Background task to train the model"""
    global training_metrics
    
    try:
        logger.info(f"Starting model training on node {NODE_ID}")
        training_metrics["current_round"] += 1
        training_metrics["training_complete"] = False
        
        # Load the model
        model_path = os.path.join("models", f"model_node_{NODE_ID}.pt")
        
        # Train the model
        results = model_trainer.train(model_path)
        
        # Update metrics
        training_metrics.update(results)
        training_metrics["training_complete"] = True
        
        # Send model updates back to aggregator
        send_model_updates()
        
        logger.info(f"Training completed on node {NODE_ID}")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        training_metrics["error"] = str(e)

def send_model_updates():
    """Send model updates back to the aggregator"""
    try:
        # Get the updated model
        model_path = os.path.join("models", f"model_node_{NODE_ID}.pt")
        
        if not os.path.exists(model_path):
            logger.error("Model file not found")
            return
        
        # Prepare files for upload
        files = {
            'model_file': ('model.pt', open(model_path, 'rb'), 'application/octet-stream'),
            'metrics': ('metrics.json', json.dumps(training_metrics), 'application/json')
        }
        
        # Send to aggregator
        response = requests.post(
            f"{AGGREGATOR_URL}/receive_update",
            files=files,
            data={"node_id": NODE_ID}
        )
        
        if response.status_code == 200:
            logger.info("Model updates sent successfully to aggregator")
        else:
            logger.error(f"Failed to send model updates: {response.text}")
    
    except Exception as e:
        logger.error(f"Error sending model updates: {str(e)}")
