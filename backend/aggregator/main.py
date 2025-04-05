import os
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import requests
from typing import Dict, List, Optional, Any, Tuple
import time
from model_aggregator import ModelAggregator
from fl_server import FlowerServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("aggregator")

# Initialize FastAPI app
app = FastAPI(title="Federated Learning Aggregator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get environment variables
NODE_COUNT = int(os.getenv("NODE_COUNT", "3"))

# Initialize model aggregator
model_aggregator = ModelAggregator()

# Initialize Flower server
flower_server = FlowerServer(node_count=NODE_COUNT)

# Store registered nodes
registered_nodes = {}
node_updates = {}
current_round = 0
federated_round_active = False

@app.get("/")
async def root():
    """Root endpoint for the aggregator API"""
    return {
        "message": "Federated Learning Aggregator API",
        "status": "running",
        "registered_nodes": len(registered_nodes),
        "current_round": current_round
    }

@app.post("/register_node")
async def register_node(node_info: Dict):
    """Register an edge node with the aggregator"""
    node_id = node_info.get("node_id")
    api_url = node_info.get("api_url")
    
    if not node_id or not api_url:
        raise HTTPException(status_code=400, detail="Missing node_id or api_url")
    
    registered_nodes[node_id] = {
        "api_url": api_url,
        "last_seen": time.time(),
        "status": "registered"
    }
    
    logger.info(f"Node {node_id} registered with URL {api_url}")
    
    return {"status": "registered", "node_id": node_id}

@app.get("/nodes")
async def get_nodes():
    """Get all registered nodes"""
    return registered_nodes

@app.post("/start_federated_round")
async def start_federated_round(background_tasks: BackgroundTasks, config: Optional[Dict[str, Any]] = None):
    """Start a new federated learning round using Flower"""
    global current_round, federated_round_active, node_updates
    
    if federated_round_active:
        raise HTTPException(status_code=400, detail="A federated round is already active")
    
    if len(registered_nodes) == 0:
        raise HTTPException(status_code=400, detail="No nodes registered")
    
    # Increment round counter
    current_round += 1
    federated_round_active = True
    node_updates = {}
    
    logger.info(f"Starting federated round {current_round}")
    
    # Update Flower server configuration if provided
    if config:
        flower_server.update_config(config)
    
    # Start Flower server in background
    background_tasks.add_task(start_flower_server)
    
    # For backward compatibility, also distribute model using the old method
    model_path = "models/global_model.pt"
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(model_path):
        # Initialize a new model if none exists
        logger.info("Initializing new global model")
        flower_server.initialize_model()
    
    # Create metadata
    metadata = {
        "version": f"round_{current_round}",
        "timestamp": time.time(),
        "round": current_round,
        "fl_server_address": "aggregator:8080"  # Flower server address
    }
    
    # Save metadata
    with open("models/global_metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Distribute model to all nodes
    distribution_results = await distribute_model_to_nodes(model_path, metadata)
    
    return {
        "round": current_round,
        "status": "started",
        "nodes": distribution_results,
        "fl_server": flower_server.get_server_status()
    }

def start_flower_server():
    """Start the Flower server for federated learning"""
    try:
        # Configure the server
        config = {
            "min_fit_clients": max(2, len(registered_nodes) - 1),
            "min_evaluate_clients": max(2, len(registered_nodes) - 1),
            "min_available_clients": len(registered_nodes)
        }
        
        # Start the server
        flower_server.start_server(config)
        
    except Exception as e:
        logger.error(f"Error starting Flower server: {str(e)}")
        
@app.post("/fl/config")
async def update_fl_config(config: Dict[str, Any]):
    """Update the federated learning configuration"""
    try:
        flower_server.update_config(config)
        return {"status": "success", "config": flower_server.config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")

@app.get("/fl/status")
async def get_fl_status():
    """Get the status of the Flower federated learning server"""
    return flower_server.get_server_status()

@app.post("/fl/rollback/{version}")
async def rollback_model(version: int):
    """Roll back to a specific model version"""
    success = flower_server.rollback_model(version)
    if success:
        return {"status": "success", "message": f"Rolled back to version {version}"}
    else:
        raise HTTPException(status_code=404, detail=f"Model version {version} not found or rollback failed")

async def distribute_model_to_nodes(model_path: str, metadata: Dict):
    """Distribute the global model to all registered nodes"""
    results = {}
    
    for node_id, node_info in registered_nodes.items():
        try:
            # Prepare files for upload
            with open(model_path, "rb") as model_file, open("models/global_metadata.json", "rb") as metadata_file:
                files = {
                    "model_file": ("model.pt", model_file, "application/octet-stream"),
                    "metadata": ("metadata.json", metadata_file, "application/json")
                }
                
                # Send to node
                response = requests.post(
                    f"{node_info['api_url']}/api/receive_model",
                    files=files
                )
                
                if response.status_code == 200:
                    logger.info(f"Model distributed to node {node_id}")
                    results[node_id] = "success"
                    registered_nodes[node_id]["status"] = "training"
                else:
                    logger.error(f"Failed to distribute model to node {node_id}: {response.text}")
                    results[node_id] = f"error: {response.text}"
        
        except Exception as e:
            logger.error(f"Error distributing model to node {node_id}: {str(e)}")
            results[node_id] = f"error: {str(e)}"
    
    return results

@app.post("/receive_update")
async def receive_update(
    model_file: UploadFile = File(...),
    metrics: UploadFile = File(...),
    node_id: str = Form(...)
):
    """Receive model updates from an edge node"""
    global node_updates
    
    if not federated_round_active:
        raise HTTPException(status_code=400, detail="No active federated round")
    
    if node_id not in registered_nodes:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not registered")
    
    try:
        # Create directory for node updates
        os.makedirs(f"updates/round_{current_round}", exist_ok=True)
        
        # Save model update
        model_path = f"updates/round_{current_round}/model_node_{node_id}.pt"
        with open(model_path, "wb") as f:
            f.write(await model_file.read())
        
        # Save metrics
        metrics_path = f"updates/round_{current_round}/metrics_node_{node_id}.json"
        metrics_content = await metrics.read()
        with open(metrics_path, "wb") as f:
            f.write(metrics_content)
        
        # Parse metrics
        metrics_dict = json.loads(metrics_content)
        
        # Update node status
        registered_nodes[node_id]["status"] = "update_received"
        registered_nodes[node_id]["last_seen"] = time.time()
        
        # Record update
        node_updates[node_id] = {
            "model_path": model_path,
            "metrics": metrics_dict,
            "timestamp": time.time()
        }
        
        logger.info(f"Received update from node {node_id} for round {current_round}")
        
        # Check if all nodes have reported
        if len(node_updates) == len(registered_nodes):
            # All nodes have reported, trigger aggregation
            await aggregate_models()
        
        return {"status": "update_received", "node_id": node_id}
    
    except Exception as e:
        logger.error(f"Error receiving update from node {node_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error receiving update: {str(e)}")

async def aggregate_models():
    """Aggregate model updates from all nodes"""
    global federated_round_active
    
    logger.info(f"Aggregating models for round {current_round}")
    
    try:
        # Get model paths
        model_paths = [update_info["model_path"] for update_info in node_updates.values()]
        
        # Aggregate models
        aggregated_model = model_aggregator.aggregate_models(model_paths)
        
        # Save aggregated model
        os.makedirs("models", exist_ok=True)
        torch.save(aggregated_model.state_dict(), "models/global_model.pt")
        
        # Update metadata
        metadata = {
            "version": f"round_{current_round}",
            "timestamp": time.time(),
            "round": current_round,
            "participating_nodes": list(node_updates.keys())
        }
        
        with open("models/global_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Reset for next round
        federated_round_active = False
        
        logger.info(f"Model aggregation complete for round {current_round}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during model aggregation: {str(e)}")
        federated_round_active = False
        return False

@app.get("/round_status")
async def get_round_status():
    """Get the status of the current federated round"""
    return {
        "round": current_round,
        "active": federated_round_active,
        "nodes_reported": len(node_updates),
        "total_nodes": len(registered_nodes),
        "node_statuses": {node_id: info["status"] for node_id, info in registered_nodes.items()}
    }

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("updates", exist_ok=True)
    os.makedirs("models/versions", exist_ok=True)
    
    # Run the FastAPI application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
