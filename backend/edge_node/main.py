import os
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import requests
from api.routes import router as api_router
from training.trainer import ModelTrainer
from fl_client import EdgeNodeFlowerClient
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("edge_node")

# Initialize FastAPI app
app = FastAPI(title="Edge Node API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)

# Get environment variables
NODE_ID = os.getenv("NODE_ID", "1")
AGGREGATOR_URL = os.getenv("AGGREGATOR_URL", "http://localhost:8000")

# Initialize model trainer
model_trainer = ModelTrainer(node_id=NODE_ID)

# Initialize Flower client
fl_client = EdgeNodeFlowerClient(node_id=NODE_ID, aggregator_url=AGGREGATOR_URL)

# Health monitoring variables
health_metrics = {
    "status": "healthy",
    "cpu_usage": 0.0,
    "memory_usage": 0.0,
    "last_training_time": None,
    "model_version": None,
    "training_metrics": {}
}

def update_health_metrics():
    """Update system health metrics periodically"""
    while True:
        # In a real implementation, this would collect actual system metrics
        # For simulation, we'll just update with mock data
        import random
        health_metrics["cpu_usage"] = random.uniform(10.0, 80.0)
        health_metrics["memory_usage"] = random.uniform(20.0, 70.0)
        time.sleep(30)  # Update every 30 seconds

# Start health monitoring in a background thread
health_thread = threading.Thread(target=update_health_metrics, daemon=True)
health_thread.start()

@app.on_event("startup")
async def startup_event():
    """Register with the aggregator on startup"""
    logger.info(f"Edge Node {NODE_ID} starting up")
    try:
        response = requests.post(
            f"{AGGREGATOR_URL}/register_node",
            json={"node_id": NODE_ID, "api_url": f"http://edge-node-{NODE_ID}:8001"}
        )
        if response.status_code == 200:
            logger.info(f"Successfully registered with aggregator")
        else:
            logger.error(f"Failed to register with aggregator: {response.text}")
    except Exception as e:
        logger.error(f"Error connecting to aggregator: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint for the edge node API"""
    return {
        "message": f"Edge Node {NODE_ID} API",
        "status": "running"
    }

@app.get("/health")
async def get_health():
    """Get the health status of the edge node"""
    return health_metrics

@app.post("/fl/start")
async def start_flower_client(background_tasks: BackgroundTasks, server_address: str = "aggregator:8080"):
    """Start Flower client and connect to the server"""
    try:
        # Start Flower client in background
        background_tasks.add_task(start_fl_client, server_address)
        return {"status": "started", "server_address": server_address}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting Flower client: {str(e)}")

def start_fl_client(server_address: str):
    """Start Flower client in background"""
    try:
        # Load model if available
        model_path = f"models/model_node_{NODE_ID}.pt"
        if os.path.exists(model_path):
            fl_client.load_model(model_path)
        
        # Start client
        fl_client.start_client(server_address)
    except Exception as e:
        logger.error(f"Error in Flower client: {str(e)}")

@app.get("/fl/status")
async def get_fl_status():
    """Get Flower client status"""
    return fl_client.get_client_status()

@app.post("/fl/privacy")
async def update_privacy_settings(settings: dict):
    """Update privacy settings for federated learning"""
    try:
        fl_client.update_privacy_settings(settings)
        return {"status": "success", "settings": fl_client.privacy_setting}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating privacy settings: {str(e)}")

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
