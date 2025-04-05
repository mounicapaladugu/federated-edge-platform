import os
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import requests
from api.routes import router as api_router
from training.trainer import ModelTrainer
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

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
