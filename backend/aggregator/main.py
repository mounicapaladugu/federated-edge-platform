import os
import uvicorn
import logging
import pandas as pd
import numpy as np
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
import torch
import json
import requests
from typing import Dict, List, Optional, Any, Tuple
import time
import threading
from model_aggregator import ModelAggregator
from fl_server import FlowerServer
from common.fl.monitoring import FederatedMonitor
from common.fl.secure_transfer import SecureModelTransfer

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

# Initialize monitoring system
monitor = FederatedMonitor(base_dir="monitoring")

# Initialize secure transfer
SECURE_MOUNT_DIR = os.getenv("SECURE_MOUNT_DIR", "/mnt/secure_transfer")
AIR_GAPPED_MODE = os.getenv("AIR_GAPPED_MODE", "false").lower() == "true"
secure_transfer = SecureModelTransfer(is_aggregator=True, mount_dir=SECURE_MOUNT_DIR)

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
        # Log the rollback in monitoring system
        monitor.log_deployment(
            model_version=version,
            round_num=flower_server.current_round,
            metadata={"action": "rollback", "reason": "manual_rollback"}
        )
        return {"status": "success", "message": f"Rolled back to version {version}"}
    else:
        raise HTTPException(status_code=404, detail=f"Model version {version} not found or rollback failed")

async def distribute_model_to_nodes(model_path: str, metadata: Dict):
    """Distribute the global model to all registered nodes"""
    results = {}
    
    # If in air-gapped mode, export the model to the secure mount directory
    if AIR_GAPPED_MODE:
        logger.info("Air-gapped mode: exporting global model to secure mount directory")
        success = secure_transfer.export_global_model(model_path, metadata)
        if success:
            logger.info("Global model exported successfully to secure mount directory")
            results["secure_export"] = "success"
        else:
            logger.error("Failed to export global model to secure mount directory")
            results["secure_export"] = "error: failed to export model"
        return results
    
    # In normal mode, distribute the model to each node via HTTP
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
        
        # Log node metrics in monitoring system
        monitor.log_node_metrics(
            node_id=node_id,
            round_num=current_round,
            metrics=metrics_dict
        )
        
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
        version = current_round  # Use round number as version
        metadata = {
            "version": version,
            "timestamp": time.time(),
            "round": current_round,
            "participating_nodes": list(node_updates.keys())
        }
        
        with open("models/global_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Save versioned copy
        os.makedirs("models/versions", exist_ok=True)
        torch.save(aggregated_model.state_dict(), f"models/versions/global_model_v{version}.pt")
        
        with open(f"models/versions/global_metadata_v{version}.json", "w") as f:
            json.dump(metadata, f)
        
        # Evaluate aggregated model (mock metrics for now)
        metrics = model_aggregator.evaluate_aggregated_model(aggregated_model)
        
        # Log metrics in monitoring system
        monitor.log_aggregated_metrics(
            round_num=current_round,
            metrics=metrics,
            model_metadata=metadata
        )
        
        # Log deployment
        monitor.log_deployment(
            model_version=version,
            round_num=current_round,
            metadata={"action": "deployment", "metrics": metrics}
        )
        
        # Check if we should automatically rollback based on performance
        if current_round > 1:
            # Get previous round metrics
            prev_round_metrics = monitor.get_round_metrics(current_round - 1)
            if prev_round_metrics and "aggregated" in prev_round_metrics:
                prev_metrics = prev_round_metrics["aggregated"]["metrics"]
                
                # Check if we should rollback
                should_rollback, reason = monitor.should_rollback(metrics, prev_metrics)
                if should_rollback:
                    # Get previous version
                    prev_version = current_round - 1
                    
                    # Rollback to previous version
                    success = flower_server.rollback_model(prev_version)
                    if success:
                        logger.warning(f"Automatic rollback to version {prev_version}: {reason}")
                        
                        # Log the automatic rollback
                        monitor.log_deployment(
                            model_version=prev_version,
                            round_num=current_round,
                            metadata={"action": "auto_rollback", "reason": reason}
                        )
        
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

# Monitoring and model management endpoints

@app.get("/monitoring/metrics")
async def get_metrics(node_id: Optional[str] = None, round_num: Optional[int] = None):
    """Get monitoring metrics, optionally filtered by node or round"""
    if node_id:
        return monitor.get_node_metrics_history(node_id)
    elif round_num:
        return monitor.get_round_metrics(round_num)
    else:
        return monitor.get_all_metrics()

@app.get("/monitoring/deployment_logs")
async def get_deployment_logs():
    """Get all model deployment logs"""
    return monitor.get_deployment_logs()

@app.get("/monitoring/plots/{metric_name}")
async def get_metric_plot(
    metric_name: str = Path(..., description="Metric name to plot (e.g., accuracy, loss)"),
    node_id: Optional[str] = Query(None, description="Optional node ID to filter by")
):
    """Get a plot of metrics over rounds"""
    img_str = monitor.generate_metrics_plot(metric_name, node_id)
    if not img_str:
        raise HTTPException(status_code=404, detail=f"No {metric_name} data found")
    
    return Response(content=f'<img src="data:image/png;base64,{img_str}" />', media_type="text/html")

@app.get("/models/versions")
async def get_model_versions():
    """Get all available model versions"""
    versions_dir = "models/versions"
    if not os.path.exists(versions_dir):
        return []
    
    versions = []
    for filename in os.listdir(versions_dir):
        if filename.startswith("global_metadata_v") and filename.endswith(".json"):
            try:
                with open(os.path.join(versions_dir, filename), "r") as f:
                    metadata = json.load(f)
                    versions.append(metadata)
            except Exception as e:
                logger.error(f"Error loading model metadata {filename}: {str(e)}")
    
    # Sort by version number (descending)
    versions.sort(key=lambda x: x.get("version", 0), reverse=True)
    
    return versions

@app.post("/monitoring/thresholds")
async def update_validation_thresholds(thresholds: Dict[str, float]):
    """Update validation thresholds for automatic rollback"""
    monitor.update_validation_thresholds(thresholds)
    return {"status": "success", "thresholds": monitor.validation_thresholds}

@app.post("/data_drift/calculate")
async def calculate_data_drift(
    node_id: str = Form(...),
    reference_data: UploadFile = File(...),
    current_data: UploadFile = File(...)
):
    """Calculate data drift between reference and current data"""
    try:
        # Read CSV files
        reference_df = pd.read_csv(reference_data.file)
        current_df = pd.read_csv(current_data.file)
        
        # Calculate drift
        drift_scores = monitor.calculate_data_drift(node_id, reference_df, current_df)
        
        return {"node_id": node_id, "drift_scores": drift_scores}
    
    except Exception as e:
        logger.error(f"Error calculating data drift: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating data drift: {str(e)}")

@app.get("/monitoring/dashboard")
async def get_monitoring_dashboard():
    """Get a simple HTML dashboard for monitoring"""
    # Get metrics and model versions
    metrics = monitor.get_all_metrics()
    deployment_logs = monitor.get_deployment_logs()
    
    # Create a simple HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Federated Learning Monitoring Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .card {{ background: #f9f9f9; border-radius: 5px; padding: 15px; margin: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metrics {{ width: 45%; }}
            .models {{ width: 45%; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .plot {{ margin: 20px 0; }}
            .button {{ background-color: #4CAF50; border: none; color: white; padding: 10px 15px; 
                      text-align: center; text-decoration: none; display: inline-block; 
                      font-size: 14px; margin: 4px 2px; cursor: pointer; border-radius: 4px; }}
            .button.rollback {{ background-color: #f44336; }}
        </style>
    </head>
    <body>
        <h1>Federated Learning Monitoring Dashboard</h1>
        
        <div class="container">
            <div class="card metrics">
                <h2>Current Round Status</h2>
                <p>Round: {current_round}</p>
                <p>Active: {federated_round_active}</p>
                <p>Nodes Reported: {len(node_updates)} / {len(registered_nodes)}</p>
            </div>
            
            <div class="card metrics">
                <h2>Metrics Visualization</h2>
                <div class="plot">
                    <img src="/monitoring/plots/accuracy" alt="Accuracy Plot" width="100%">
                </div>
                <div class="plot">
                    <img src="/monitoring/plots/loss" alt="Loss Plot" width="100%">
                </div>
            </div>
        </div>
        
        <div class="card models">
            <h2>Model Versions</h2>
            <table>
                <tr>
                    <th>Version</th>
                    <th>Round</th>
                    <th>Timestamp</th>
                    <th>Actions</th>
                </tr>
    """
    
    # Add model versions to the table
    versions_dir = "models/versions"
    if os.path.exists(versions_dir):
        versions = []
        for filename in os.listdir(versions_dir):
            if filename.startswith("global_metadata_v") and filename.endswith(".json"):
                try:
                    with open(os.path.join(versions_dir, filename), "r") as f:
                        metadata = json.load(f)
                        versions.append(metadata)
                except Exception as e:
                    logger.error(f"Error loading model metadata {filename}: {str(e)}")
        
        # Sort by version number (descending)
        versions.sort(key=lambda x: x.get("version", 0), reverse=True)
        
        for version in versions:
            version_num = version.get("version", "Unknown")
            round_num = version.get("round", "Unknown")
            timestamp = version.get("timestamp", "Unknown")
            
            # Format timestamp
            if isinstance(timestamp, (int, float)):
                from datetime import datetime
                timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            html_content += f"""
                <tr>
                    <td>{version_num}</td>
                    <td>{round_num}</td>
                    <td>{timestamp}</td>
                    <td>
                        <a href="/fl/rollback/{version_num}" class="button rollback" 
                           onclick="return confirm('Are you sure you want to rollback to version {version_num}?')">
                           Rollback
                        </a>
                    </td>
                </tr>
            """
    
    html_content += """
            </table>
        </div>
        
        <div class="card">
            <h2>Deployment Logs</h2>
            <table>
                <tr>
                    <th>Version</th>
                    <th>Round</th>
                    <th>Timestamp</th>
                    <th>Action</th>
                </tr>
    """
    
    # Add deployment logs to the table
    for log in deployment_logs:
        version = log.get("model_version", "Unknown")
        round_num = log.get("round", "Unknown")
        timestamp = log.get("timestamp", "Unknown")
        action = log.get("metadata", {}).get("action", "deployment")
        
        html_content += f"""
            <tr>
                <td>{version}</td>
                <td>{round_num}</td>
                <td>{timestamp}</td>
                <td>{action}</td>
            </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.post("/secure/check_updates")
async def check_secure_updates(background_tasks: BackgroundTasks):
    """Check for model updates in the secure mount directory"""
    if not AIR_GAPPED_MODE:
        raise HTTPException(status_code=400, detail="This endpoint is only available in air-gapped mode")
    
    background_tasks.add_task(import_secure_updates)
    return {"status": "checking", "message": "Checking for model updates in secure mount directory"}

def import_secure_updates():
    """Import model updates from the secure mount directory"""
    try:
        logger.info("Checking for model updates in secure mount directory...")
        updates = secure_transfer.import_model_updates()
        
        if updates:
            logger.info(f"Found {len(updates)} model updates in secure mount directory")
            
            # Process each update
            for update in updates:
                node_id = update["node_id"]
                model_path = update["model_path"]
                metadata = update["metadata"]
                
                logger.info(f"Processing update from node {node_id}")
                
                # Save update to the updates directory
                os.makedirs(f"updates/node_{node_id}", exist_ok=True)
                update_path = f"updates/node_{node_id}/update_{int(time.time())}.pt"
                shutil.copy(model_path, update_path)
                
                # Update node status if registered
                if node_id in registered_nodes:
                    registered_nodes[node_id]["status"] = "update_received"
                    registered_nodes[node_id]["last_seen"] = time.time()
                
                # Save metrics
                metrics = metadata.get("metrics", {})
                if metrics:
                    # Log metrics to monitoring system
                    monitor.log_metrics(
                        node_id=node_id,
                        round_num=current_round,
                        metrics=metrics
                    )
                    
                    # Store update for aggregation
                    node_updates[node_id] = {
                        "model_path": update_path,
                        "metrics": metrics,
                        "timestamp": time.time()
                    }
            
            return True
        else:
            logger.info("No model updates found")
            return False
            
    except Exception as e:
        logger.error(f"Error importing model updates: {str(e)}")
        return False

def check_secure_updates_periodically():
    """Background thread to periodically check for model updates in air-gapped mode"""
    while True:
        try:
            if AIR_GAPPED_MODE:
                logger.info("Checking for model updates in secure mount directory...")
                import_secure_updates()
        except Exception as e:
            logger.error(f"Error in secure update check thread: {str(e)}")
            
        # Wait before checking again (e.g., every 5 minutes)
        check_interval = int(os.getenv("SECURE_CHECK_INTERVAL", "300"))
        time.sleep(check_interval)

@app.post("/secure/export_model")
async def manual_export_model():
    """Manually export the global model to the secure mount directory"""
    if not AIR_GAPPED_MODE:
        raise HTTPException(status_code=400, detail="This endpoint is only available in air-gapped mode")
    
    try:
        model_path = "models/global_model.pt"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Global model not found")
        
        # Load metadata
        metadata_path = "models/global_metadata.json"
        if not os.path.exists(metadata_path):
            # Create default metadata if none exists
            metadata = {
                "version": f"manual_export_{int(time.time())}",
                "timestamp": time.time(),
                "round": current_round
            }
        else:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
        # Export the model
        success = secure_transfer.export_global_model(model_path, metadata)
        
        if success:
            return {"status": "success", "message": "Global model exported successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to export global model")
    except Exception as e:
        logger.error(f"Error exporting global model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting global model: {str(e)}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("updates", exist_ok=True)
    os.makedirs("models/versions", exist_ok=True)
    
    # Start secure update check thread if in air-gapped mode
    if AIR_GAPPED_MODE:
        secure_check_thread = threading.Thread(target=check_secure_updates_periodically, daemon=True)
        secure_check_thread.start()
        logger.info("Started secure update check thread")
    
    # Run the FastAPI application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
