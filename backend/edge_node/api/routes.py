from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import os
import torch
import json
import logging
import requests
from typing import Dict, Any, List, Optional
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime

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

# Store historical metrics
metrics_history = []

# Store data drift information
data_drift_metrics = {}

# Reference data for drift detection
reference_data = None

@router.get("/status")
async def get_status():
    """Get the status of the edge node"""
    model_path = os.path.join("models", f"model_node_{NODE_ID}.pt")
    model_exists = os.path.exists(model_path)
    
    return {
        "node_id": NODE_ID,
        "status": "running",
        "model_loaded": model_exists,
        "training_metrics": training_metrics,
        "data_drift": data_drift_metrics.get("latest", {})
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

@router.get("/metrics/history")
async def get_metrics_history():
    """Get historical training metrics"""
    return metrics_history

@router.get("/data_drift")
async def get_data_drift(round_num: Optional[int] = None):
    """Get data drift metrics, optionally for a specific round"""
    if round_num is not None:
        return data_drift_metrics.get(str(round_num), {})
    return data_drift_metrics

@router.post("/data_drift/calculate")
async def calculate_data_drift(background_tasks: BackgroundTasks):
    """Calculate data drift using current data compared to reference data"""
    global reference_data, data_drift_metrics
    
    try:
        # Load current data
        current_data_path = model_trainer.data_file
        if not os.path.exists(current_data_path):
            raise HTTPException(status_code=404, detail="Current data file not found")
        
        current_data = pd.read_csv(current_data_path)
        
        # If no reference data exists, set current data as reference
        if reference_data is None:
            reference_data = current_data.copy()
            
            # Save reference data
            os.makedirs("reference_data", exist_ok=True)
            reference_data_path = os.path.join("reference_data", f"reference_node_{NODE_ID}.csv")
            reference_data.to_csv(reference_data_path, index=False)
            
            return {"status": "success", "message": "Reference data set. No drift calculation performed."}
        
        # Calculate drift in background
        background_tasks.add_task(calculate_data_drift_task, current_data)
        
        return {"status": "success", "message": "Data drift calculation started in background"}
    
    except Exception as e:
        logger.error(f"Error calculating data drift: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating data drift: {str(e)}")

@router.post("/set_reference_data")
async def set_reference_data():
    """Set current data as reference data for drift detection"""
    global reference_data
    
    try:
        # Load current data
        current_data_path = model_trainer.data_file
        if not os.path.exists(current_data_path):
            raise HTTPException(status_code=404, detail="Current data file not found")
        
        # Set as reference data
        reference_data = pd.read_csv(current_data_path)
        
        # Save reference data
        os.makedirs("reference_data", exist_ok=True)
        reference_data_path = os.path.join("reference_data", f"reference_node_{NODE_ID}.csv")
        reference_data.to_csv(reference_data_path, index=False)
        
        return {"status": "success", "message": "Reference data updated"}
    
    except Exception as e:
        logger.error(f"Error setting reference data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting reference data: {str(e)}")

def train_model_task():
    """Background task to train the model"""
    global training_metrics, metrics_history
    
    try:
        logger.info(f"Starting model training on node {NODE_ID}")
        current_round = training_metrics["current_round"] + 1
        training_metrics["current_round"] = current_round
        training_metrics["training_complete"] = False
        
        # Load the model
        model_path = os.path.join("models", f"model_node_{NODE_ID}.pt")
        
        # Train the model
        results = model_trainer.train(model_path)
        
        # Update metrics
        training_metrics.update(results)
        training_metrics["training_complete"] = True
        training_metrics["timestamp"] = datetime.now().isoformat()
        
        # Store in history
        metrics_entry = training_metrics.copy()
        metrics_entry["round"] = current_round
        metrics_history.append(metrics_entry)
        
        # Save metrics history to file
        os.makedirs("metrics", exist_ok=True)
        with open(os.path.join("metrics", f"metrics_history_node_{NODE_ID}.json"), "w") as f:
            json.dump(metrics_history, f, indent=2)
        
        # Calculate data drift if reference data exists
        if reference_data is not None:
            # Load current data
            current_data_path = model_trainer.data_file
            if os.path.exists(current_data_path):
                current_data = pd.read_csv(current_data_path)
                calculate_data_drift_task(current_data)
        
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


def calculate_data_drift_task(current_data):
    """Calculate data drift between reference and current data"""
    global data_drift_metrics, reference_data
    
    try:
        if reference_data is None:
            logger.warning("Reference data not available for drift calculation")
            return
        
        # Calculate drift for numerical features
        drift_scores = {}
        
        # Get common columns
        common_columns = set(reference_data.columns).intersection(set(current_data.columns))
        numerical_columns = [col for col in common_columns if 
                            pd.api.types.is_numeric_dtype(reference_data[col]) and 
                            pd.api.types.is_numeric_dtype(current_data[col])]        
        
        for col in numerical_columns:
            # Calculate statistical measures
            ref_mean = reference_data[col].mean()
            ref_std = reference_data[col].std()
            cur_mean = current_data[col].mean()
            cur_std = current_data[col].std()
            
            # Calculate normalized difference
            if ref_std > 0:
                mean_diff = abs(cur_mean - ref_mean) / ref_std
            else:
                mean_diff = abs(cur_mean - ref_mean) if cur_mean != ref_mean else 0
            
            # Calculate ratio of standard deviations
            if ref_std > 0 and cur_std > 0:
                std_ratio = max(cur_std / ref_std, ref_std / cur_std)
            else:
                std_ratio = 1.0 if cur_std == ref_std else 2.0
            
            # Calculate drift score (higher means more drift)
            drift_score = (mean_diff + std_ratio - 1) / 2
            drift_scores[col] = drift_score
        
        # Calculate overall drift score (average of feature drift scores)
        if drift_scores:
            overall_drift = sum(drift_scores.values()) / len(drift_scores)
        else:
            overall_drift = 0.0
        
        # Create drift report
        drift_report = {
            "overall_drift": overall_drift,
            "feature_drift": drift_scores,
            "timestamp": datetime.now().isoformat(),
            "round": training_metrics["current_round"]
        }
        
        # Store drift metrics
        round_key = str(training_metrics["current_round"])
        data_drift_metrics[round_key] = drift_report
        data_drift_metrics["latest"] = drift_report
        
        # Save drift metrics to file
        os.makedirs("drift", exist_ok=True)
        with open(os.path.join("drift", f"drift_metrics_node_{NODE_ID}.json"), "w") as f:
            json.dump(data_drift_metrics, f, indent=2)
        
        logger.info(f"Data drift calculated: overall={overall_drift:.4f}")
        
        # If drift is significant, send alert to aggregator
        if overall_drift > 0.3:  # Threshold for significant drift
            try:
                requests.post(
                    f"{AGGREGATOR_URL}/drift_alert",
                    json={
                        "node_id": NODE_ID,
                        "drift_report": drift_report
                    }
                )
                logger.warning(f"Drift alert sent to aggregator: overall={overall_drift:.4f}")
            except Exception as e:
                logger.error(f"Error sending drift alert: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in data drift calculation: {str(e)}")
