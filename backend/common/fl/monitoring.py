import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy.stats import wasserstein_distance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fl_monitoring")

class FederatedMonitor:
    """
    Monitor for tracking federated learning metrics, data drift, and model performance.
    """
    def __init__(self, base_dir: str = "monitoring"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "drift"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = self._load_metrics_history()
        
        # Initialize deployment logs
        self.deployment_logs = self._load_deployment_logs()
        
        # Validation thresholds for auto-rollback
        self.validation_thresholds = {
            "accuracy": 0.7,  # Minimum acceptable accuracy
            "max_loss_increase": 0.2,  # Maximum acceptable loss increase
            "drift_threshold": 0.3  # Maximum acceptable drift score
        }
    
    def _load_metrics_history(self) -> Dict:
        """Load metrics history from file or initialize if not exists"""
        metrics_file = os.path.join(self.base_dir, "metrics_history.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics history: {str(e)}")
                return {"rounds": {}, "nodes": {}}
        else:
            return {"rounds": {}, "nodes": {}}
    
    def _save_metrics_history(self):
        """Save metrics history to file"""
        metrics_file = os.path.join(self.base_dir, "metrics_history.json")
        try:
            with open(metrics_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics history: {str(e)}")
    
    def _load_deployment_logs(self) -> List[Dict]:
        """Load deployment logs from file or initialize if not exists"""
        logs_file = os.path.join(self.base_dir, "deployment_logs.json")
        if os.path.exists(logs_file):
            try:
                with open(logs_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading deployment logs: {str(e)}")
                return []
        else:
            return []
    
    def _save_deployment_logs(self):
        """Save deployment logs to file"""
        logs_file = os.path.join(self.base_dir, "deployment_logs.json")
        try:
            with open(logs_file, "w") as f:
                json.dump(self.deployment_logs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving deployment logs: {str(e)}")
    
    def log_node_metrics(self, node_id: str, round_num: int, metrics: Dict[str, Any]):
        """
        Log metrics from an edge node for a specific round
        
        Args:
            node_id: ID of the edge node
            round_num: Federated learning round number
            metrics: Dictionary of metrics (accuracy, loss, etc.)
        """
        timestamp = datetime.now().isoformat()
        
        # Initialize round in history if not exists
        round_key = str(round_num)
        if round_key not in self.metrics_history["rounds"]:
            self.metrics_history["rounds"][round_key] = {
                "nodes": {},
                "aggregated": None,
                "timestamp": timestamp
            }
        
        # Initialize node in history if not exists
        if node_id not in self.metrics_history["nodes"]:
            self.metrics_history["nodes"][node_id] = []
        
        # Add metrics to round history
        self.metrics_history["rounds"][round_key]["nodes"][node_id] = {
            "metrics": metrics,
            "timestamp": timestamp
        }
        
        # Add metrics to node history
        self.metrics_history["nodes"][node_id].append({
            "round": round_num,
            "metrics": metrics,
            "timestamp": timestamp
        })
        
        # Save metrics to file
        metrics_file = os.path.join(self.base_dir, "metrics", f"node_{node_id}_round_{round_num}.json")
        with open(metrics_file, "w") as f:
            json.dump({
                "node_id": node_id,
                "round": round_num,
                "metrics": metrics,
                "timestamp": timestamp
            }, f, indent=2)
        
        # Save updated metrics history
        self._save_metrics_history()
        
        logger.info(f"Logged metrics for node {node_id} in round {round_num}")
    
    def log_aggregated_metrics(self, round_num: int, metrics: Dict[str, Any], model_metadata: Dict[str, Any]):
        """
        Log metrics from the aggregated global model
        
        Args:
            round_num: Federated learning round number
            metrics: Dictionary of metrics (accuracy, loss, etc.)
            model_metadata: Metadata about the model (version, etc.)
        """
        timestamp = datetime.now().isoformat()
        
        # Initialize round in history if not exists
        round_key = str(round_num)
        if round_key not in self.metrics_history["rounds"]:
            self.metrics_history["rounds"][round_key] = {
                "nodes": {},
                "timestamp": timestamp
            }
        
        # Add aggregated metrics to round history
        self.metrics_history["rounds"][round_key]["aggregated"] = {
            "metrics": metrics,
            "metadata": model_metadata,
            "timestamp": timestamp
        }
        
        # Save metrics to file
        metrics_file = os.path.join(self.base_dir, "metrics", f"aggregated_round_{round_num}.json")
        with open(metrics_file, "w") as f:
            json.dump({
                "round": round_num,
                "metrics": metrics,
                "metadata": model_metadata,
                "timestamp": timestamp
            }, f, indent=2)
        
        # Save updated metrics history
        self._save_metrics_history()
        
        logger.info(f"Logged aggregated metrics for round {round_num}")
    
    def log_deployment(self, model_version: int, round_num: int, metadata: Dict[str, Any]):
        """
        Log a model deployment
        
        Args:
            model_version: Version of the deployed model
            round_num: Federated learning round number
            metadata: Additional metadata about the deployment
        """
        timestamp = datetime.now().isoformat()
        
        # Create deployment log entry
        deployment_log = {
            "model_version": model_version,
            "round": round_num,
            "timestamp": timestamp,
            "metadata": metadata
        }
        
        # Add to deployment logs
        self.deployment_logs.append(deployment_log)
        
        # Save to file
        self._save_deployment_logs()
        
        # Also save as individual log file
        log_file = os.path.join(self.base_dir, "logs", f"deployment_v{model_version}_r{round_num}.json")
        with open(log_file, "w") as f:
            json.dump(deployment_log, f, indent=2)
        
        logger.info(f"Logged deployment of model version {model_version} from round {round_num}")
    
    def calculate_data_drift(self, node_id: str, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate data drift between reference and current data distributions
        
        Args:
            node_id: ID of the edge node
            reference_data: Reference data (baseline)
            current_data: Current data to compare against reference
            
        Returns:
            Dictionary of drift metrics per feature
        """
        drift_scores = {}
        
        # Ensure both dataframes have the same columns
        common_columns = set(reference_data.columns).intersection(set(current_data.columns))
        
        for col in common_columns:
            if pd.api.types.is_numeric_dtype(reference_data[col]) and pd.api.types.is_numeric_dtype(current_data[col]):
                # Calculate Wasserstein distance (Earth Mover's Distance) for numeric features
                # This measures the minimum "cost" of transforming one distribution into another
                drift_scores[col] = wasserstein_distance(
                    reference_data[col].dropna().values,
                    current_data[col].dropna().values
                )
        
        # Calculate overall drift score (average of feature drift scores)
        if drift_scores:
            drift_scores["overall"] = sum(drift_scores.values()) / len(drift_scores)
        else:
            drift_scores["overall"] = 0.0
        
        # Save drift scores
        timestamp = datetime.now().isoformat()
        drift_file = os.path.join(self.base_dir, "drift", f"drift_{node_id}_{timestamp.replace(':', '-')}.json")
        with open(drift_file, "w") as f:
            json.dump({
                "node_id": node_id,
                "timestamp": timestamp,
                "drift_scores": drift_scores
            }, f, indent=2)
        
        logger.info(f"Calculated data drift for node {node_id}: overall={drift_scores['overall']:.4f}")
        
        return drift_scores
    
    def should_rollback(self, current_metrics: Dict[str, Any], previous_metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if model should be rolled back based on validation thresholds
        
        Args:
            current_metrics: Current model metrics
            previous_metrics: Previous model metrics
            
        Returns:
            Tuple of (should_rollback, reason)
        """
        reasons = []
        
        # Check accuracy threshold
        if current_metrics.get("accuracy", 1.0) < self.validation_thresholds["accuracy"]:
            reasons.append(f"Accuracy below threshold: {current_metrics.get('accuracy', 0):.2f} < {self.validation_thresholds['accuracy']}")
        
        # Check loss increase
        if previous_metrics and "loss" in current_metrics and "loss" in previous_metrics:
            loss_increase = current_metrics["loss"] - previous_metrics["loss"]
            if loss_increase > self.validation_thresholds["max_loss_increase"]:
                reasons.append(f"Loss increase above threshold: {loss_increase:.2f} > {self.validation_thresholds['max_loss_increase']}")
        
        # Check drift threshold if available
        if "drift" in current_metrics and current_metrics["drift"].get("overall", 0) > self.validation_thresholds["drift_threshold"]:
            reasons.append(f"Data drift above threshold: {current_metrics['drift'].get('overall', 0):.2f} > {self.validation_thresholds['drift_threshold']}")
        
        should_rollback = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No issues detected"
        
        if should_rollback:
            logger.warning(f"Automatic rollback recommended: {reason}")
        
        return should_rollback, reason
    
    def update_validation_thresholds(self, thresholds: Dict[str, float]):
        """Update validation thresholds for automatic rollback"""
        self.validation_thresholds.update(thresholds)
        
        # Save thresholds to file
        thresholds_file = os.path.join(self.base_dir, "validation_thresholds.json")
        with open(thresholds_file, "w") as f:
            json.dump(self.validation_thresholds, f, indent=2)
        
        logger.info(f"Updated validation thresholds: {self.validation_thresholds}")
    
    def get_node_metrics_history(self, node_id: str) -> List[Dict]:
        """Get metrics history for a specific node"""
        return self.metrics_history["nodes"].get(node_id, [])
    
    def get_round_metrics(self, round_num: int) -> Dict:
        """Get metrics for a specific round"""
        return self.metrics_history["rounds"].get(str(round_num), {})
    
    def get_all_metrics(self) -> Dict:
        """Get all metrics history"""
        return self.metrics_history
    
    def get_deployment_logs(self) -> List[Dict]:
        """Get all deployment logs"""
        return self.deployment_logs
    
    def generate_metrics_plot(self, metric_name: str = "accuracy", node_id: Optional[str] = None) -> str:
        """
        Generate a plot of metrics over rounds
        
        Args:
            metric_name: Name of the metric to plot
            node_id: Optional node ID to filter by (None for all nodes)
            
        Returns:
            Base64 encoded PNG image
        """
        plt.figure(figsize=(10, 6))
        
        if node_id:
            # Plot metrics for a specific node
            node_history = self.get_node_metrics_history(node_id)
            if not node_history:
                logger.warning(f"No metrics history found for node {node_id}")
                return ""
            
            rounds = [entry["round"] for entry in node_history if metric_name in entry["metrics"]]
            values = [entry["metrics"][metric_name] for entry in node_history if metric_name in entry["metrics"]]
            
            if not rounds:
                logger.warning(f"No {metric_name} metrics found for node {node_id}")
                return ""
            
            plt.plot(rounds, values, marker='o', label=f"Node {node_id}")
            plt.title(f"{metric_name.capitalize()} for Node {node_id}")
        
        else:
            # Plot metrics for all nodes
            node_ids = list(self.metrics_history["nodes"].keys())
            
            for node_id in node_ids:
                node_history = self.get_node_metrics_history(node_id)
                rounds = [entry["round"] for entry in node_history if metric_name in entry["metrics"]]
                values = [entry["metrics"][metric_name] for entry in node_history if metric_name in entry["metrics"]]
                
                if rounds:
                    plt.plot(rounds, values, marker='o', label=f"Node {node_id}")
            
            # Also plot aggregated metrics if available
            agg_rounds = []
            agg_values = []
            
            for round_key, round_data in self.metrics_history["rounds"].items():
                if round_data.get("aggregated") and metric_name in round_data["aggregated"].get("metrics", {}):
                    agg_rounds.append(int(round_key))
                    agg_values.append(round_data["aggregated"]["metrics"][metric_name])
            
            if agg_rounds:
                plt.plot(agg_rounds, agg_values, marker='s', linestyle='--', linewidth=2, label="Aggregated Model")
            
            plt.title(f"{metric_name.capitalize()} Across Nodes")
        
        plt.xlabel("Round")
        plt.ylabel(metric_name.capitalize())
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        
        # Convert to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
