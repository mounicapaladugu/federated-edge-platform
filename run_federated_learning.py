#!/usr/bin/env python3
"""
Script to manage the federated learning process for the edge nodes.
This script provides a CLI interface to:
1. Start the federated learning system
2. Initiate federated learning rounds
3. Monitor the status of edge nodes and training
"""

import argparse
import requests
import json
import time
import os
import sys
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("federated_manager")

def initialize_data():
    """Initialize data for all edge nodes"""
    logger.info("Initializing data for edge nodes...")
    try:
        # Import and run the data initialization script
        sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
        from init_data import main as init_data_main
        init_data_main()
        logger.info("Data initialization complete")
        return True
    except Exception as e:
        logger.error(f"Error initializing data: {str(e)}")
        return False

def start_federated_round(aggregator_url: str = "http://localhost:8000"):
    """Start a new federated learning round"""
    logger.info("Starting new federated learning round...")
    try:
        response = requests.post(f"{aggregator_url}/start_federated_round")
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Federated round {result['round']} started successfully")
            logger.info(f"Distribution results: {json.dumps(result['nodes'], indent=2)}")
            return result
        else:
            logger.error(f"Failed to start federated round: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error starting federated round: {str(e)}")
        return None

def get_round_status(aggregator_url: str = "http://localhost:8000"):
    """Get the status of the current federated round"""
    try:
        response = requests.get(f"{aggregator_url}/round_status")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get round status: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting round status: {str(e)}")
        return None

def get_node_status(node_id: str, node_url: str):
    """Get the status of a specific edge node"""
    try:
        response = requests.get(f"{node_url}/api/status")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get node {node_id} status: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting node {node_id} status: {str(e)}")
        return None

def monitor_federated_round(aggregator_url: str = "http://localhost:8000", interval: int = 5, timeout: int = 300):
    """Monitor the progress of a federated learning round"""
    logger.info("Monitoring federated round progress...")
    
    start_time = time.time()
    completed = False
    
    while not completed and (time.time() - start_time) < timeout:
        # Get round status
        status = get_round_status(aggregator_url)
        
        if not status:
            logger.error("Failed to get round status")
            time.sleep(interval)
            continue
        
        logger.info(f"Round {status['round']} - Active: {status['active']}")
        logger.info(f"Nodes reported: {status['nodes_reported']}/{status['total_nodes']}")
        
        # Check node statuses
        for node_id, node_status in status['node_statuses'].items():
            logger.info(f"Node {node_id}: {node_status}")
            
            # Get detailed node status if available
            node_url = f"http://localhost:800{node_id}"
            node_details = get_node_status(node_id, node_url)
            if node_details and 'training_metrics' in node_details:
                metrics = node_details['training_metrics']
                if 'accuracy' in metrics:
                    logger.info(f"  - Accuracy: {metrics['accuracy']:.2f}%")
                if 'loss' in metrics:
                    logger.info(f"  - Loss: {metrics['loss']:.4f}")
        
        # Check if round is complete
        if not status['active'] and status['nodes_reported'] == status['total_nodes']:
            logger.info(f"Federated round {status['round']} completed successfully!")
            completed = True
            break
        
        time.sleep(interval)
    
    if not completed:
        logger.warning(f"Monitoring timed out after {timeout} seconds")
    
    return completed

def main():
    """Main function to manage the federated learning process"""
    parser = argparse.ArgumentParser(description="Federated Learning Manager")
    parser.add_argument("--init-data", action="store_true", help="Initialize data for edge nodes")
    parser.add_argument("--start-round", action="store_true", help="Start a new federated learning round")
    parser.add_argument("--monitor", action="store_true", help="Monitor the progress of a federated round")
    parser.add_argument("--aggregator", type=str, default="http://localhost:8000", help="URL of the aggregator service")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--timeout", type=int, default=300, help="Monitoring timeout in seconds")
    
    args = parser.parse_args()
    
    if args.init_data:
        initialize_data()
    
    if args.start_round:
        start_federated_round(args.aggregator)
    
    if args.monitor:
        monitor_federated_round(args.aggregator, args.interval, args.timeout)
    
    if not any([args.init_data, args.start_round, args.monitor]):
        # If no arguments provided, show help
        parser.print_help()

if __name__ == "__main__":
    main()
