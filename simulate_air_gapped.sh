#!/bin/bash

# Script to demonstrate the secure/air-gapped simulation for federated learning

# Create the secure transfer directory structure if it doesn't exist
mkdir -p secure_transfer

# Function to display section headers
print_header() {
  echo ""
  echo "==============================================="
  echo "$1"
  echo "==============================================="
  echo ""
}

# Function to wait for user input to continue
wait_for_input() {
  read -p "Press Enter to continue..."
}

print_header "Secure/Air-Gapped Federated Learning Simulation"
echo "This script will demonstrate the secure/air-gapped federated learning setup."
echo "It simulates an environment where edge nodes have restricted connectivity"
echo "and can only exchange models via a secure mount (like a USB drive)."
echo ""
echo "The simulation will:"
echo "1. Start the system in air-gapped mode"
echo "2. Export a global model from the aggregator to the secure mount"
echo "3. Allow edge nodes to fetch models from the secure mount"
echo "4. Allow edge nodes to export model updates to the secure mount"
echo "5. Allow the aggregator to import model updates from the secure mount"
echo ""

wait_for_input

print_header "Starting the system in air-gapped mode"
echo "Setting AIR_GAPPED_MODE=true and starting the containers..."
export AIR_GAPPED_MODE=true
docker-compose down
docker-compose up -d
echo "Waiting for services to start..."
sleep 10

wait_for_input

print_header "Checking the status of the services"
echo "Aggregator status:"
curl -s http://localhost:8000/ | jq .
echo ""
echo "Edge Node 1 status:"
curl -s http://localhost:8001/ | jq .
echo ""
echo "Edge Node 2 status:"
curl -s http://localhost:8002/ | jq .
echo ""
echo "Edge Node 3 status:"
curl -s http://localhost:8003/ | jq .

wait_for_input

print_header "Starting a federated learning round"
echo "This will export the global model to the secure mount directory..."
curl -s -X POST http://localhost:8000/start_federated_round | jq .

wait_for_input

print_header "Manually checking for new models on Edge Node 1"
echo "Edge Node 1 will check the secure mount directory for new models..."
curl -s -X GET http://localhost:8001/secure/check_models | jq .
echo ""
echo "Waiting for model import and training to complete..."
sleep 10

wait_for_input

print_header "Checking training status on Edge Node 1"
curl -s -X GET http://localhost:8001/api/training_status | jq .

wait_for_input

print_header "Manually exporting model update from Edge Node 1"
echo "Edge Node 1 will export its model update to the secure mount directory..."
curl -s -X POST http://localhost:8001/secure/export_update | jq .

wait_for_input

print_header "Checking for model updates on the Aggregator"
echo "The aggregator will check the secure mount directory for model updates..."
curl -s -X POST http://localhost:8000/secure/check_updates | jq .
echo ""
echo "Waiting for update import to complete..."
sleep 5

wait_for_input

print_header "Checking node updates status on the Aggregator"
curl -s -X GET http://localhost:8000/nodes | jq .

wait_for_input

print_header "Simulation Complete"
echo "The secure/air-gapped simulation has demonstrated:"
echo "- Edge nodes fetching models at intervals via secure mount"
echo "- Disabled direct network communication between orchestrator and edge nodes"
echo "- Pull-based update mechanism instead of push"
echo "- Minimal network exposure between components"
echo ""
echo "To run the system in normal mode, set AIR_GAPPED_MODE=false:"
echo "export AIR_GAPPED_MODE=false"
echo "docker-compose down"
echo "docker-compose up -d"
echo ""
echo "To stop the containers:"
echo "docker-compose down"

# Make the script executable
chmod +x simulate_air_gapped.sh
