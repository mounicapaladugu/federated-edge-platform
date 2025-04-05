#!/bin/bash
# Initialize directories for the federated learning system

# Create data directories for each node
mkdir -p backend/edge_node/data/node1
mkdir -p backend/edge_node/data/node2
mkdir -p backend/edge_node/data/node3

# Create model directories
mkdir -p backend/edge_node/models
mkdir -p backend/aggregator/models
mkdir -p backend/aggregator/updates

echo "Directory structure initialized successfully!"
