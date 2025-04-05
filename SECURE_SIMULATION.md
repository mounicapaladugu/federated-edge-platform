# Secure/Air-Gapped Federated Learning Simulation

This document explains the secure/air-gapped simulation feature implemented in the federated edge platform.

## Overview

The secure/air-gapped simulation allows you to test federated learning in environments with restricted connectivity, such as:

- Air-gapped networks where direct internet access is prohibited
- Secure environments with limited network exposure
- Scenarios requiring physical media (USB) for model transfer
- Pull-based update mechanisms instead of push-based communication

## Key Features

1. **Secure Model Transfer**: Models are transferred via a shared mount directory that simulates a USB drive or secure file share
2. **Minimal Network Exposure**: Edge nodes can operate without direct network connectivity to the aggregator
3. **Pull-Based Updates**: Edge nodes fetch models at intervals rather than receiving pushed updates
4. **USB Mount Emulation**: The system simulates transferring models via physical media

## How It Works

The secure/air-gapped simulation works by:

1. Using a shared volume (`secure_transfer`) mounted to all containers
2. The aggregator exports models to this shared volume
3. Edge nodes periodically check for new models in the shared volume
4. Edge nodes train on their local data and export updates to the shared volume
5. The aggregator periodically checks for updates in the shared volume

## Configuration

The air-gapped mode is controlled by the `AIR_GAPPED_MODE` environment variable:

```bash
# Enable air-gapped mode
export AIR_GAPPED_MODE=true
docker-compose down
docker-compose up -d

# Disable air-gapped mode (normal operation)
export AIR_GAPPED_MODE=false
docker-compose down
docker-compose up -d
```

Additional configuration options:

- `SECURE_MOUNT_DIR`: Directory path for the secure mount (default: `/mnt/secure_transfer`)
- `SECURE_CHECK_INTERVAL`: Interval in seconds for checking the secure mount (default: 60)

## API Endpoints

### Aggregator Endpoints

- `POST /secure/export_model`: Manually export the global model to the secure mount
- `POST /secure/check_updates`: Check for model updates in the secure mount

### Edge Node Endpoints

- `GET /secure/check_models`: Check for new models in the secure mount
- `POST /secure/import_model`: Import a model from the secure mount
- `POST /secure/export_update`: Export a model update to the secure mount

## Running the Simulation

A demonstration script is provided to showcase the secure/air-gapped simulation:

```bash
./simulate_air_gapped.sh
```

This script will:

1. Start the system in air-gapped mode
2. Export a global model from the aggregator to the secure mount
3. Allow edge nodes to fetch models from the secure mount
4. Allow edge nodes to export model updates to the secure mount
5. Allow the aggregator to import model updates from the secure mount

## Security Considerations

In a real-world deployment, consider these additional security measures:

1. **Encryption**: Encrypt model files before transferring them
2. **Authentication**: Implement digital signatures to verify model authenticity
3. **Access Control**: Restrict access to the secure mount directory
4. **Audit Logging**: Log all model transfers for security auditing
5. **Media Sanitization**: Ensure proper sanitization of physical media after use

## Use Cases

This simulation is particularly useful for:

1. **Healthcare**: Hospitals with strict data privacy requirements
2. **Military/Government**: Classified environments with air-gapped networks
3. **Industrial IoT**: Factory floors with limited connectivity
4. **Critical Infrastructure**: Power plants, water treatment facilities, etc.
5. **Financial Services**: High-security trading environments
