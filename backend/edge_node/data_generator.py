import os
import numpy as np
import pandas as pd
import random
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_generator")

def generate_turbine_data(node_id, output_dir="data", n_samples=5000):
    """
    Generate simulated turbine time-series data for a specific edge node
    
    Args:
        node_id: ID of the node to generate data for
        output_dir: Directory to save the generated data
        n_samples: Number of data points to generate
    """
    logger.info(f"Generating simulated turbine data for node {node_id}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add some randomness to the number of samples for each node
    n_samples = n_samples + random.randint(-500, 500)
    
    # Time index
    start_date = datetime(2023, 1, 1)
    time_idx = [start_date + timedelta(minutes=10*i) for i in range(n_samples)]
    
    # Features: temperature, pressure, vibration, rotational_speed, power_output
    # Each node has slightly different data distributions to simulate different turbines
    base_temp = 60 + int(node_id) * 5  # Different base temperature per node
    
    # Generate data with some randomness
    temperature = np.random.normal(base_temp, 10, n_samples)
    pressure = np.random.normal(100, 15, n_samples)
    vibration = np.random.normal(0.5, 0.2, n_samples)
    rotational_speed = np.random.normal(1500, 200, n_samples)
    power_output = np.random.normal(800, 150, n_samples)
    
    # Add some seasonal patterns
    anomalies = []
    for i in range(n_samples):
        # Daily pattern
        hour = i % 24
        temperature[i] += 5 * np.sin(hour * np.pi / 12)
        power_output[i] += 50 * np.sin(hour * np.pi / 12)
        
        # Add some anomalies (5% of data)
        if random.random() < 0.05:
            vibration[i] *= 2.5
            temperature[i] += random.choice([-15, 15])
            anomaly = 1
        else:
            anomaly = 0
            
        anomalies.append(anomaly)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': time_idx,
        'temperature': temperature,
        'pressure': pressure,
        'vibration': vibration,
        'rotational_speed': rotational_speed,
        'power_output': power_output,
        'anomaly': anomalies
    })
    
    # Save to CSV
    output_file = os.path.join(output_dir, f"turbine_data_node_{node_id}.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"Generated {n_samples} data points for node {node_id}, saved to {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Generate data for each node
    for node_id in range(1, 4):
        generate_turbine_data(node_id)
