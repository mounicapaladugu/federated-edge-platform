import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import json
import random
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_trainer")

class AnomalyDetectionModel(nn.Module):
    """Simple LSTM-based model for anomaly detection on turbine data"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(AnomalyDetectionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        return out

class ModelTrainer:
    """Class to handle model training on edge nodes"""
    def __init__(self, node_id):
        self.node_id = node_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create data directory if it doesn't exist
        os.makedirs(f"data", exist_ok=True)
        
        # Generate simulated data if it doesn't exist
        self.data_file = f"data/turbine_data_node_{node_id}.csv"
        if not os.path.exists(self.data_file):
            self._generate_simulated_data()
    
    def _generate_simulated_data(self):
        """Generate simulated turbine data for training"""
        logger.info(f"Generating simulated turbine data for node {self.node_id}")
        
        # Number of data points
        n_samples = 5000 + random.randint(-500, 500)  # Slightly different dataset sizes
        
        # Time index
        time_idx = pd.date_range(start='2023-01-01', periods=n_samples, freq='10min')
        
        # Features: temperature, pressure, vibration, rotational_speed, power_output
        # Each node has slightly different data distributions to simulate different turbines
        base_temp = 60 + int(self.node_id) * 5  # Different base temperature per node
        
        # Generate data with some randomness
        temperature = np.random.normal(base_temp, 10, n_samples)
        pressure = np.random.normal(100, 15, n_samples)
        vibration = np.random.normal(0.5, 0.2, n_samples)
        rotational_speed = np.random.normal(1500, 200, n_samples)
        power_output = np.random.normal(800, 150, n_samples)
        
        # Add some seasonal patterns
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
                
            # Store anomaly label
            if i == 0:
                anomalies = [anomaly]
            else:
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
        df.to_csv(self.data_file, index=False)
        logger.info(f"Generated {n_samples} data points for node {self.node_id}")
    
    def _prepare_data(self):
        """Prepare data for training"""
        # Load data
        df = pd.read_csv(self.data_file)
        
        # Extract features and target
        features = df[['temperature', 'pressure', 'vibration', 'rotational_speed', 'power_output']].values
        targets = df['anomaly'].values
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Convert to PyTorch tensors
        X = torch.tensor(features_normalized, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        
        # Split into train/test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, test_loader, len(features[0])
    
    def train(self, model_path):
        """Train the model using local data"""
        # Prepare data
        train_loader, test_loader, input_size = self._prepare_data()
        
        # Load or initialize model
        if os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            model = AnomalyDetectionModel(input_size=input_size)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.info("Initializing new model")
            model = AnomalyDetectionModel(input_size=input_size)
        
        model = model.to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 5  # Limited number of epochs for simulation
        model.train()
        
        logger.info(f"Starting training on node {self.node_id}")
        
        training_losses = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(inputs.unsqueeze(1))  # Add sequence dimension
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            training_losses.append(avg_loss)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs.unsqueeze(1))
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        logger.info(f"Test Accuracy: {accuracy:.2f}%")
        
        # Save the updated model
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Return metrics
        return {
            "accuracy": accuracy,
            "loss": avg_test_loss,
            "training_loss": training_losses[-1] if training_losses else None
        }
