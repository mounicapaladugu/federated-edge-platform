import flwr as fl
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg, FedProx

logger = logging.getLogger("fl_strategy")

class FedAvgWithModelVersioning(FedAvg):
    """
    FedAvg strategy with model versioning and rollback capabilities.
    """
    def __init__(
        self,
        *args,
        mu: float = 0.01,  # Proximal term weight for FedProx
        use_fedprox: bool = False,  # Whether to use FedProx instead of FedAvg
        privacy_setting: Dict = None,  # Privacy settings (DP noise, clipping)
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.use_fedprox = use_fedprox
        self.privacy_setting = privacy_setting or {
            "enabled": False,
            "noise_multiplier": 0.1,
            "max_grad_norm": 1.0
        }
        
        # Model version history for rollback capability
        self.model_history = []
        self.current_version = 0
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model updates using FedAvg or FedProx with privacy mechanisms."""
        
        # Apply differential privacy if enabled
        if self.privacy_setting["enabled"]:
            results = self._apply_differential_privacy(results)
        
        # Use FedProx if specified, otherwise use standard FedAvg
        if self.use_fedprox:
            aggregated_params = self._aggregate_fedprox(server_round, results, failures)
        else:
            # Use parent class (FedAvg) aggregation
            aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Store model version for rollback capability
        if aggregated_params is not None:
            # Convert parameters to numpy arrays for storage
            model_weights = parameters_to_ndarrays(aggregated_params)
            
            # Store model version
            self.model_history.append({
                "version": self.current_version + 1,
                "round": server_round,
                "weights": model_weights,
                "metrics": metrics if 'metrics' in locals() else {}
            })
            
            self.current_version += 1
            logger.info(f"Stored model version {self.current_version} from round {server_round}")
            
            # Keep only the last 5 versions to save memory
            if len(self.model_history) > 5:
                self.model_history.pop(0)
        
        return aggregated_params, {"version": self.current_version} if 'metrics' not in locals() else {**metrics, "version": self.current_version}
    
    def _aggregate_fedprox(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate model updates using FedProx."""
        # Implementation of FedProx aggregation
        # This is a simplified version - in a real implementation, the proximal term
        # would be applied during client training
        
        # Fallback to standard FedAvg for now
        return super().aggregate_fit(server_round, results, failures)
    
    def _apply_differential_privacy(
        self,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]]:
        """Apply differential privacy to model updates."""
        noise_multiplier = self.privacy_setting["noise_multiplier"]
        max_grad_norm = self.privacy_setting["max_grad_norm"]
        
        # Process each client's update
        processed_results = []
        for client, fit_res in results:
            # Get model parameters
            parameters = fit_res.parameters
            weights = parameters_to_ndarrays(parameters)
            
            # Apply weight clipping (L2 norm)
            weights_clipped = []
            for w in weights:
                norm = np.linalg.norm(w)
                if norm > max_grad_norm:
                    w = w * (max_grad_norm / norm)
                weights_clipped.append(w)
            
            # Add Gaussian noise
            weights_with_noise = []
            for w in weights_clipped:
                noise = np.random.normal(0, noise_multiplier, w.shape)
                w_noisy = w + noise
                weights_with_noise.append(w_noisy)
            
            # Convert back to parameters
            noisy_parameters = ndarrays_to_parameters(weights_with_noise)
            
            # Create new FitRes with noisy parameters
            noisy_fit_res = FitRes(
                parameters=noisy_parameters,
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
            )
            
            processed_results.append((client, noisy_fit_res))
        
        return processed_results
    
    def get_model_version(self, version: int = None) -> Optional[Dict]:
        """Get a specific model version or the latest if version is None."""
        if not self.model_history:
            return None
        
        if version is None:
            # Return the latest version
            return self.model_history[-1]
        
        # Find the requested version
        for model_version in self.model_history:
            if model_version["version"] == version:
                return model_version
        
        return None
    
    def rollback_to_version(self, version: int) -> bool:
        """Roll back to a specific model version."""
        model_version = self.get_model_version(version)
        if model_version is None:
            logger.error(f"Model version {version} not found")
            return False
        
        # Set as current version
        self.current_version = model_version["version"]
        logger.info(f"Rolled back to model version {version}")
        return True
