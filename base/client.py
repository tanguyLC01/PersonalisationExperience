from flwr.client import NumPyClient
from typing import List
import torch
from collections import OrderedDict
import numpy as np
from typing import Tuple, List, Dict, Union
from flwr.common import NDArrays, Scalar
from flwr.common import Context
import os
from base.model import ModelManager 
from flwr.common.logger import log
from logging import INFO


class BaseClient(NumPyClient):
    
    def __init__(self, partition_id: int, model_manager: type[ModelManager], config: Dict[str, Scalar]) -> None:
        self.partition_id = partition_id
        self.model_manager = model_manager
        self.epochs = config.client_config.num_epochs
        
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Return the current parameters of the global network."""
        return self.model_manager.model.get_parameters()
    
    def set_parameters(
        self, parameters: List[np.ndarray], evaluate: bool = False
    ) -> None:
        """Set the local_net model parameters to the received parameters.

        Args:
            parameters: parameters to set the model to.
        """
        
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_global_net") or k.startswith("_local_net")
        ]
        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model_manager.model.set_parameters(state_dict)
        
    def perform_train(
        self, verbose: bool = False
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Perform local_net training to the whole model.

        Returns
        -------
            Dict with the train metrics.
        """
        epochs = self.epochs

        self.model_manager.model.enable_global_net()
        self.model_manager.model.enable_local_net()

        return self.model_manager.train(
            epochs=epochs,  
        )
    
    def fit(self, parameters, config) -> List[np.ndarray]:
        """Train the network and return the updated parameters."""
        log(INFO, f"[Client {self.partition_id}] fit, config: {config}")
        self.set_parameters(parameters)
        train_results = self.perform_train()
        
        log(INFO, f"Training Results ------- Client {self.partition_id}")
        log(INFO, f'{train_results}')

        return self.get_parameters(config), self.model_manager.train_dataset_size(), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, dict]:
        """Evaluate the network and return the loss and accuracy."""
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        self.set_parameters(parameters, evaluate=True)
        loss, accuracy = self.model_manager.test().values()
        return float(loss), self.model_manager.test_dataset_size(), {"accuracy": float(accuracy)}

class PersonalizedClient(BaseClient):
    def __init__(self, partition_id: int, model_manager: type[ModelManager], config: Dict[str, Scalar]) -> None:
        super().__init__(partition_id, model_manager, config)  

    def get_parameters(self, config) -> List[np.ndarray]:
        """Return the current local_net global_net parameters."""
        return [
            val.cpu().numpy()
            for _, val in self.model_manager.model.global_net.state_dict().items()
        ]
        
    def set_parameters(self, parameters: List[np.ndarray], evaluate: bool =False) -> None:
        """Set the global_net parameters to the received parameters.

        Args:
            parameters: parameters to set the body to.
            evaluate: whether the client is evaluating or not.
        """
        global_model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_global_net")
        ]
        # Set global model
        params_dict = zip(global_model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model_manager.model.set_parameters(state_dict)
        
    
    
    
