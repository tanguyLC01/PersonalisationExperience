from typing import List, Tuple, Dict
from flwr.common import Metrics

from flwr.server.strategy import FedAvg
import os
import numpy as np
import flwr
import pickle

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
    

class FedAvgWithModelSaving(FedAvg):
    """FedAvg strategy that saves the global model after each round."""
    
    def __init__(self, save_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_model_path = save_path
        os.makedirs(self.global_model_path, exist_ok=True)
        
    def _save_global_model(self, server_round: int, parameters):
        ndarrays = flwr.common.parameters_to_ndarrays(parameters)
        data = {'global_parameters': ndarrays}
        filename = f'{self.global_model_path}/parameters_round_{server_round}.pkl'
        with open(filename, 'wb') as h:
            pickle.dump(data, h, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate(self, server_round: int, parameters: List[np.ndarray]) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the global model and save it."""
        # Call the original evaluate method        
        # Save the global model after evaluation
        self._save_global_model(server_round, parameters)
        
        # Call the original evaluate method        
        return super().evaluate(server_round, parameters)
