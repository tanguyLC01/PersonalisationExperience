from typing import List, Tuple, Dict, Callable, Optional, Union
from flwr.common import Metrics, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import os
import numpy as np
import flwr
import pickle
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
import logging
from logging import WARNING, INFO
from flwr.server.client_proxy import ClientProxy
from flwr.common import parameters_to_ndarrays, FitRes
from flwr.common.logger import log

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
            
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        coeff = []
        for _, res in results:
            log(INFO, res)
        
        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def evaluate(self, server_round: int, parameters: List[np.ndarray]) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the global model and save it."""    
        # Save the global model before evaluation
        self._save_global_model(server_round, parameters)
        
        # Call the original evaluate method        
        return super().evaluate(server_round, parameters)
